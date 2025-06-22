# modified_densenet_fullpipeline.py
# End-to-end implementation of the "Modified DenseNet-121" pipeline on CBIS-DDSM,
# including patient-level stratified splits, DICOM preprocessing, skip corrupted images,
# DenseNet-121 fine-tuning (RGB input), and evaluation on MPS device.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
from PIL import Image

# -----------------------------
# DATASET DEFINITION
# -----------------------------
class MammogramDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row['full_path'])
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            return None
        # normalize
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        # grayscale to RGB
        img_pil = Image.fromarray((arr * 255).astype(np.uint8))
        img = Image.merge('RGB', (img_pil, img_pil, img_pil))
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label

# custom collate to skip None samples
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
def main():
    # ---------- CONFIGURATION ----------
    DATA_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
    CSV_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_with_correct_roi.csv"
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_EPOCHS = 100
    L1_LAMBDA = 1e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10

    DEVICE = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
    print("Using device:", DEVICE)

    # ---------- LOAD & SPLIT METADATA ----------
    df = pd.read_csv(CSV_PATH)
    df['path_lower'] = df['full_path'].str.lower()

    # test set
    test_df = df[df['path_lower'].str.contains('test_')].copy().reset_index(drop=True)
    # train+val pool
    train_val_df = df[df['path_lower'].str.contains('training_')].copy().reset_index(drop=True)

    # cleanup
    for d in (df, train_val_df, test_df):
        if 'path_lower' in d:
            d.drop(columns=['path_lower'], inplace=True)

    # case-insensitive pathology
    for d in (train_val_df, test_df):
        d['pathology_lower'] = d['pathology'].str.lower().str.strip()
        d['label'] = d['pathology_lower']
        d['label'] = d['label'].replace({'benign without callback':'benign'})
        d['label'] = d['label'].map({'benign':0,'malignant':1})
        d.dropna(subset=['label'], inplace=True)
        d['label'] = d['label'].astype(int)

    # stratify groups
    groups_df = train_val_df.groupby('patient_id')[['label','breast_density']].first().reset_index()
    groups_df['strat_key'] = groups_df['label'].astype(str) + '_' + groups_df['breast_density'].astype(str)
    val_frac = 15/85
    train_pats, val_pats = train_test_split(groups_df['patient_id'], test_size=val_frac,
                                            stratify=groups_df['strat_key'], random_state=42)
    train_df = train_val_df[train_val_df['patient_id'].isin(train_pats)].reset_index(drop=True)
    val_df = train_val_df[train_val_df['patient_id'].isin(val_pats)].reset_index(drop=True)

    # ---------- TRANSFORMS & DATALOADERS ----------
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_loader = DataLoader(MammogramDataset(train_df, DATA_DIR, train_tf), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(MammogramDataset(val_df, DATA_DIR, eval_tf), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(MammogramDataset(test_df, DATA_DIR, eval_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---------- MODEL DEFINITION ----------
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f,2048), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(2048,512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512,2)
    )
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---------- TRAIN & EARLY STOPPING ----------
    best_acc, trials = 0.0, 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train(); rloss=0.0
        for imgs,labels in train_loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            out=model(imgs)
            loss=criterion(out,labels)+L1_LAMBDA*sum(p.abs().sum() for p in model.parameters())
            loss.backward(); optimizer.step()
            rloss+=loss.item()*imgs.size(0)
        rloss/=len(train_loader.dataset)

        model.eval(); preds, truths = [],[]
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                p=model(imgs).argmax(1).cpu().numpy()
                preds.extend(p); truths.extend(labels.cpu().numpy())
        vacc=accuracy_score(truths,preds)
        print(f"Epoch {epoch}: loss={rloss:.4f} val_acc={vacc:.4f}")
        if vacc>best_acc: best_acc,vacc_trials=vacc,0; torch.save(model.state_dict(),"densenet121_modified.pth")
        else: trials+=1
        if trials>=PATIENCE: print("Early stopping."); break

    # ---------- TEST & METRICS ----------
    model.load_state_dict(torch.load("densenet121_modified.pth")); model.eval()
    yt,yp,ys=[],[],[]
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            logits=model(imgs); probs=torch.nn.functional.softmax(logits,1)[:,1].cpu().numpy()
            pred=logits.argmax(1).cpu().numpy()
            yt.extend(labels.cpu().numpy()); yp.extend(pred); ys.extend(probs)
    print("Test: acc={:.4f}, prec={:.4f}, rec={:.4f}, f1={:.4f}, auc={:.4f}".format(
        accuracy_score(yt,yp), precision_score(yt,yp), recall_score(yt,yp), f1_score(yt,yp), roc_auc_score(yt,ys)
    ))

if __name__=="__main__":
    from multiprocessing import freeze_support; freeze_support(); main()