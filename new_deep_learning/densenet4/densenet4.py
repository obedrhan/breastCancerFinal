#!/usr/bin/env python3
"""
modified_densenet_improved_kfold.py

Enhanced DenseNet-121 5-fold CV pipeline on cropped ROIs with:
 - lesion-focused augmentations
 - two-stage freeze (head) → unfreeze (full) training
 - class-weighted CrossEntropyLoss
 - ReduceLROnPlateau scheduler
 - test-time ensembling of 5 fold models
 - proper multiprocessing guard for macOS
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data._utils.collate import default_collate

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH     = os.path.join(DATA_DIR, "data/final_cropped_full.csv")
BATCH_SIZE   = 32
LR_HEAD      = 1e-3
LR_FINETUNE  = 1e-5
EPOCHS_HEAD  = 5
EPOCHS_FULL  = 20
PATIENCE     = 3
WEIGHT_DECAY = 1e-4
NUM_FOLDS    = 5
DEVICE       = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# custom collate

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

# -----------------------------
# DATASET DEFINITON
# -----------------------------
class CroppedDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row['full_path_x'])
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            return None
        if arr.max()>arr.min():
            arr = (arr - arr.min())/(arr.max()-arr.min())
        else:
            arr = np.zeros_like(arr)
        img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label

# -----------------------------
# MODEL DEFINITION
# -----------------------------

def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f,2048), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(2048,512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512,2)
    )
    return model

# -----------------------------
# TRAIN & VALIDATION FUNCTION
# -----------------------------

def train_fold(train_df, val_df, fold_idx):
    # compute class weights
    labels = train_df['label'].dropna().astype(int).values
    counts = np.bincount(labels)
    class_weights = torch.tensor([1/counts[0], 1/counts[1]], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    train_loader = DataLoader(CroppedDataset(train_df, DATA_DIR, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(CroppedDataset(val_df,   DATA_DIR, eval_tf),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    model = build_model().to(DEVICE)

    # Stage 1: freeze features
    for p in model.features.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                                 lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    best_auc=0
    for e in range(1, EPOCHS_HEAD+1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        # validate
        model.eval()
        y_true,y_score=[],[]
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs=imgs.to(DEVICE)
                probs=torch.softmax(model(imgs),1)[:,1].cpu().numpy()
                y_score.extend(probs); y_true.extend(labels.numpy())
        auc=roc_auc_score(y_true,y_score)
        print(f"Fold{fold_idx} Stage1 Epoch{e} Val AUC={auc:.4f}")
        if auc>best_auc:
            best_auc=auc; torch.save(model.state_dict(),f"fold{fold_idx}_stage1.pth")

    # Stage 2: unfreeze all
    model.load_state_dict(torch.load(f"fold{fold_idx}_stage1.pth"))
    for p in model.parameters(): p.requires_grad=True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        mode='max', factor=0.5, patience=PATIENCE)
    best_auc=0; patience_cnt=0
    for e in range(1, EPOCHS_FULL+1):
        model.train()
        for imgs,labels in train_loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad(); out=model(imgs)
            loss=criterion(out,labels)
            loss.backward(); optimizer.step()
        # validate
        model.eval(); y_true,y_score=[],[]
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs=imgs.to(DEVICE)
                probs=torch.softmax(model(imgs),1)[:,1].cpu().numpy()
                y_score.extend(probs); y_true.extend(labels.numpy())
        auc=roc_auc_score(y_true,y_score)
        scheduler.step(auc)
        print(f"Fold{fold_idx} Stage2 Epoch{e} Val AUC={auc:.4f}")
        if auc>best_auc:
            best_auc=auc; patience_cnt=0
            torch.save(model.state_dict(),f"fold{fold_idx}_best.pth")
        else:
            patience_cnt+=1
            if patience_cnt>=PATIENCE: break
    return f"fold{fold_idx}_best.pth"

# -----------------------------
# MAIN EXECUTION
# -----------------------------

def main():
    df = pd.read_csv(CSV_PATH)
    # encode labels
    df['path_low']=df['pathology_x'].str.lower().str.strip()
    df['label']=df['path_low'].replace({'benign without callback':'benign'})
    df['label']=df['label'].map({'benign':0,'malignant':1})
    df=df.dropna(subset=['label'])
    # only training
    df_trainpool = df[df['full_path_x'].str.contains('training_',case=False)]

    groups = df_trainpool['patient_id'].values
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    fold_models=[]
    for i,(train_idx,val_idx) in enumerate(gkf.split(df_trainpool,df_trainpool['label'],groups),1):
        train_df=df_trainpool.iloc[train_idx]
        val_df  =df_trainpool.iloc[val_idx]
        mpth = train_fold(train_df,val_df,i)
        fold_models.append(mpth)

    # test set
    df_test = df[df['full_path_x'].str.contains('test_',case=False)].reset_index(drop=True)
    test_tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    test_loader=DataLoader(CroppedDataset(df_test,DATA_DIR,test_tf),
                           batch_size=BATCH_SIZE,shuffle=False,
                           num_workers=4,collate_fn=collate_fn)

    # ensemble
    models_list=[]
    for mp in fold_models:
        m=build_model().to(DEVICE)
        m.load_state_dict(torch.load(mp))
        m.eval(); models_list.append(m)

    y_true,y_pred,y_score=[],[],[]
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs=imgs.to(DEVICE)
            # avg probabilities
            probs=[torch.softmax(m(imgs),1)[:,1] for m in models_list]
            avgp=torch.stack(probs).mean(0).cpu().numpy()
            preds=(avgp>=0.5).astype(int)
            y_true.extend(labels.numpy()); y_pred.extend(preds); y_score.extend(avgp)

    print("\nEnsembled Test Performance:")
    print(f" AUC:      {roc_auc_score(y_true,y_score):.4f}")
    print(f" Accuracy: {accuracy_score(y_true,y_pred):.4f}")
    print(f" Precision:{precision_score(y_true,y_pred):.4f}")
    print(f" Recall:   {recall_score(y_true,y_pred):.4f}")
    print(f" F1 Score: {f1_score(y_true,y_pred):.4f}")

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
