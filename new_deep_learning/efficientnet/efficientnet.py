#!/usr/bin/env python3
"""
efficientnet_lstm_main.py

End-to-end implementation with proper __main__ guard for:
- Multi-view EfficientNet-B0 + BiLSTM pipeline (Lilhore et al., 2025)
- Handles multiprocessing on macOS with freeze_support
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data._utils.collate import default_collate

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH   = os.path.join(DATA_DIR, "data/final_cropped_full.csv")  # expects columns: patient_id, view, full_path, pathology
BATCH_SIZE = 16
LR         = 1e-4
NUM_EPOCHS = 50
PATIENCE   = 7
DEVICE     = torch.device('mps') if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device('cpu')

# custom collate to drop None
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

# -----------------------------
# DATASET
# -----------------------------
class MultiViewDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        # pivot multi-view per patient
        pivot = df.pivot_table(index='patient_id', columns='view',
                               values='full_path_x', aggfunc='first').reset_index()
        pivot['path_low'] = df.groupby('patient_id')['pathology_x'] \
                              .first().str.lower().str.strip().values
        pivot['label'] = pivot['path_low'] \
            .replace({'benign without callback':'benign'}) \
            .map({'benign':0,'malignant':1})
        pivot.dropna(subset=['label', 'CC', 'MLO'], inplace=True)
        self.df = pivot.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        for view in ['CC', 'MLO']:
            path = os.path.join(self.data_dir, row[view])
            try:
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
            except Exception:
                return None
            arr = (arr - arr.min())/(arr.max()-arr.min()) if arr.max()>arr.min() else arr*0
            img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # shape [2, C, H, W]
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return imgs, label

# -----------------------------
# MODEL
# -----------------------------
class EfficientNetLSTM(nn.Module):
    def __init__(self, hidden_size=512, num_layers=1, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.lstm = nn.LSTM(input_size=in_feats,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [B, T=2, C, H, W]
        B, T, C, H, W = x.size()
        xt = x.view(B*T, C, H, W)
        feats = self.backbone(xt)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# -----------------------------
# TRAIN & EVAL
# -----------------------------
def main():
    print("Using device:", DEVICE)
    # load metadata
    df_all = pd.read_csv(CSV_PATH)
    patients = df_all['patient_id'].unique()
    train_pats, test_pats = train_test_split(patients, test_size=0.15, random_state=42)
    train_pats, val_pats = train_test_split(train_pats, test_size=0.1765, random_state=42)

    train_df = df_all[df_all['patient_id'].isin(train_pats)]
    val_df   = df_all[df_all['patient_id'].isin(val_pats)]
    test_df  = df_all[df_all['patient_id'].isin(test_pats)]

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_loader = DataLoader(
        MultiViewDataset(train_df, DATA_DIR, transform=train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        MultiViewDataset(val_df,   DATA_DIR, transform=eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        MultiViewDataset(test_df,  DATA_DIR, transform=eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    model = EfficientNetLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc, patience = 0.0, 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_scores.extend(probs)
                all_labels.extend(labels.numpy())
        val_auc = roc_auc_score(all_labels, all_scores)
        print(f"Epoch {epoch}: Val AUC = {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, patience = val_auc, 0
            torch.save(model.state_dict(), "best_efficientnet_lstm.pth")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break

    # final test
    model.load_state_dict(torch.load("best_efficientnet_lstm.pth"))
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_score.extend(probs)
    print("\nTest Performance:")
    print(f" AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f" Acc: {accuracy_score(y_true, y_pred):.4f}")
    print(f" Prec: {precision_score(y_true, y_pred):.4f}")
    print(f" Rec: {recall_score(y_true, y_pred):.4f}")
    print(f" F1: {f1_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()