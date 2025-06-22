#!/usr/bin/env python3
"""
train_resnet50_cropped.py

ResNet-50 full-image classifier on CBIS-DDSM cropped ROIs,
using your final_cropped_full.csv and best-practice pipeline.
"""

import os, random, cv2
import pandas as pd, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ─── CONFIG ─────────────────────────────────────────────────────────
BASE_DIR     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH     = os.path.join(BASE_DIR, "data/final_cropped_full.csv")
CKPT_PATH    = os.path.join(BASE_DIR, "new_deep_learning/resnet/best_resnet50.pth")
BATCH_SIZE   = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS   = 50
PATIENCE     = 5
VAL_FRAC     = 0.15
DEVICE       = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SEED         = 42

# ─── REPRO ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── DATASET ─────────────────────────────────────────────────────────
class CroppedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # map pathology → benign/malignant → 0/1
        self.df['path_low'] = (
            self.df['pathology_x']
              .str.lower().str.strip()
              .replace({'benign without callback':'benign'})
        )
        self.df['label'] = self.df['path_low'].map({'benign':0,'malignant':1})
        self.df.dropna(subset=['label'], inplace=True)
        self.df['label'] = self.df['label'].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = os.path.join(BASE_DIR, row['full_path_x'])
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        # normalize to 0–255
        arr = np.uint8(255 * (arr - arr.min())/(arr.max()-arr.min()+1e-6))
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        arr = clahe.apply(arr)

        img = Image.fromarray(arr).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row['label'], dtype=torch.long)
        return img, label

# ─── COLLATE ─────────────────────────────────────────────────────────
from torch.utils.data._utils.collate import default_collate
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

# ─── MODEL BUILD ─────────────────────────────────────────────────────
def build_model():
    backbone = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    return nn.Sequential(backbone, head).to(DEVICE)

# ─── TRANSFORMS ─────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ─── READ & SPLIT ────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# test = any path containing "Test_", case-insensitive
test_df = df[df['full_path_x'].str.contains('test_', case=False)].reset_index(drop=True)
# train+val pool = "Training_"
tv_df   = df[df['full_path_x'].str.contains('training_', case=False)].reset_index(drop=True)

# patient-level val split inside train+val
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
train_idx, val_idx = next(gss.split(tv_df, groups=tv_df['patient_id']))
train_df = tv_df.iloc[train_idx].reset_index(drop=True)
val_df   = tv_df.iloc[val_idx].reset_index(drop=True)

# ─── DATALOADERS ────────────────────────────────────────────────────
train_ds = CroppedDataset(train_df, transform=train_tf)
val_ds   = CroppedDataset(val_df,   transform=eval_tf)
test_ds  = CroppedDataset(test_df,  transform=eval_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=collate_fn
)

# ─── SETUP ───────────────────────────────────────────────────────────
model     = build_model()
# class weights
lbls     = train_ds.df['label'].values
counts   = np.bincount(lbls)
class_w  = torch.tensor([1/counts[0], 1/counts[1]],
                        dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

best_auc, patience_cnt = 0.0, 0

# ─── TRAIN & VALIDATE ───────────────────────────────────────────────
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    y_true, y_pred = [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.cpu().numpy())

    train_acc = accuracy_score(y_true, y_pred)

    model.eval()
    v_true, v_score = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            v_score.extend(probs)
            v_true.extend(labels.numpy())

    val_auc = roc_auc_score(v_true, v_score)
    scheduler.step(val_auc)

    print(f"Epoch {epoch:02d}  Train Acc: {train_acc:.4f}  Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc, patience_cnt = val_auc, 0
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        torch.save(model.state_dict(), CKPT_PATH)
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping.")
            break

print(f"\nBest Val AUC: {best_auc:.4f}")

# ─── FINAL TEST EVALUATION ────────────────────────────────────────────
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

t_true, t_pred, t_score = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        score  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        pred   = logits.argmax(1).cpu().numpy()
        t_score.extend(score)
        t_true.extend(labels.numpy())
        t_pred.extend(pred)

test_acc  = accuracy_score(t_true, t_pred)
test_prec = precision_score(t_true, t_pred)
test_rec  = recall_score(t_true, t_pred)
test_f1   = f1_score(t_true, t_pred)
test_auc  = roc_auc_score(t_true, t_score)

print("\nTest Performance:")
print(f"  AUC:      {test_auc:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Precision:{test_prec:.4f}")
print(f"  Recall:   {test_rec:.4f}")
print(f"  F1 Score: {test_f1:.4f}")