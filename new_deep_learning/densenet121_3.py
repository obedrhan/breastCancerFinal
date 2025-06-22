# modified_densenet_kfold.py
# End-to-end Modified DenseNet-121 pipeline on cropped ROIs with 5-fold patient-level CV,
# two-stage freeze/fine-tune, and gradient clipping

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
from PIL import Image

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/final_cropped_full.csv"
BATCH_SIZE   = 32
LR_HEAD      = 1e-4
LR_FINETUNE  = 1e-5
EPOCHS_HEAD  = 10
EPOCHS_FULL  = 20
PATIENCE     = 5
WEIGHT_DECAY = 1e-4
NUM_FOLDS    = 5
DEVICE       = torch.device('mps') if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device('cpu')
print("Using device:", DEVICE)

# -----------------------------
# DATASET DEFINITION
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
        # use the cropped DICOM path from 'full_path_x'
        path = os.path.join(self.data_dir, row['full_path_x'])
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min()) if arr.max()!=arr.min() else arr*0
        img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label

# -----------------------------
# TRANSFORMS
# -----------------------------
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# LOAD AND PREPARE DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df['path_low'] = df['pathology_x'].str.lower().str.strip()
df['label'] = df['path_low'].replace({'benign without callback':'benign'}).map({'benign':0,'malignant':1})
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)
# use only training rows
# use only training rows based on cropped DICOM paths in 'full_path_x'
df = df[df['full_path_x'].str.lower().str.contains('training_')]

# -----------------------------
# GROUP K-FOLD CV
# -----------------------------
groups = df['patient_id'].values
gkf = GroupKFold(n_splits=NUM_FOLDS)

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups), 1):
    print(f"\n--- Fold {fold} ---")
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    train_loader = DataLoader(CroppedDataset(train_df, DATA_DIR, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(CroppedDataset(val_df,   DATA_DIR, eval_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
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

    # 1) Train head only
    for p in model.features.parameters(): p.requires_grad = False
    optimizer_head = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                      lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS_HEAD+1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer_head.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_head.step()
        print(f"Head Epoch {epoch} done")

    # 2) Fine-tune all layers
    for p in model.parameters(): p.requires_grad = True
    optimizer_full = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    best_val_auc, patience = 0.0, 0
    for epoch in range(1, EPOCHS_FULL+1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer_full.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_full.step()
        # Validation
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
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping on full fine-tune")
                break
    fold_metrics.append(best_val_auc)
    print(f"Fold {fold} best AUC: {best_val_auc:.4f}")

# Summary of CV
avg_auc = np.mean(fold_metrics)
print(f"{NUM_FOLDS}-Fold CV average AUC: {avg_auc:.4f}")

# -----------------------------
# FINAL TEST EVALUATION
# -----------------------------
# Load test set from full CSV
full_df = pd.read_csv(CSV_PATH)
full_df['path_low'] = full_df['pathology_x'].str.lower().str.strip()
full_df['label'] = full_df['path_low'].replace({'benign without callback':'benign'}).map({'benign':0,'malignant':1})
full_df = full_df.dropna(subset=['label']).copy()
full_df['label'] = full_df['label'].astype(int)
# Filter test rows
test_df = full_df[ full_df['full_path_x'].str.lower().str.contains('test_') ].reset_index(drop=True)

# DataLoader for test
test_loader = DataLoader(
    CroppedDataset(test_df, DATA_DIR, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Select best fold model
best_fold = int(np.argmax(fold_metrics)) + 1
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
in_f = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.BatchNorm1d(in_f),
    nn.Linear(in_f,2048), nn.ReLU(True), nn.Dropout(0.5),
    nn.Linear(2048,512), nn.ReLU(True), nn.Dropout(0.5),
    nn.Linear(512,2)
)
model.load_state_dict(torch.load(f"best_model_fold{best_fold}.pth"))
model.to(DEVICE).eval()

# Evaluate
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

test_auc = roc_auc_score(y_true, y_score)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Test Set Performance using best fold model:")
print(f"  AUC: {test_auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall: {rec:.4f}")
print(f"  F1 Score: {f1:.4f}")
