#!/usr/bin/env python3
"""
ensemble_fold2_stage1_2.py

Ensemble Fold 2 Stage-1 and Stage-2 DenseNet-121 models by averaging their softmax probabilities.
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data._utils.collate import default_collate

# CONFIG
DATA_DIR    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH    = os.path.join(DATA_DIR, "data/final_cropped_full.csv")
CKPT1       = "fold2_stage1.pth"
CKPT2       = "fold2_best.pth"
BATCH_SIZE  = 32
DEVICE      = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# custom collate to skip corrupted entries
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

# Dataset identical to your training script
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
        except:
            return None
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.zeros_like(arr)
        img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label

# rebuild your modified DenseNet-121
def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f, 2048), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(2048, 512),  nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    return model

# inference
def main():
    print(f"Loading models onto {DEVICE}")
    m1 = build_model().to(DEVICE)
    m2 = build_model().to(DEVICE)
    m1.load_state_dict(torch.load(CKPT1, map_location=DEVICE))
    m2.load_state_dict(torch.load(CKPT2, map_location=DEVICE))
    m1.eval(); m2.eval()

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # load test set
    df = pd.read_csv(CSV_PATH)
    df['path_low'] = df['pathology_x'].str.lower().str.strip()
    df['label'] = df['path_low'].replace({'benign without callback':'benign'}).map({'benign':0,'malignant':1})
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    test_df = df[df['full_path_x'].str.lower().str.contains('test_')].reset_index(drop=True)

    loader = DataLoader(
        CroppedDataset(test_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            # get softmax probs from both models
            p1 = torch.softmax(m1(imgs), 1)[:,1]
            p2 = torch.softmax(m2(imgs), 1)[:,1]
            # average
            avgp = ((p1 + p2) / 2.0).cpu().numpy()

            preds = (avgp >= 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_score.extend(avgp)

    # metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_score)

    print("\nEnsemble Fold2 Stage1+2 Test Performance:")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision:{prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()