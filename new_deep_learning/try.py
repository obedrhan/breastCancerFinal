# ensemble_inference.py
# Script to ensemble your 5 CV-trained DenseNet-121 models on the test set

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms

# -----------------------------
# CONFIGURATION (match your previous settings)
# -----------------------------
DATA_DIR     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/final_cropped_full.csv"
BATCH_SIZE   = 32
NUM_FOLDS    = 5
DEVICE       = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# -----------------------------
# DATASET + DATALOADER for test set
# -----------------------------
class CroppedDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['full_path_x']
        ds = pydicom.dcmread(f"{self.data_dir}/{path}")
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min()) if arr.max()!=arr.min() else arr*0
        img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])
        return img, label

# transforms must match your eval_tf
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# load full CSV, filter test rows and labels
full_df = pd.read_csv(CSV_PATH)
full_df['path_low'] = full_df['pathology_x'].str.lower().str.strip()
full_df['label'] = full_df['path_low'].replace({'benign without callback':'benign'}) \
                                        .map({'benign':0,'malignant':1})
full_df = full_df.dropna(subset=['label']).copy()
full_df['label'] = full_df['label'].astype(int)
test_df = full_df[ full_df['full_path_x'].str.lower().str.contains('test_') ].reset_index(drop=True)

test_loader = DataLoader(
    CroppedDataset(test_df, DATA_DIR, eval_tf),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# BUILD & LOAD ENSEMBLE MODELS
# -----------------------------
ensemble_models = []
for fold in range(1, NUM_FOLDS+1):
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f,2048), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(2048,512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512,2)
    )
    ckpt = torch.load(f"best_model_fold{fold}.pth", map_location=DEVICE)
    m.load_state_dict(ckpt)
    m.to(DEVICE).eval()
    ensemble_models.append(m)

# -----------------------------
# ENSEMBLE INFERENCE
# -----------------------------
y_true, y_pred, y_score = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        # stack each model's malignant probability
        probs = torch.stack([torch.softmax(model(imgs), dim=1)[:,1] for model in ensemble_models], dim=0)
        avg_prob = probs.mean(dim=0).cpu().numpy()
        preds = (avg_prob >= 0.5).astype(int)

        y_true.extend(labels)
        y_pred.extend(preds)
        y_score.extend(avg_prob)

# -----------------------------
# METRICS
# -----------------------------
test_auc = roc_auc_score(y_true, y_score)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEnsembled Test Set Performance:")
print(f"  AUC:      {test_auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision:{prec:.4f}")
print(f"  Recall:   {rec:.4f}")
print(f"  F1 Score: {f1:.4f}")