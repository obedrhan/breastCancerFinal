import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torchvision.transforms as transforms

# === CONFIG ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
MODEL_PATH = os.path.join(BASE_DIR, "cropped_deep_learning/models/vit_cropped.pth")
BATCH_SIZE = 16

# === Dataset ===
class CroppedMammoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.copy()
        self.transform = transform
        self.df["pathology"] = self.df["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
        self.df["label"] = self.df["pathology"].map({"BENIGN": 0, "MALIGNANT": 1})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']

        try:
            dicom = pydicom.dcmread(img_path, force=True)
            img_array = dicom.pixel_array.astype(np.float32)
        except Exception:
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)

        # Normalize to [0, 255] and convert to RGB
        img_array -= img_array.min()
        img_array /= (img_array.max() + 1e-6)
        img_array *= 255
        image = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# === Transform for ViT ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# === Load filtered DataFrame ===
df = pd.read_csv(CSV_PATH)
df = df[(df["label"].str.lower() == "cropped") & (df["image_path"].str.contains("Test"))].reset_index(drop=True)

# === Dataset and Dataloader
dataset = CroppedMammoDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Evaluation
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# === Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")