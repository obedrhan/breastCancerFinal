import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTForImageClassification

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "cropped_deep_learning/models/vit_cropped.pth")
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.0001

# === Dataset for ViT ===
class CroppedMammoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform
        self.dataframe["pathology"] = self.dataframe["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
        self.dataframe["label"] = self.dataframe["pathology"].map({"BENIGN": 0, "MALIGNANT": 1})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]

        try:
            dicom = pydicom.dcmread(image_path, force=True)
            if 'PixelData' not in dicom:
                raise ValueError("Missing PixelData")
            img_array = dicom.pixel_array.astype(np.float32)

            img_array -= img_array.min()
            img_array /= (img_array.max() + 1e-6)
            img_array *= 255.0
            image = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")  # ViT expects RGB
        except Exception as e:
            new_idx = (idx + 1) % len(self.dataframe)
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)
        return image, label

# === ViT-specific transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Load metadata and filter for cropped training samples ===
df = pd.read_csv(CSV_PATH)
df = df[(df["label"].str.lower() == "cropped") & (df["image_path"].str.contains("Training"))].reset_index(drop=True)

# === DataLoader ===
dataset = CroppedMammoDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# === Load ViT model from HuggingFace ===
def get_vit_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    return model

# === Training Loop ===
def train_model(model, dataloader, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔁 Epoch [{epoch+1}/{NUM_EPOCHS}]")
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs.logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f" 🔄 Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total * 100
        print(f"✅ Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    return model

# === Run Training ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_model()
    trained_model = train_model(model, dataloader, device)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")

