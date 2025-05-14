import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# === CONFIG ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
MODEL_PATH = os.path.join(BASE_DIR, "cropped_deep_learning/models/custom_cnn_cropped.pth")

# === Dataset ===
class CroppedMammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform
        self.dataframe['pathology'] = self.dataframe['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        self.dataframe['label'] = self.dataframe['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        try:
            dicom = pydicom.dcmread(img_path, force=True)
            if 'PixelData' not in dicom:
                raise ValueError("Missing PixelData")
            img_array = dicom.pixel_array.astype(np.float32)
        except Exception:
            new_idx = (idx + 1) % len(self.dataframe)
            return self.__getitem__(new_idx)

        img_array -= img_array.min()
        img_array /= (img_array.max() + 1e-6)
        img_array *= 255.0
        image = Image.fromarray(img_array.astype(np.uint8)).convert("L")

        if self.transform:
            image = self.transform(image)
        return image, label

# === Model ===
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# === Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Load test set
df = pd.read_csv(CSV_PATH)
test_df = df[(df['label'] == 'cropped') & (df['image_path'].str.contains('Test'))].reset_index(drop=True)
test_dataset = CroppedMammogramDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))
print(f"✅ Test Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")