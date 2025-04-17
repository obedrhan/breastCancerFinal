import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# === Paths ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, 'data/full_mammogram_paths.csv')
SEGMENTED_DIR = os.path.join(BASE_DIR, 'segmented_output')

# === Dataset ===
class MammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None, only_training=True, max_retries=5):
        self.transform = transform
        self.max_retries = max_retries
        self.valid_data = []

        dataframe['pathology'] = dataframe['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        dataframe['label'] = dataframe['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

        # Filter only training data
        if only_training:
            dataframe = dataframe[dataframe['full_path'].str.contains('Training')]

        for _, row in dataframe.iterrows():
            relative_path = row['full_path']
            flat_name = relative_path.replace("/", "_") + "_segmented.png"
            segmented_path = os.path.join(SEGMENTED_DIR, flat_name)

            if os.path.exists(segmented_path):
                self.valid_data.append({
                    "path": segmented_path,
                    "label": row['label']
                })
            else:
                print(f"⚠️ Skipping missing file: {segmented_path}")

        print(f"✅ Loaded {len(self.valid_data)} valid segmented training images")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        entry = self.valid_data[idx]
        retries = 0

        while retries < self.max_retries:
            try:
                image = Image.open(entry['path']).convert("L")
                if self.transform:
                    image = self.transform(image)
                return image, entry['label']
            except Exception as e:
                print(f"⚠️ Failed to load {entry['path']}: {e}")
                idx = np.random.randint(0, len(self.valid_data))
                entry = self.valid_data[idx]
                retries += 1

        raise RuntimeError("Too many failed attempts to load image")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load CSV & Dataset ===
df = pd.read_csv(CSV_PATH)
train_dataset = MammogramDataset(df, transform=transform, only_training=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# === Load ResNet18 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0005)

# === Training ===
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch [{epoch + 1}/{EPOCHS}]")
    resnet.train()
    correct_train = 0
    total_train = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# === Save Trained Model ===
save_path = os.path.join(BASE_DIR, 'segmentation/models/resnet18_segmented.pth')
torch.save(resnet.state_dict(), save_path)
print(f"✅ ResNet model saved to: {save_path}")