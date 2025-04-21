import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
SEGMENTED_DIR = os.path.join(BASE_DIR, 'segmented_output')

class MammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.dataframe['pathology'] = self.dataframe['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        self.dataframe['label'] = self.dataframe['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        relative_path = self.dataframe.iloc[idx]['full_path']
        flat_name = relative_path.replace("/", "_") + "_segmented.png"
        segmented_path = os.path.join(SEGMENTED_DIR, flat_name)
        label = self.dataframe.iloc[idx]['label']

        try:
            image = Image.open(segmented_path).convert("L")
        except Exception as e:
            print(f"⚠️ Failed to load image at {segmented_path}: {e}")
            new_idx = np.random.randint(0, len(self.dataframe))
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)

        return image, label

# Load only Training data
csv_path = os.path.join(BASE_DIR, 'data/full_mammogram_paths.csv')
df = pd.read_csv(csv_path)
train_df = df[df['full_path'].str.contains('Training')].reset_index(drop=True)

# Transformations for DenseNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# DataLoader for Training Only
train_dataset = MammogramDataset(train_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# DenseNet-121 Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
densenet = densenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(), lr=0.0005)

# Training Loop (No validation)
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch [{epoch+1}/{EPOCHS}]")
    densenet.train()
    total_train, correct_train = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = densenet(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(densenet.state_dict(), os.path.join(BASE_DIR, 'Segmented_deep_learning/Segmented_deep_learning/densenet121_segmented.pth'))
print("✅ DenseNet-121 model saved!")