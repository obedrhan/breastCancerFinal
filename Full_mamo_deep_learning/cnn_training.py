import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pydicom
import numpy as np

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

# Dataset
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
        img_path = os.path.join(BASE_DIR, relative_path)
        label = self.dataframe.iloc[idx]['label']

        try:
            dicom = pydicom.dcmread(img_path, force=True)
            if 'PixelData' not in dicom:
                raise ValueError("Missing PixelData")
            img_array = dicom.pixel_array.astype(np.float32)
        except Exception as e:
            # fallback: pick a new random sample
            new_idx = np.random.randint(0, len(self.dataframe))
            return self.__getitem__(new_idx)

        # Normalize image
        img_array -= img_array.min()
        img_array /= (img_array.max() + 1e-6)
        img_array *= 255.0
        image = Image.fromarray(img_array.astype(np.uint8)).convert("L")

        if self.transform:
            image = self.transform(image)
        return image, label

# Model
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
            nn.Linear(128 * 64 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Only define dataset and model here, remove training logic
df = pd.read_csv(os.path.join(BASE_DIR, 'data/full_mammogram_paths.csv'))
train_df = df[df['full_path'].str.contains('Training')].reset_index(drop=True)
test_df = df[df['full_path'].str.contains('Test')].reset_index(drop=True)

# Optional: Keep dataloaders for reuse
train_dataset = MammogramDataset(train_df, transform=transform)
test_dataset = MammogramDataset(test_df, transform=transform)

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 20
    for epoch in range(EPOCHS):
        print(f"\n Starting Epoch [{epoch+1}/{EPOCHS}]")
        model.train()
        total_train, correct_train = 0, 0
        batch_counter = 0

        for imgs, labels in train_loader:
            batch_counter += 1
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if batch_counter % 10 == 0:
                print(f"   🔄 Batch {batch_counter} | Loss: {loss.item():.4f}")

        train_acc = correct_train / total_train * 100
        print(f" Epoch [{epoch+1}/{EPOCHS}] Completed | Train Accuracy: {train_acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'Segmented_deep_learning/custom_cnn_full_mammo.pth'))
    print(" Model saved to Segmented_deep_learning/custom_cnn_full_mammo.pth ")