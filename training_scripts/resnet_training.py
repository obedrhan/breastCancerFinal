import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pydicom
import numpy as np

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

class MammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        # Normalize labels
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
            img_array = dicom.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"❌ Skipping corrupted DICOM file: {img_path} | Error: {e}")
            return self.__getitem__(np.random.randint(0, len(self.dataframe)))

        # Normalize image
        img_array -= img_array.min()
        img_array /= (img_array.max() + 1e-6)
        img_array *= 255.0
        img_array = img_array.astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(img_array)

        if self.transform:
            image = self.transform(image)
        return image, label

# ✅ Load CSV
csv_path = os.path.join(BASE_DIR, 'full_mammogram_paths.csv')
df = pd.read_csv(csv_path)
train_df = df[df['full_path'].str.contains('Training')].reset_index(drop=True)
test_df = df[df['full_path'].str.contains('Test')].reset_index(drop=True)

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Dataloaders
train_dataset = MammogramDataset(train_df, transform=transform)
test_dataset = MammogramDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Load ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # 2 classes: benign & malignant
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0005)

# ✅ Train Loop
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch [{epoch+1}/{EPOCHS}]")
    resnet.train()
    total_train, correct_train = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train * 100
    print(f"✅ Train Accuracy: {train_acc:.2f}%")

# ✅ Evaluate on Test Set
print("\n🧪 Evaluating on test set...")
resnet.eval()
total_test, correct_test = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = resnet(imgs)
        _, preds = torch.max(outputs, 1)
        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)

print(f"🎯 ✅ Final Test Accuracy: {correct_test / total_test * 100:.2f}%")

# ✅ Save ResNet model
torch.save(resnet.state_dict(), '/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/resnet18_full_mammo.pth')
print("💾 ResNet model saved!")