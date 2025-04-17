import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from cnn_training import MammogramDataset
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

# CSV Split
csv_path = os.path.join(BASE_DIR, 'full_mammogram_paths.csv')
df = pd.read_csv(csv_path)
train_df = df[df['full_path'].str.contains('Training')].reset_index(drop=True)
test_df = df[df['full_path'].str.contains('Test')].reset_index(drop=True)

# Transforms for DenseNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Loaders
train_dataset = MammogramDataset(train_df, transform=transform)
test_dataset = MammogramDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# DenseNet-121 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
densenet = densenet.to(device)

# Train Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(), lr=0.0005)

# Training Loop
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

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train * 100
    print(f" Train Accuracy: {train_acc:.2f}%")

# Evaluation
print("\n🧪 Evaluating on test set...")
densenet.eval()
total_test, correct_test = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = densenet(imgs)
        _, preds = torch.max(outputs, 1)
        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)

print(f" Final Test Accuracy: {correct_test / total_test * 100:.2f}%")

torch.save(densenet.state_dict(), '/models/deep_learning/densenet121_full_mammo.pth')
print(" DenseNet-121 model saved!")