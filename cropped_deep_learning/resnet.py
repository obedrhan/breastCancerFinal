import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from cnn_training import CroppedMammogramDataset

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"


def main():
    # Load CSV
    csv_path = os.path.join(BASE_DIR, 'data/roi_cropped_with_pathology.csv')
    df = pd.read_csv(csv_path)

    # ✅ Keep only cropped images
    df = df[df['label'].astype(str).str.lower() == 'cropped']

    # ✅ Map pathology to binary label
    df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    df['class'] = df['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

    # ✅ Drop rows with unknown/missing pathology
    df = df[df['class'].notna()].reset_index(drop=True)

    train_df = df[df['full_path'].str.contains('Training')].reset_index(drop=True)
    test_df = df[df['full_path'].str.contains('Test')].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dataset = CroppedMammogramDataset(train_df, transform=transform)
    test_dataset = CroppedMammogramDataset(test_df, transform=transform)

    # Use num_workers=0 for safety on macOS
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.0005)

    # Training
    for epoch in range(10):
        print(f"\n🚀 Epoch [{epoch + 1}/10]")
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

        acc = correct_train / total_train * 100
        print(f"✅ Train Accuracy: {acc:.2f}%")

    # Save model
    save_path = os.path.join(BASE_DIR, "cropped_deep_learning/models/resnet18_cropped.pth")
    torch.save(resnet.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")


if __name__ == "__main__":
    main()