import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import densenet121
from torch.utils.data import DataLoader
from cnn_training import CroppedMammogramDataset  # your custom dataset

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

def main():
    csv_path = os.path.join(BASE_DIR, 'data/roi_cropped_with_pathology.csv')
    # Load CSV
    df = pd.read_csv(csv_path)

    # ✅ Keep only cropped images
    df = df[df['label'].astype(str).str.lower() == 'cropped']

    # ✅ Map pathology to binary label
    df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    df['class'] = df['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

    # ✅ Drop rows with unknown/missing pathology
    df = df[df['class'].notna()].reset_index(drop=True)

    train_df = df[df["image_path"].str.contains("Training")]
    test_df = df[df["image_path"].str.contains("Test")]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = CroppedMammogramDataset(train_df, transform=transform)
    test_dataset = CroppedMammogramDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # ← safest on macOS
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(weights="IMAGENET1K_V1")
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch [{epoch+1}/{EPOCHS}] -------------------------")
        model.train()
        total_train, correct_train = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train * 100
        print(f"✅ Epoch {epoch+1} Accuracy: {train_acc:.2f}%")

    # Save model
    save_path = os.path.join(BASE_DIR, "cropped_deep_learning/models/densenet_cropped_mammo.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    main()