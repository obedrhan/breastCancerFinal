# evaluate_resnet18.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common_dataset import CroppedMammoDataset, BASE_DIR, CSV_PATH
import pandas as pd
import os

model_path = os.path.join(BASE_DIR, "cropped_deep_learning/models/resnet18_cropped.pth")

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

df = pd.read_csv(CSV_PATH)
df = df[df["label"].str.lower() == "cropped"]
test_df = df[df["image_path"].str.contains("Test")].reset_index(drop=True)

test_dataset = CroppedMammoDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Evaluation
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 ResNet18 Cropped Test Accuracy: {correct / total * 100:.2f}%")