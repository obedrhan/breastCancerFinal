# evaluate_densenet121.py

import torch
import torch.nn as nn
from torchvision.models import densenet121
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common_dataset import CroppedMammoDataset, BASE_DIR, CSV_PATH
import pandas as pd
import os

model_path = os.path.join(BASE_DIR, "cropped_deep_learning/models/densenet_cropped_mammo.pth")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

df = pd.read_csv(CSV_PATH)
df = df[df["label"].str.lower() == "cropped"]
test_df = df[df["image_path"].str.contains("Test")].reset_index(drop=True)

test_dataset = CroppedMammoDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = densenet121()
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 DenseNet121 Cropped Test Accuracy: {correct / total * 100:.2f}%")