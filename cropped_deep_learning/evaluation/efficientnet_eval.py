# evaluate_efficientnet.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common_dataset import CroppedMammoDataset, BASE_DIR, CSV_PATH
import pandas as pd
import os

model_path = os.path.join(BASE_DIR, "cropped_deep_learning/models/efficientnet_cropped.pth")

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
model = efficientnet_b0()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
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

print(f"🎯 EfficientNet Cropped Test Accuracy: {correct / total * 100:.2f}%")
