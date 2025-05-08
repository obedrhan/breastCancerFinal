import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# === Configuration ===
print("🔧 Configuring paths and parameters...")
IMG_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/segmented_output"
CSV_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv"

class SegmentedDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        while idx < len(self.dataframe):
            try:
                row = self.dataframe.iloc[idx]
                img_name = os.path.join(self.img_dir, row["full_path"].replace("/", "_") + "_segmented.png")
                if not os.path.exists(img_name):
                    print(f"❌ Skipping missing file: {img_name}")
                    idx += 1
                    continue

                image = Image.open(img_name).convert("RGB")
                label = int(row["label"])
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                print(f"⚠️ Error loading image at idx {idx}: {e}")
                idx += 1

        raise IndexError("All entries skipped or invalid.")

# === Transformations ===
print("🖼️ Setting up image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Load DataFrame and Prepare Dataset ===
print("📄 Loading and processing CSV file...")
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['label'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})
df = df[df['full_path'].str.contains("Training")]

print("📊 Splitting dataset into train and validation...")
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = SegmentedDataset(train_df, IMG_DIR, transform)
val_dataset = SegmentedDataset(val_df, IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Load DenseNet121 Model ===
print("📥 Loading DenseNet121 model...")
model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Loss & Optimizer ===
print("⚙️ Setting up loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
print("🚀 Starting training loop...")
EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f"  🔁 Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Complete - Average Loss: {running_loss/len(train_loader):.4f}")

# === Save Model ===
torch.save(model.state_dict(), "densenet121_segmented_model.pth")
print("\n💾 DenseNet121 model trained and saved to: densenet121_segmented_model.pth")