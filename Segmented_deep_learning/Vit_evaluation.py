import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === Configuration ===
print("🔧 Configuring paths and parameters...")
IMG_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/segmented_Test_output"
CSV_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv"
MODEL_PATH = "vit_segmented_model.pth"

# === Dataset ===
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
    transforms.Resize((224, 224)),  # ViT default input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Load DataFrame and Split ===
print("📄 Loading and processing CSV file...")
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['label'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})
df = df[df['full_path'].str.contains("Test")]

print("📊 Splitting dataset...")
_, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

val_dataset = SegmentedDataset(val_df, IMG_DIR, transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Load Vision Transformer Model ===
print("📥 Loading ViT-B_16 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vit_b_16(weights=None)  # Set weights=None when loading saved model
model.heads.head = nn.Linear(768, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Evaluate ===
print("🔍 Evaluating model...")
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

# === Results ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"]))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["BENIGN", "MALIGNANT"], yticklabels=["BENIGN", "MALIGNANT"])
plt.title("Confusion Matrix - ViT B-16")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_vit_b16.png")
plt.show()