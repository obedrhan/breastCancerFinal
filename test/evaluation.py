import os
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

# 🔄 Reuse Dataset & CNN class from training script
from training_scripts.cnn_training import MammogramDataset, CustomCNN
'''
# Base directory
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODELS_DIR = os.path.join(BASE_DIR, "segmentation/models")
LBP_CSV = os.path.join(BASE_DIR, "test/test_lbp_features.csv")
DL_CSV = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")

# ========================================
#  Traditional ML Models (LBP features)
# ========================================
print("\n🔍 Evaluating Traditional ML Models on LBP features...")

lbp_df = pd.read_csv(LBP_CSV)
X_lbp = lbp_df.iloc[:, 1:-1].values
y_lbp = lbp_df.iloc[:, -1].map({'benign': 0, 'malignant': 1}).values

rf = joblib.load(os.path.join(MODELS_DIR, "random_forest_model_ddsm.pkl"))
knn = joblib.load(os.path.join(MODELS_DIR, "knn_model_ddsm.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm_model_ddsm.pkl"))

for model_name, model in zip(['Random Forest', 'KNN', 'SVM'], [rf, knn, svm]):
    preds = model.predict(X_lbp)
    preds = pd.Series(preds).map({'benign': 0, 'malignant': 1}).values
    acc = accuracy_score(y_lbp, preds)
    print(f" {model_name} Accuracy: {acc * 100:.2f}%")

# ========================================
#  2️⃣ Deep Learning Models (Full Images)
# ========================================
print("\n🔍 Evaluating Deep Learning Models on Full Mammograms...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Separate transforms for CNN (1 channel) and ResNet/DenseNet (3 channels)
transform_cnn = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_resnet = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset & Dataloaders
full_df = pd.read_csv(DL_CSV)
test_df = full_df[full_df['full_path'].str.contains('Test')].reset_index(drop=True)

# Loader for CNN (1 channel)
test_dataset_cnn = MammogramDataset(test_df, transform=transform_cnn)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=16, shuffle=False)

# Loader for ResNet/DenseNet (3 channels)
test_dataset_resnet = MammogramDataset(test_df, transform=transform_resnet)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=16, shuffle=False)

# Load Models
cnn = CustomCNN().to(device)
cnn.load_state_dict(torch.load(os.path.join(MODELS_DIR, "custom_cnn_full_mammo.pth")))
cnn.eval()

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet18_full_mammo.pth"), map_location=device))
resnet = resnet.to(device)
resnet.eval()

densenet = models.densenet121(weights=None)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
densenet.load_state_dict(torch.load(os.path.join(MODELS_DIR, "densenet121_full_mammo.pth"), map_location=device))
densenet = densenet.to(device)
densenet.eval()

# Evaluation Function
def evaluate_dl(model, model_name, loader):
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f" {model_name} Accuracy: {correct / total * 100:.2f}%")

# Evaluate all DL models
evaluate_dl(cnn, "Custom CNN", test_loader_cnn)
evaluate_dl(resnet, "ResNet18", test_loader_resnet)
evaluate_dl(densenet, "DenseNet121", test_loader_resnet)'''

import os
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# === Configuration ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")
SEGMENTED_TEST_DIR = os.path.join(BASE_DIR, "segmented_Test_output")
MODEL_DIR = os.path.join(BASE_DIR, "segmentation/models")

# === Custom CNN Model (matching training structure) ===
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

# === Dataset Loader for Segmented Test Images ===
class SegmentedTestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.samples = []
        self.transform = transform

        dataframe['pathology'] = dataframe['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        dataframe['label'] = dataframe['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

        for _, row in dataframe.iterrows():
            if 'Test' not in row['full_path']:
                continue
            flat_name = row['full_path'].replace("/", "_") + "_segmented.png"
            image_path = os.path.join(SEGMENTED_TEST_DIR, flat_name)
            if os.path.exists(image_path):
                self.samples.append((image_path, row['label']))
            else:
                print(f"⚠️ Missing file: {flat_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert("L")
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))  # skip to next
        if self.transform:
            image = self.transform(image)
        return image, label

# === Transforms ===
transform_cnn = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_resnet = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load Data ===
df = pd.read_csv(CSV_PATH)
test_dataset_cnn = SegmentedTestDataset(df, transform=transform_cnn)
test_dataset_resnet = SegmentedTestDataset(df, transform=transform_resnet)

test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=16, shuffle=False)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=16, shuffle=False)

# === Load Models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = CustomCNN().to(device)
cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "custom_cnn_segmented.pth"), map_location=device))
cnn.eval()

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet18_segmented.pth"), map_location=device))
resnet = resnet.to(device)
resnet.eval()

densenet = models.densenet121(weights=None)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
densenet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "densenet121_segmented.pth"), map_location=device))
densenet = densenet.to(device)
densenet.eval()

# === Evaluation Function ===
def evaluate_dl(model, model_name, loader):
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"✅ {model_name} Accuracy on segmented test images: {correct / total * 100:.2f}%")

# === Run Evaluation ===
print(f"\n📊 Evaluating on {len(test_dataset_cnn)} segmented test images...")
evaluate_dl(cnn, "Custom CNN", test_loader_cnn)
evaluate_dl(resnet, "ResNet18", test_loader_resnet)
evaluate_dl(densenet, "DenseNet121", test_loader_resnet)