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

# Base directory
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODELS_DIR = os.path.join(BASE_DIR, "models")
LBP_CSV = os.path.join(BASE_DIR, "test/test_lbp_features.csv")
DL_CSV = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")

# ========================================
# ✅ 1️⃣ Traditional ML Models (LBP features)
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
    print(f"✅ {model_name} Accuracy: {acc * 100:.2f}%")

# ========================================
# ✅ 2️⃣ Deep Learning Models (Full Images)
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
    print(f"✅ {model_name} Accuracy: {correct / total * 100:.2f}%")

# Evaluate all DL models
evaluate_dl(cnn, "Custom CNN", test_loader_cnn)
evaluate_dl(resnet, "ResNet18", test_loader_resnet)
evaluate_dl(densenet, "DenseNet121", test_loader_resnet)