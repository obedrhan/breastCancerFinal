import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import pydicom
import torchvision.transforms as T
from torchvision.models import densenet121, efficientnet_b0
from skimage.feature import hog, local_binary_pattern

# === Model Paths ===
BASE_DIR = "/Users/Bedirhan/Desktop/BreastCancer"
DN121_PATH = os.path.join(BASE_DIR, "Models/densenet_full_mammo.pth")
EFF_PATH = os.path.join(BASE_DIR, "Models/efficientnet_full_mammo.pth")
RF_PATH = os.path.join(BASE_DIR, "Models/random_forest_model_hog_lbp.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Models/scaler_rf_hog_lbp.pkl")

# === Load Random Forest & Scaler ===
rf = joblib.load(RF_PATH)
scaler = joblib.load(SCALER_PATH)

# === Load DenseNet121 ===
dn = densenet121(weights=None)
dn.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
dn.classifier = nn.Linear(dn.classifier.in_features, 2)
dn.load_state_dict(torch.load(DN121_PATH, map_location="cpu"))
dn.eval()

# === Load EfficientNet-B0 ===
ef = efficientnet_b0(weights=None)
ef.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
ef.classifier[1] = nn.Linear(ef.classifier[1].in_features, 2)
ef.load_state_dict(torch.load(EFF_PATH, map_location="cpu"))
ef.eval()

# === Ensemble Weights ===
WEIGHTS = {
    "random_forest": 0.6376,
    "densenet": 0.6140,
    "efficientnet": 0.6915
}


def load_full_image(path, model_type):
    """Load and preprocess a DICOM image for DL models."""
    dcm = pydicom.dcmread(path, force=True)
    arr = dcm.pixel_array.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + 1e-6)
    arr *= 255
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")

    size = (512, 512) if model_type == "densenet" else (224, 224)
    tf = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    return tf(img).unsqueeze(0)  # shape: (1, 1, H, W)


def load_cropped_image(arr):
    """Extract HOG + LBP features from a cropped region."""
    arr = cv2.resize(arr, (128, 128))

    # HOG
    hog_feats = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # LBP (fixed 59 bins for uniform P=8)
    lbp = local_binary_pattern(arr, P=8, R=1, method="uniform")
    n_bins = 59
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    feats = np.hstack([hog_feats, hist])

    # === Fix for StandardScaler shape mismatch ===
    expected = scaler.mean_.shape[0]
    if feats.shape[0] < expected:
        pad = np.zeros(expected - feats.shape[0])
        feats = np.concatenate([feats, pad])
    elif feats.shape[0] > expected:
        feats = feats[:expected]

    return feats


def predict_mammogram(full_path, cropped_arr):
    """Run all 3 models and return ensemble result."""
    # --- Random Forest on ROI features ---
    feats = load_cropped_image(cropped_arr).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    p_rf = rf.predict_proba(feats_scaled)[0, 1]

    # --- DenseNet prediction ---
    inp_dn = load_full_image(full_path, "densenet")
    with torch.no_grad():
        out_dn = dn(inp_dn)
        p_dn = torch.softmax(out_dn, dim=1)[0, 1].item()

    # --- EfficientNet prediction ---
    inp_ef = load_full_image(full_path, "efficientnet")
    with torch.no_grad():
        out_ef = ef(inp_ef)
        p_ef = torch.softmax(out_ef, dim=1)[0, 1].item()

    # --- Weighted average ensemble ---
    total_w = sum(WEIGHTS.values())
    score = (
        p_rf * WEIGHTS["random_forest"] +
        p_dn * WEIGHTS["densenet"] +
        p_ef * WEIGHTS["efficientnet"]
    ) / total_w

    pred = "MALIGNANT" if score >= 0.5 else "BENIGN"
    return {
        "prediction": pred,
        "confidence": float(score),
        "random_forest": float(p_rf),
        "densenet": float(p_dn),
        "efficientnet": float(p_ef)
    }