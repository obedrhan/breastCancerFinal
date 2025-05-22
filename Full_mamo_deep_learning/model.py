import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pydicom
import torchvision.transforms as T
from torchvision.models import densenet121, efficientnet_b0
from skimage.feature import hog, local_binary_pattern

# — Configure your saved model paths —
BASE_DIR    = "/Users/ecekocabay/Desktop/2025SPRING/CNG492/DDSM"
DN121_PATH  = os.path.join(BASE_DIR, "models/full_mamo_deep_learning/densenet_full_mammo.pth")
EFF_PATH    = os.path.join(BASE_DIR, "models/full_mamo_deep_learning/efficientnet_full_mammo.pth")
RF_PATH     = os.path.join(BASE_DIR, "models/HOG_LBP/random_forest_model_hog_lbp.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/HOG_LBP/scaler_rf_hog_lbp.pkl")

# — Load ML models —
rf     = joblib.load(RF_PATH)
scaler = joblib.load(SCALER_PATH)

# — Load DenseNet121 (1-channel input) —
dn = densenet121(weights=None)
dn.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
dn.classifier = nn.Linear(dn.classifier.in_features, 2)
dn.load_state_dict(torch.load(DN121_PATH, map_location="cpu"))
dn.eval()

# — Load EfficientNet-B0 (1-channel input) —
ef = efficientnet_b0(weights=None)
ef.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
ef.classifier[1] = torch.nn.Linear(ef.classifier[1].in_features, 2)
ef.load_state_dict(torch.load(EFF_PATH, map_location="cpu"))
ef.eval()

# — Ensemble weights —
WEIGHTS = {
    "random_forest": 0.6376,
    "densenet":      0.6140,
    "efficientnet":  0.6915
}

# — Preprocessing Functions —
def load_full_image(path, model_type):
    """
    Load a DICOM at `path`, normalize to [0,255], convert to grayscale tensor
    of shape (1,1,H,W) for DenseNet or EfficientNet.
    """
    dcm = pydicom.dcmread(path, force=True)
    arr = dcm.pixel_array.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + 1e-6)
    arr *= 255
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")  # grayscale

    if model_type == "densenet":
        tf = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # 1-channel
        ])
    else:  # efficientnet
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # 1-channel
        ])
    return tf(img).unsqueeze(0)  # add batch dimension

def load_cropped_image(arr):
    """
    Given a uint8 ROI array (0–255), compute HOG + LBP histogram
    and return a 1D numpy feature vector.
    """
    hog_feats = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        feature_vector=True
    )
    lbp      = local_binary_pattern(arr, P=8, R=1, method="uniform")
    n_bins   = int(lbp.max() + 1)
    hist, _  = np.histogram(
        lbp,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    return np.hstack([hog_feats, hist])

# — Prediction Function —
def predict_mammogram(full_path, cropped_arr):
    """
    full_path: path to the original DICOM
    cropped_arr: numpy array of segmented ROI (uint8)
    """
    # — Random Forest on ROI —
    feats  = load_cropped_image(cropped_arr).reshape(1, -1)
    p_rf   = rf.predict_proba(scaler.transform(feats))[0, 1]

    # — DenseNet121 on full image —
    inp_dn = load_full_image(full_path, "densenet")
    with torch.no_grad():
        out_dn = dn(inp_dn)
        p_dn   = torch.softmax(out_dn, dim=1)[0, 1].item()

    # — EfficientNet-B0 on full image —
    inp_ef = load_full_image(full_path, "efficientnet")
    with torch.no_grad():
        out_ef = ef(inp_ef)
        p_ef   = torch.softmax(out_ef, dim=1)[0, 1].item()

    # — Weighted ensemble —
    total_w = sum(WEIGHTS.values())
    score   = (
        p_rf * WEIGHTS["random_forest"] +
        p_dn * WEIGHTS["densenet"] +
        p_ef * WEIGHTS["efficientnet"]
    ) / total_w

    pred    = "malignant" if score >= 0.5 else "benign"
    return {"prediction": pred, "confidence": float(score)}