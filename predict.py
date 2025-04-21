import os
import joblib
import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from segmentation_feature import (
    load_dicom_as_image,
    contrast_enhancement,
    region_growing,
    morphological_operations,
    contour_extraction,
    crop_image_with_contours,
    compute_lbp
)
import torchvision.models as models
import torch.nn as nn

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
SCALER_PATH = os.path.join(BASE_DIR, "Segmented_deep_learning", "scaler_rf.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "Segmented_deep_learning", "random_forest_model_ddsm.pkl")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "Segmented_deep_learning", "resnet18_full_mammo.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RF + scaler
scaler = joblib.load(SCALER_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
resnet = resnet.to(device)
resnet.eval()

def predict_mammogram(image_path):
    print(f"\n📥 Processing: {image_path}")

    # ---- Traditional ML Pipeline ----
    original_image = load_dicom_as_image(image_path)
    enhanced = contrast_enhancement(original_image)
    seed_point = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
    grown = region_growing(enhanced, seed_point, threshold=15)
    refined = morphological_operations(grown)
    _, contours = contour_extraction(refined)
    cropped_img = crop_image_with_contours(original_image, contours)

    lbp_hist = compute_lbp(cropped_img)
    lbp_scaled = scaler.transform([lbp_hist])
    ml_pred = rf_model.predict(lbp_scaled)[0]
    ml_conf = rf_model.predict_proba(lbp_scaled).max() * 100

    if ml_pred in [0, "0", "BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        ml_pred = "BENIGN"
    else:
        ml_pred = "MALIGNANT"

    # ---- Deep Learning Pipeline ----
    resized_img = resize(original_image, (224, 224), anti_aliasing=True)
    resized_img = (resized_img / 255.0 - 0.5) / 0.5
    tensor_img = torch.tensor(resized_img).unsqueeze(0).repeat(1, 3, 1, 1).float().to(device)

    with torch.no_grad():
        logits = resnet(tensor_img)
        probs = F.softmax(logits, dim=1)
        dl_conf, dl_pred_idx = torch.max(probs, dim=1)
        dl_pred = "BENIGN" if dl_pred_idx.item() == 0 else "MALIGNANT"

    # ---- Model Selection ----
    if dl_conf.item() * 100 > ml_conf:
        print("🔍 Selected Model: ResNet18")
        return {
            "prediction": dl_pred,
            "confidence": round(dl_conf.item() * 100, 2),
            "model": "ResNet18"
        }
    else:
        print("🔍 Selected Model: Random Forest")
        return {
            "prediction": ml_pred,
            "confidence": round(ml_conf, 2),
            "model": "Random Forest"
        }

if __name__ == "__main__":
    test_img = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/DDSM_IMAGES/CBIS-DDSM/Calc-Test_P_00077_RIGHT_CC/08-29-2017-DDSM-NA-38195/1.000000-full mammogram images-87486/1-1.dcm"
    result = predict_mammogram(test_img)
    print(result)