import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.filters import gabor
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_cropped_training.csv")

# === Load metadata (for labels) ===
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['pathology'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

# === GLCM-style custom metrics ===
def compute_glcm_like_features(image):
    black_ratio = np.sum(image == 0) / image.size
    white_ratio = np.sum(image == 255) / image.size
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    return [black_ratio, white_ratio, mean_intensity, std_intensity]

# === Gabor Features ===
def compute_gabor_features(image):
    gabor_feats = []
    frequencies = [0.1, 0.2, 0.3]
    thetas = [0, np.pi/4, np.pi/2]
    for theta in thetas:
        for freq in frequencies:
            filt_real, filt_imag = gabor(image, frequency=freq, theta=theta)
            gabor_feats.append(np.mean(filt_real))
            gabor_feats.append(np.std(filt_real))
    return gabor_feats

# === Feature storage ===
features = []
image_names = []
labels = []

# === Process cropped test images ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row.get("image_path", "")
    label_type = str(row.get("label", "")).strip().lower()

    if "training" not in image_path.lower() or label_type != "cropped":
        continue

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        continue

    try:
        image = imread(image_path, as_gray=True)
        image = cv2.resize(image, (128, 128))

        glcm_feat = compute_glcm_like_features(image)
        gabor_feat = compute_gabor_features(image)
        all_feats = glcm_feat + gabor_feat

        features.append(all_feats)
        image_names.append(os.path.basename(image_path))
        labels.append(row["pathology"])

    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        continue

# === Save to CSV ===
if features:
    columns = [f"Feature_{i+1}" for i in range(len(features[0]))]
    df_features = pd.DataFrame(features, columns=columns)
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ GLCM + Gabor features saved to: {OUTPUT_CSV}")
else:
    print("⚠️ No features were extracted.")