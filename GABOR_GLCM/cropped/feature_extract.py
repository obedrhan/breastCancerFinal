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
OUTPUT_CSV = os.path.join(BASE_DIR, "Gabor_GLCM/data/glcm_gabor_features_cropped_training.csv")

# === Load metadata ===
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['pathology'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

# === Real GLCM (multi-orientation) ===
def compute_glcm_features(image, levels=8):
    image = cv2.normalize(image, None, 0, levels - 1, cv2.NORM_MINMAX).astype(int)
    directions = [(0, 1), (1, -1), (1, 0), (-1, -1)]  # 0°, 45°, 90°, 135°
    features = []

    for dx, dy in directions:
        glcm = np.zeros((levels, levels), dtype=np.float32)
        h, w = image.shape
        for i in range(h - abs(dy)):
            for j in range(w - abs(dx)):
                r = image[i, j]
                c = image[i + dy, j + dx]
                glcm[r, c] += 1
        glcm += glcm.T
        glcm /= (glcm.sum() + 1e-6)

        contrast = np.sum([(i - j) ** 2 * glcm[i, j] for i in range(levels) for j in range(levels)])
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum([glcm[i, j] / (1 + abs(i - j)) for i in range(levels) for j in range(levels)])
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
        features.append([contrast, energy, homogeneity, entropy])

    features = np.mean(features, axis=0)
    return features.tolist()

# === Gabor Features ===
def compute_gabor_features(image):
    gabor_feats = []
    frequencies = [0.1, 0.2, 0.3]
    thetas = [0, np.pi / 4, np.pi / 2]
    for theta in thetas:
        for freq in frequencies:
            filt_real, _ = gabor(image, frequency=freq, theta=theta)
            gabor_feats.append(np.mean(filt_real))
            gabor_feats.append(np.std(filt_real))
    return gabor_feats

# === Feature Extraction ===
features = []
image_names = []
labels = []

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
        image = (image / image.max()).astype(np.float32)

        glcm_feat = compute_glcm_features(image)
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
    df_out = pd.DataFrame(features, columns=columns)
    df_out.insert(0, "Image Name", image_names)
    df_out["Label"] = labels

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Features saved to: {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted.")