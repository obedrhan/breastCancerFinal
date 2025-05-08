import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "cropped_machine_learning/data/hog_features_cropped_test.csv")

# === Load metadata ===
df = pd.read_csv(CSV_PATH)

# Normalize pathology to numeric labels
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['pathology'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

# === Prepare feature storage ===
features = []
image_names = []
labels = []

# === Process each image (only test + cropped) ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["image_path"]
    image_type = str(row.get("label", "")).strip().lower()  # 'roi' or 'cropped'

    # Only process TEST and CROPPED images
    if "test" not in image_path.lower() or image_type != "cropped":
        continue

    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        continue

    try:
        image = imread(image_path, as_gray=True)
        image = cv2.resize(image, (128, 128))

        hog_features = hog(image,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys',
                           visualize=False,
                           feature_vector=True)

        features.append(hog_features)
        image_names.append(os.path.basename(image_path))
        labels.append(row["pathology"])  # Use 0/1 from pathology

        print(f"✅ Processed {idx + 1}: {image_names[-1]}")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        continue

# === Save to CSV ===
if features:
    df_features = pd.DataFrame(features)
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 HOG features saved to: {OUTPUT_CSV}")
else:
    print("⚠️ No HOG features extracted.")