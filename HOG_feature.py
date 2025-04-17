import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
SEGMENTED_DIR = os.path.join(BASE_DIR, "segmented_output")
CSV_PATH = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data/hog_features_segmented.csv")

# === Load metadata (for labels) ===
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['label'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

# === Prepare list of HOG features ===
features = []
image_names = []
labels = []

# === Process only Training segmented images ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if 'Training' not in row['full_path']:
        continue

    flat_name = row['full_path'].replace("/", "_") + "_segmented.png"
    image_path = os.path.join(SEGMENTED_DIR, flat_name)

    if not os.path.exists(image_path):
        continue

    try:
        image = imread(image_path, as_gray=True)
        image = cv2.resize(image, (128, 128))  # Resize for uniform HOG features

        # --- HOG Extraction ---
        hog_features = hog(image,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys',
                           visualize=False,
                           feature_vector=True)

        features.append(hog_features)
        image_names.append(flat_name)
        labels.append(row['label'])

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        continue

# === Save to CSV ===
if features:
    df_features = pd.DataFrame(features)
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ HOG features saved to {OUTPUT_CSV}")
else:
    print("⚠️ No HOG features extracted.")