import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import mahotas
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "cropped_machine_learning/data/glcm_features_mahotas_test.csv")

# === Load metadata ===
df = pd.read_csv(CSV_PATH)
df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
df['pathology'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

print(f"📄 Loaded metadata with {len(df)} entries.")

# === Prepare feature storage ===
features = []
image_names = []
labels = []

# === Process each image ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["image_path"]
    image_type = str(row.get("label", "")).strip().lower()

    print(f"\n🔍 [{idx}] Checking image: {image_path}")
    print(f"    ⮞ Label: {image_type}")

    if "evaluation_test" not in image_path.lower() or image_type != "cropped":
        print("    ⏭️ Skipped (not 'training' or not 'cropped')")
        continue

    if not os.path.exists(image_path):
        print(f"    ❌ File not found: {image_path}")
        continue

    try:
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)
        print(f"    ✅ DICOM loaded. Shape: {image.shape}")

        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        haralick = mahotas.features.haralick(image).mean(axis=0)

        contrast = haralick[1]
        correlation = haralick[2]
        asm = haralick[0]
        homogeneity = haralick[4]

        print(f"    📊 Features: ASM={asm:.4f}, Contrast={contrast:.4f}, Correlation={correlation:.4f}, Homogeneity={homogeneity:.4f}")

        features.append([asm, contrast, correlation, homogeneity])
        image_names.append(os.path.basename(image_path))
        labels.append(row["pathology"])

    except Exception as e:
        print(f"    ❌ Error processing {image_path}: {e}")
        continue

# === Save features to CSV ===
print(f"\n📦 Total extracted features: {len(features)}")

if features:
    df_features = pd.DataFrame(features, columns=["ASM", "Contrast", "Correlation", "Homogeneity"])
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 GLCM features saved to: {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted.")