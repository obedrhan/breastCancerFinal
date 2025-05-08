import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.io import imread

# === Define LBP Extraction Function ===
def compute_lbp(image, radius=1, n_points=8):
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# === Load CSV with image paths and labels ===
csv_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/roi_cropped_with_pathology.csv"
df = pd.read_csv(csv_path)

# === Output Storage ===
feature_matrix = []
image_names = []
labels = []

# === Feature Extraction ===
for idx, row in df.iterrows():
    image_path = row["image_path"]
    pathology = row.get("pathology", "")
    label_type = row.get("label", "").strip().lower()

    # Only process "Training" cropped images with a valid pathology
    if not isinstance(pathology, str) or "test" not in image_path.lower() or label_type != "cropped":
        continue

    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        continue

    try:
        image = imread(image_path, as_gray=True)
        lbp_hist = compute_lbp(image)

        label = pathology.strip().upper()
        if label == "BENIGN_WITHOUT_CALLBACK":
            label = "BENIGN"

        feature_matrix.append(lbp_hist)
        image_names.append(os.path.basename(image_path))
        labels.append(label)

        print(f"✅ Processed {idx + 1}: {image_names[-1]}")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

# === Save Feature Matrix to CSV ===
if feature_matrix:
    columns = [f"Feature_{i+1}" for i in range(len(feature_matrix[0]))]
    df_features = pd.DataFrame(feature_matrix, columns=columns)
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels

    output_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/lbp_features_cropped_test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"\n🎉 LBP feature CSV saved to:\n{output_path}")
else:
    print("\n⚠️ No features were extracted.")