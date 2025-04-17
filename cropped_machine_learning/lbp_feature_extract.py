import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.io import imread

# Define LBP extraction function
def compute_lbp(image, radius=1, n_points=8):
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# Paths
base_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
csv_path = os.path.join(base_dir, "data/roi_mask_paths.csv")
df = pd.read_csv(csv_path)

# Output data
feature_matrix = []
image_names = []
labels = []

# Process only training data
for idx, row in df.iterrows():
    if "Training" not in row['full_path']:
        continue

    image_path = os.path.join(base_dir, "DDSM_IMAGES/CBIS-DDSM", row['ROI mask file path'])
    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        continue

    try:
        image = imread(image_path, as_gray=True)
        lbp_hist = compute_lbp(image)

        label = row['pathology'].strip().upper()
        if label == "BENIGN_WITHOUT_CALLBACK":
            label = "BENIGN"

        feature_matrix.append(lbp_hist)
        image_names.append(os.path.basename(image_path))
        labels.append(label)

        print(f"✅ Processed {idx + 1}: {image_names[-1]}")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

# Save to CSV
if feature_matrix:
    columns = [f"Feature_{i+1}" for i in range(len(feature_matrix[0]))]
    df_features = pd.DataFrame(feature_matrix, columns=columns)
    df_features.insert(0, "Image Name", image_names)
    df_features["Label"] = labels

    output_csv_path = os.path.join(base_dir, "cropped_machine_learning/data/lbp_features_cropped.csv")
    df_features.to_csv(output_csv_path, index=False)
    print(f"🎉 Feature CSV saved to: {output_csv_path}")
else:
    print("⚠️ No features were extracted.")