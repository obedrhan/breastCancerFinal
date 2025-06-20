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
image_names    = []
type_labels    = []  # always "CROPPED"
pathology_labels = []

# === Feature Extraction on mass + training (path) and cropped (label) ===
for idx, row in df.iterrows():
    image_path = row["image_path"]
    path_lower = image_path.lower()
    label_col  = str(row.get("label", "")).strip().lower()
    pathology  = str(row.get("pathology", "")).strip().upper()

    # path must contain mass & training, and label column must be "cropped"
    if not ("mass" in path_lower and "test" in path_lower and label_col == "cropped"):
        continue

    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        continue

    # normalize pathology field
    if pathology == "BENIGN_WITHOUT_CALLBACK":
        pathology = "BENIGN"

    try:
        image    = imread(image_path, as_gray=True)
        lbp_hist = compute_lbp(image)

        feature_matrix.append(lbp_hist)
        image_names.append(os.path.basename(image_path))
        type_labels.append("CROPPED")
        pathology_labels.append(pathology)

        print(f"✅ Processed {idx + 1}: {image_names[-1]} ({pathology})")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

# === Save Feature Matrix to CSV ===
if feature_matrix:
    # build feature DataFrame
    feature_cols = [f"Feature_{i+1}" for i in range(len(feature_matrix[0]))]
    df_features = pd.DataFrame(feature_matrix, columns=feature_cols)
    # prepend image name
    df_features.insert(0, "Image Name", image_names)
    # insert pathology and type label
    df_features.insert(1, "Pathology", pathology_labels)
    df_features["Label"] = type_labels

    output_path = "/cropped_machine_learning/new/new_data/lbp_features_mass_cropped_test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"\n🎉 LBP feature CSV saved to:\n{output_path}")
else:
    print("\n⚠️ No features were extracted.")