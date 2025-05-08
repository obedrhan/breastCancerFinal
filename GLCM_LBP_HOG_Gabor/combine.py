import pandas as pd
import os

# === File paths ===
base_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
lbp_path = os.path.join(base_path, "test/test_lbp_features.csv")
hog_path = os.path.join(base_path, "test/hog_features_segmented.csv")
glcm_gabor_path = os.path.join(base_path, "GABOR_GLCM/data/glcm_gabor_features_segmented_test.csv")
output_path = os.path.join(base_path, "GLCM_LBP_HOG_Gabor/data/combined_features_test.csv")

# === Load CSVs ===
lbp_df = pd.read_csv(lbp_path)
hog_df = pd.read_csv(hog_path)
glcm_df = pd.read_csv(glcm_gabor_path)

# === Align by index and drop duplicates of label/image name ===
combined = pd.concat([
    lbp_df.drop(columns=["Label", "Image Name"], errors='ignore'),
    hog_df.drop(columns=["Label", "Image Name"], errors='ignore'),
    glcm_df.drop(columns=["Image Name"], errors='ignore')
], axis=1)

# === Use label from LBP (or any since they're aligned) ===
combined.insert(0, "Image Name", lbp_df["Image Name"])
combined["Label"] = lbp_df["Label"]

# === Save combined ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined.to_csv(output_path, index=False)
print(f"✅ Combined feature set saved to:\n{output_path}")