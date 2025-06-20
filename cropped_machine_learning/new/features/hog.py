import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

# === CONFIG ===
TARGET_SIZE = (256, 256)   # resize all images to this shape
CSV_INPUT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/roi_cropped_with_pathology.csv"
CSV_OUTPUT  = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/hog_features_mass_test_cropped.csv"

# === HOG Extraction Function ===
def compute_hog(image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'):
    """
    Compute a fixed-length HOG descriptor for a resized grayscale image.
    """
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True
    )

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    records = []

    for idx, row in df.iterrows():
        img_path  = row["image_path"]
        path_low  = img_path.lower()
        lbl_col   = str(row.get("label", "")).strip().lower()
        pathology = str(row.get("pathology", "")).strip().upper()

        # normalize pathology
        if pathology == "BENIGN_WITHOUT_CALLBACK":
            pathology = "BENIGN"

        # filters: must contain "mass" & "training" in path, and label == "cropped"
        if not ("mass" in path_low and "test" in path_low and lbl_col == "cropped"):
            continue

        if not os.path.exists(img_path):
            print(f"⚠️ File not found: {img_path}")
            continue

        try:
            # Read and resize
            img = imread(img_path, as_gray=True)
            img_resized = resize(img, TARGET_SIZE, anti_aliasing=True)

            # Compute HOG
            hog_vec = compute_hog(img_resized)

            rec = {
                "Image Name": os.path.basename(img_path),
                "Pathology":  pathology,
                "Label":      "CROPPED"
            }
            for i, val in enumerate(hog_vec, start=1):
                rec[f"HOG_{i}"] = val

            records.append(rec)
            print(f"✅ Processed {idx+1}: {rec['Image Name']} ({pathology})")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    # Save to CSV
    if records:
        out_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
        out_df.to_csv(CSV_OUTPUT, index=False)
        print(f"\n🎉 HOG features saved to:\n{CSV_OUTPUT}")
    else:
        print("\n⚠️ No features extracted.")