import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage.io import imread
from skimage.transform import resize

# === CONFIG ===
TARGET_SIZE = (256, 256)   # all images will be resized to this
CSV_INPUT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/roi_cropped_with_pathology.csv"
CSV_OUTPUT  = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/lbp_hog_mass_features_test.csv"

# === Feature functions ===
def compute_lbp(image, radius=1, n_points=8):
    """Uniform LBP → histogram of length n_points+2 (here 10)."""
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float32')
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_hog(image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'):
    """Fixed-length HOG on a TARGET_SIZE image."""
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True
    )

# === MAIN ===
if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    records = []

    for idx, row in df.iterrows():
        path = row["image_path"]
        p_low = path.lower()
        lbl   = str(row.get("label","")).strip().lower()
        pat   = str(row.get("pathology","")).strip().upper()

        if pat == "BENIGN_WITHOUT_CALLBACK":
            pat = "BENIGN"

        # filters: mass & training in path, label == cropped
        if not ("mass" in p_low and "test" in p_low and lbl == "cropped"):
            continue
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            continue

        try:
            img = imread(path, as_gray=True)
            # resize → values in [0,1]
            img_resized = resize(img, TARGET_SIZE, anti_aliasing=True)

            # scale to [0,255] uint8 for LBP
            img_uint8 = (img_resized * 255).astype(np.uint8)

            lbp_vec = compute_lbp(img_uint8)
            hog_vec = compute_hog(img_resized)

            rec = {
                "Image Name": os.path.basename(path),
                "Pathology": pat,
                "Label":     "CROPPED"
            }
            # add LBP features
            for i, v in enumerate(lbp_vec, start=1):
                rec[f"LBP_{i}"] = v
            # add HOG features
            for j, v in enumerate(hog_vec, start=1):
                rec[f"HOG_{j}"] = v

            records.append(rec)
            print(f"✅ {idx+1}: {rec['Image Name']} ({pat})")

        except Exception as e:
            print(f"❌ Error {path}: {e}")

    if records:
        out_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
        out_df.to_csv(CSV_OUTPUT, index=False)
        print(f"\n🎉 Saved fixed-length features to:\n{CSV_OUTPUT}")
    else:
        print("\n⚠️ No records to save.")