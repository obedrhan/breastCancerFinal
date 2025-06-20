import os
import glob
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern

# ───────── CONFIG ─────────────────────────────────────────────────
CROPS_ROOT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/eval_crops"
FULL_CSV     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv"
OUT_CSV      = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/lbp_features_with_pathology_unet.csv"

# LBP parameters
P        = 8        # number of neighbor points
R        = 1        # radius
METHOD   = "uniform"
N_BINS   = P + 2    # uniform patterns

# ───────── LOAD FULL CSV & BUILD LABEL MAP ─────────────────────────
full_df = pd.read_csv(FULL_CSV)
# detect full-image column and pathology column
full_col = next(c for c in full_df.columns if 'full' in c.lower())
path_col = 'pathology'
if path_col not in full_df.columns:
    raise RuntimeError(f"No '{path_col}' column in {FULL_CSV}")
# map from relative full path to pathology
label_map = dict(zip(full_df[full_col], full_df[path_col]))

# ───────── FEATURE EXTRACTION ──────────────────────────────────────
rows = []
# use recursive glob to find all segmented_image.png under nested folders
pattern = os.path.join(CROPS_ROOT, "**", "segmented_image.png")
for seg_path in glob.glob(pattern, recursive=True):
    # derive relative full-image path key by reversing segmentation dir structure
    # seg_path: .../eval_crops/<rel_path_without_ext>/segmented_image.png
    rel_dir = os.path.relpath(os.path.dirname(seg_path), CROPS_ROOT)
    # reconstruct full DICOM path
    rel_full = rel_dir + ".dcm"
    pathology = label_map.get(rel_full)
    if pathology is None:
        print(f"⚠️  No label for {rel_full}, skipping")
        continue

    # load lesion-only patch
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        print(f"⚠️  Could not read {seg_path}, skipping")
        continue

    mask = seg > 0
    if not mask.any():
        print(f"⚠️  No lesion pixels in {seg_path}, skipping")
        continue

    # compute LBP map & histogram on lesion pixels
    lbp      = local_binary_pattern(seg, P, R, METHOD).astype(int)
    lbp_vals = lbp[mask]
    hist, _ = np.histogram(lbp_vals, bins=np.arange(0, N_BINS+1), density=True)

    # assemble row with relative path and label
    feat = {
        "full_path": rel_full,
        "seg_path": seg_path,
        "pathology": pathology
    }
    for i, v in enumerate(hist):
        feat[f"lbp_{i}"] = v
    rows.append(feat)

# ───────── SAVE TO CSV ─────────────────────────────────────────────
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"✔️ Extracted features for {len(out_df)} images → {OUT_CSV}")