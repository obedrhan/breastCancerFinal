import os
import glob
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern, hog

# ───────── CONFIGURATION ───────────────────────────────────────────
CROPS_ROOT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/eval_crops"
FULL_CSV     = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv"
OUT_CSV      = ("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/unet/hog_lbp_features_with_pathology_unet.csv")
# LBP parameters
P          = 8        # number of neighbor points
R          = 1        # radius
LBP_METHOD = 'uniform'
N_BINS     = P + 2    # uniform patterns

# HOG parameters
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
HOG_BLOCK_NORM       = 'L2-Hys'
# fixed resize for HOG to reduce memory and dimension
HOG_RESIZE = (128, 128)
# compute HOG feature length on dummy resized patch
dummy = np.zeros(HOG_RESIZE, dtype=np.uint8)
_dummy_hog = hog(
    dummy,
    orientations=HOG_ORIENTATIONS,
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
    block_norm=HOG_BLOCK_NORM,
    feature_vector=True
)
HOG_LEN = _dummy_hog.size

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    full_col = next(c for c in df.columns if 'full' in c.lower())
    df['pathology'] = df['pathology'].str.lower().replace({
        'benign_without_callback': 'benign',
        'benign': 'benign',
        'malignant': 'malignant'
    })
    return dict(zip(df[full_col], df['pathology']))

# load pathology labels map
label_map = load_labels(FULL_CSV)

# prepare output CSV header
first_row = True
columns = ['full_path','seg_path','pathology'] + [f'lbp_{i}' for i in range(N_BINS)] + [f'hog_{i}' for i in range(HOG_LEN)]
with open(OUT_CSV, 'w') as f:
    f.write(','.join(columns) + '\n')

# process each segmented image
total = 0
pattern = os.path.join(CROPS_ROOT, '**', 'segmented_image.png')
for seg_path in glob.glob(pattern, recursive=True):
    rel_dir = os.path.relpath(os.path.dirname(seg_path), CROPS_ROOT)
    rel_full = rel_dir + '.dcm'
    pathology = label_map.get(rel_full)
    if pathology is None:
        continue
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        continue
    # tight lesion ROI
    mask = seg > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        continue
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    roi = seg[y0:y1+1, x0:x1+1]
    # LBP features
    lbp = local_binary_pattern(roi, P, R, LBP_METHOD).astype(int)
    lbp_vals = lbp[roi > 0]
    lbp_hist, _ = np.histogram(lbp_vals, bins=np.arange(0, N_BINS+1), density=True)
    # HOG features on resized ROI
    roi_resized = cv2.resize(roi, HOG_RESIZE, interpolation=cv2.INTER_LINEAR)
    hog_feat = hog(
        roi_resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True
    )
    # assemble and write row
    row = [rel_full, seg_path, pathology]
    row += lbp_hist.tolist()
    row += hog_feat.tolist()
    with open(OUT_CSV, 'a') as f:
        f.write(','.join(map(str, row)) + '\n')
    total += 1

print(f"✔️ Extracted HOG+LBP features for {total} images → {OUT_CSV}")
