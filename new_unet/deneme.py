#!/usr/bin/env python3
# inspect_merged_csv_save.py

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pydicom

# ───────── CONFIG ────────────────────────────────────────
DDSM_ROOT = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH  = os.path.join(DDSM_ROOT, "data/final_full_roi.csv")
N_SAMPLES = 4
OUT_FILE  = os.path.join(DDSM_ROOT, "data/sample_full_roi.png")

# ───────── HELPERS ───────────────────────────────────────
def resolve_path(root, p):
    s = str(p).strip()
    return s if os.path.isabs(s) else os.path.normpath(os.path.join(root, s))

def load_mammo(path):
    if path.lower().endswith(".dcm"):
        d = pydicom.dcmread(path, force=True)
        arr = d.pixel_array.astype(np.float32)
        return (arr - arr.min())/(arr.max()-arr.min()+1e-8)
    else:
        im = Image.open(path).convert("L")
        return np.array(im, dtype=np.float32)/255.0

def load_roi(path):
    if path.lower().endswith(".dcm"):
        d = pydicom.dcmread(path, force=True)
        arr = d.pixel_array.astype(np.float32)
        return (arr > 0).astype(np.float32)
    else:
        im = Image.open(path).convert("L")
        arr = np.array(im, dtype=np.float32)/255.0
        return (arr > 0.5).astype(np.float32)

# ───────── MAIN ───────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    if total == 0:
        print("❌ final_full_roi.csv is empty!")
        return

    # pick up to N_SAMPLES valid rows
    indices = list(range(total))
    random.shuffle(indices)
    valid = []
    for i in indices:
        row = df.iloc[i]
        fp = resolve_path(DDSM_ROOT, row["full_path"])
        rp = resolve_path(DDSM_ROOT, row["roi_path"])
        if os.path.exists(fp) and os.path.exists(rp):
            valid.append((fp, rp))
        if len(valid) >= N_SAMPLES:
            break

    if not valid:
        print("❌ No valid full/ROI pairs found on disk.")
        return

    # make the plot
    fig, axes = plt.subplots(2, len(valid), figsize=(len(valid)*4, 8))
    for col, (fp, rp) in enumerate(valid):
        mam = load_mammo(fp)
        mask = load_roi(rp)

        ax1 = axes[0, col]
        ax1.imshow(mam, cmap="gray")
        ax1.set_title(f"IMG {os.path.basename(fp)}", fontsize=8)
        ax1.axis("off")

        ax2 = axes[1, col]
        ax2.imshow(mask, cmap="gray")
        ax2.set_title(f"ROI {os.path.basename(rp)}", fontsize=8)
        ax2.axis("off")

    plt.tight_layout()
    # **Save** to a PNG you can open manually
    plt.savefig(OUT_FILE, dpi=150)
    print(f"✅ Saved sample mosaic to {OUT_FILE}")
    # Then also try to show
    plt.show()

if __name__ == "__main__":
    main()