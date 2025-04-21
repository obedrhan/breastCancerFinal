import os
import csv
import pydicom
import numpy as np

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/DDSM_IMAGES/CBIS-DDSM"
OUTPUT_CSV = "data/roi_cropped_labels.csv"

data = []

for root, dirs, files in os.walk(BASE_DIR):
    # ✅ Only process folders where the **last folder name** contains 'ROI'
    if "ROI" not in os.path.basename(root):
        continue

    dcm_files = [f for f in files if f.endswith(".dcm")]
    if not dcm_files:
        continue

    full_paths = [os.path.join(root, f) for f in dcm_files]
    black_pixel_info = []

    for path in full_paths:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
            total_pixels = img.size
            black_pixels = np.sum(img == 0)
            black_ratio = black_pixels / total_pixels
            black_pixel_info.append((path, black_ratio))
        except Exception as e:
            print(f"❌ Could not read {path}: {e}")
            continue

    if len(black_pixel_info) == 2:
        sorted_imgs = sorted(black_pixel_info, key=lambda x: x[1], reverse=True)
        data.append((sorted_imgs[0][0], "roi"))
        data.append((sorted_imgs[1][0], "cropped"))

    elif len(black_pixel_info) == 1:
        path, ratio = black_pixel_info[0]
        label = "roi" if ratio > 0.5 else "cropped"
        data.append((path, label))

# Save to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(data)

print(f"✅ Done! Saved {len(data)} labeled entries to {OUTPUT_CSV}")