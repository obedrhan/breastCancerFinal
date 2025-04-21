import csv
import os

CSV_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/roi_cropped_labels.csv"

# Counters for 1-1.dcm
roi_1_1 = 0
cropped_1_1 = 0
total_1_1 = 0

# Counters for 1-2.dcm
roi_1_2 = 0
cropped_1_2 = 0
total_1_2 = 0

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row["image_path"]
        label = row["label"]
        filename = os.path.basename(path)

        if filename == "1-1.dcm":
            total_1_1 += 1
            if label == "roi":
                roi_1_1 += 1
            elif label == "cropped":
                cropped_1_1 += 1

        elif filename == "1-2.dcm":
            total_1_2 += 1
            if label == "roi":
                roi_1_2 += 1
            elif label == "cropped":
                cropped_1_2 += 1

# Output the results
print("📊 Analysis of ROI vs Cropped labels in roi_cropped_labels.csv\n")

print("🔹 For 1-1.dcm:")
print(f"   🟢 ROI: {roi_1_1}")
print(f"   🟡 Cropped: {cropped_1_1}")
print(f"   📁 Total 1-1.dcm: {total_1_1}\n")

print("🔸 For 1-2.dcm:")
print(f"   🟢 ROI: {roi_1_2}")
print(f"   🟡 Cropped: {cropped_1_2}")
print(f"   📁 Total 1-2.dcm: {total_1_2}")