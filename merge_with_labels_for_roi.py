import pandas as pd
import os

# Load the files
roi_df = pd.read_csv("data/roi_cropped_labels.csv")
cleaned_df = pd.read_csv("data/cleaned_all_files_with_labels.csv")

# Strip paths and normalize
roi_df["relative_path"] = roi_df["image_path"].apply(
    lambda x: "DDSM_IMAGES/" + x.split("DDSM_IMAGES/")[-1]
).str.strip()

# Clean the 'full_path' field and filter only DICOM image rows
cleaned_df["full_path"] = cleaned_df["full_path"].str.strip()
cleaned_df = cleaned_df[cleaned_df["full_path"].str.endswith(".dcm")]

# Now merge
merged = pd.merge(roi_df, cleaned_df[["full_path", "pathology"]], how="left",
                  left_on="relative_path", right_on="full_path")

# Debug print
print(f"\n🔍 Total ROI rows: {len(roi_df)}")
print(f"✅ Found pathology matches: {merged['pathology'].notna().sum()}")
print("❌ Example unmatched rows:")
print(merged[merged['pathology'].isna()].head())

# Save to new file
merged.to_csv("data/roi_cropped_with_pathology.csv", index=False)
print("\n✅ Final CSV saved to: data/roi_cropped_with_pathology.csv")