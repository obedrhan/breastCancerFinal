import os
import pandas as pd
import re

# Define base directory
base_dir = "DDSM_IMAGES/CBIS-DDSM"
output_csv = "all_files_in_ddsm.csv"
final_output_csv = "all_files_with_labels.csv"

# Step 1: Recursively scan all folders and files
def walk_and_record_all_files(base_dir):
    all_records = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, base_dir)
            parent_folder = os.path.basename(root)
            all_records.append({
                "folder": parent_folder,
                "filename": file,
                "relative_path": relative_path,
                "full_path": full_path
            })
    return pd.DataFrame(all_records)

# Step 2: Load description datasets
calc_test_path = "DDSM_IMAGES/calc_case_description_test_set.csv"
calc_train_path = "DDSM_IMAGES/calc_case_description_train_set.csv"
mass_test_path = "DDSM_IMAGES/mass_case_description_test_set.csv"
mass_train_path = "DDSM_IMAGES/mass_case_description_train_set.csv"

description_dfs = []
for path in [calc_test_path, calc_train_path, mass_test_path, mass_train_path]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "image file path" in df.columns:
            df.rename(columns={"image file path": "full_image_path"}, inplace=True)
        description_dfs.append(df)

# Combine description CSVs
description_df = pd.concat(description_dfs, ignore_index=True)

# Extract patient ID and view from description_df
def extract_patient_id(path):
    match = re.search(r'P_\d+', str(path))
    return match.group(0) if match else None

def extract_view(path):
    if "CC" in str(path):
        return "CC"
    elif "MLO" in str(path):
        return "MLO"
    else:
        return "UNKNOWN"

description_df["patient_id"] = description_df["full_image_path"].apply(extract_patient_id)
description_df["view"] = description_df["full_image_path"].apply(extract_view)

# Step 3: Scan directory and record all files
ddsm_df = walk_and_record_all_files(base_dir)
ddsm_df.to_csv(output_csv, index=False)
print(f"✅ All file paths and names recorded to {output_csv}")

# Step 4: Extract patient_id and view from file structure
ddsm_df["patient_id"] = ddsm_df["relative_path"].apply(extract_patient_id)
ddsm_df["view"] = ddsm_df["relative_path"].apply(extract_view)

# Step 5: Optimized merge using patient_id and view
final_df = ddsm_df.merge(description_df, on=["patient_id", "view"], how="left")
final_df.to_csv(final_output_csv, index=False)
print(f"✅ Labeled dataset created: {final_output_csv}")