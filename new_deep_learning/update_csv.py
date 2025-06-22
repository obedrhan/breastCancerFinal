#!/usr/bin/env python3
"""
merge_and_match_cropped_full.py

Read a CSV of cropped ROI DICOM paths, extract patient IDs from the paths,
find corresponding full mammogram entries in metadata, and combine
them into a single CSV with patient_id, view, breast_density, pathology,
full_path, and cropped image_path.
"""

import os
import re
import pandas as pd

# ───────── CONFIG ────────────────────────────────────────
ROOT_DIR   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
ROI_CSV    = os.path.join(ROOT_DIR, "data/roi_cropped_with_pathology.csv")
FULL_CSV   = os.path.join(ROOT_DIR, "data/full_with_correct_roi.csv")
OUTPUT_CSV = os.path.join(ROOT_DIR, "data/final_cropped_full.csv")

# ───────── LOAD AND FILTER CROPPED CSV ──────────────────
# This CSV should contain at least an 'image_path' column and optionally a 'label' column
# Keep only rows where label == 'cropped' if that column exists.
df_cropped = pd.read_csv(ROI_CSV)
if 'label' in df_cropped.columns:
    df_cropped = df_cropped[df_cropped['label'].str.lower() == 'cropped'].copy()

# ───────── EXTRACT PATIENT ID FROM PATH ──────────────────
# Patient IDs follow the pattern P_<digits>, e.g. P_00991
pid_pattern = re.compile(r"(P_\d+)")

def extract_pid(path):
    m = pid_pattern.search(path)
    return m.group(1) if m else None

# Assume cropped CSV uses column 'image_path' for the DICOM paths
df_cropped['patient_id'] = df_cropped['image_path'].apply(extract_pid)
# Drop rows with no match
df_cropped.dropna(subset=['patient_id'], inplace=True)

# ───────── LOAD FULL METADATA ───────────────────────────
# This CSV must contain columns: patient_id, full_path, view, breast_density, pathology
df_full = pd.read_csv(FULL_CSV)
required = ['patient_id', 'full_path', 'view', 'breast_density', 'pathology']
missing = [c for c in required if c not in df_full.columns]
if missing:
    raise KeyError(f"Missing required columns in full CSV: {missing}")

df_full_sel = df_full[required].copy()

# ───────── MERGE ON PATIENT ID ──────────────────────────
# Join cropped and full metadata on patient_id
df_merged = pd.merge(
    df_cropped,
    df_full_sel,
    on='patient_id',
    how='inner'
)
print(f"Merged {len(df_merged)} records for {df_merged['patient_id'].nunique()} unique patients.")

# ───────── SAVE RESULT ──────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_merged.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Final CSV written to: {OUTPUT_CSV}")