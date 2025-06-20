#!/usr/bin/env python3
"""
merge_and_dedup.py

Read two CSVs—one with full mammogram paths & patient info,
one with ROI mask paths—merge on patient_id, drop duplicate
patient_ids, and write out a clean CSV with one record per patient.
"""

import os
import pandas as pd

# ───────── CONFIG ────────────────────────────────────────
ROOT_DIR   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
FULL_CSV   = os.path.join(ROOT_DIR, "data/full_with_correct_roi.csv")
ROI_CSV    = os.path.join(ROOT_DIR, "data/roi_mask_paths.csv")
OUTPUT_CSV = os.path.join(ROOT_DIR, "data/final_full_roi.csv")

# ───────── LOAD DATA ─────────────────────────────────────
df_full = pd.read_csv(FULL_CSV)
df_roi  = pd.read_csv(ROI_CSV)

# ───────── SELECT & RENAME COLUMNS ──────────────────────
# From full_correct_with_roi.csv:
df_full_sel = df_full[[
    "patient_id",
    "full_path",
    "view",
    "left or right breast",
    "pathology"
]].copy()

# From roi_mask_paths.csv:
df_roi_sel = df_roi[[
    "patient_id",
    "full_path"
]].rename(columns={"full_path": "roi_path"}).copy()

# ───────── MERGE ON patient_id ──────────────────────────
df_merged = pd.merge(
    df_full_sel,
    df_roi_sel,
    on="patient_id",
    how="inner"
)

print(f"Before dedup: {len(df_merged)} total records")

# ───────── DEDUPLICATE ──────────────────────────────────
# keep='first' → for each patient_id, keep the first row only
df_clean = df_merged.drop_duplicates(subset=["patient_id"], keep="first")
print(f"After  dedup: {len(df_clean)} unique patients")

# ───────── SAVE RESULT ──────────────────────────────────
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Cleaned CSV written to: {OUTPUT_CSV}")