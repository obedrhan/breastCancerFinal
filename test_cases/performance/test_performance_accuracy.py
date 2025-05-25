# test_cases/performance/test_performance_accuracy.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, recall_score
from new_predict import predict_mammogram  # must include model loading
from tqdm import tqdm
import pydicom

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")

@pytest.mark.performance
def test_model_accuracy_on_test_set():
    print("📥 Loading full_mammogram_paths.csv ...")
    df = pd.read_csv(CSV_PATH)
    df['pathology'] = df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    df['label'] = df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})
    df = df[df["full_path"].str.contains("Test")]
    print(f"🔍 Found {len(df)} test samples")

    y_true = []
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔎 Running predictions"):
        raw_path = row["full_path"]
        corrected_path = raw_path.replace("DDSM_IMAGES/CBIS-DDSM/DDSM_IMAGES", "DDSM_IMAGES")
        full_path = os.path.join(BASE_DIR, corrected_path)

        if not os.path.exists(full_path):
            print(f"⚠️ Missing: {full_path}")
            continue

        try:
            dicom = pydicom.dcmread(full_path, force=True)
            image = dicom.pixel_array.astype(np.uint8)
            crop = image[100:228, 100:228]  # dummy ROI patch
            result = predict_mammogram(full_path, crop)
            pred = 1 if result["prediction"].lower() == "malignant" else 0

            y_true.append(row["label"])
            y_pred.append(pred)
        except Exception as e:
            print(f"❌ Error processing {full_path}: {e}")
            continue

    print("\n✅ Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"])
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    recall_malignant = recall_score(y_true, y_pred, pos_label=1)

    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Malignant Recall: {recall_malignant * 100:.2f}%")

    # Assertions for performance goals
    assert accuracy >= 0.85, f"❌ Accuracy too low: {accuracy:.2f}"
    assert recall_malignant >= 0.80, f"❌ Recall for malignant too low: {recall_malignant:.2f}"