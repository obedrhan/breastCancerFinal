import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from new_predict import predict_mammogram

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

@pytest.fixture
def dummy_dicom_and_crop():
    dicom_path = os.path.join(BASE_DIR, "test_samples/test_image.dcm")  # Must exist
    crop = np.ones((128, 128), dtype=np.uint8) * 150  # Dummy cropped region
    return dicom_path, crop

def test_predict_output_structure(dummy_dicom_and_crop):
    dicom_path, crop = dummy_dicom_and_crop
    result = predict_mammogram(dicom_path, crop)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence" in result
    assert "random_forest" in result
    assert "densenet" in result
    assert "efficientnet" in result

def test_prediction_confidence_range(dummy_dicom_and_crop):
    dicom_path, crop = dummy_dicom_and_crop
    result = predict_mammogram(dicom_path, crop)

    for key in ["confidence", "random_forest", "densenet", "efficientnet"]:
        assert 0.0 <= result[key] <= 1.0, f"{key} out of bounds"