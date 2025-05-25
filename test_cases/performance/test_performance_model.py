# test_cases/performance/test_performance_model.py

import os
import sys
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from new_predict import predict_mammogram
from segmentation_feature import (
    load_dicom_as_image, contrast_enhancement,
    region_growing, morphological_operations,
    contour_extraction, crop_image_with_contours
)

def test_model_inference_time():
    dicom_path = "test_samples/test_image.dcm"
    assert os.path.exists(dicom_path), "❌ DICOM evaluation_test file not found"

    original = load_dicom_as_image(dicom_path)
    enhanced = contrast_enhancement(original)
    seed = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
    grown = region_growing(enhanced, seed, threshold=15)
    refined = morphological_operations(grown)
    _, contours = contour_extraction(refined)
    cropped = crop_image_with_contours(original, contours)

    start = time.perf_counter()
    result = predict_mammogram(dicom_path, cropped)
    end = time.perf_counter()
    elapsed = end - start

    print("✅ Result:", result)
    print(f"🕒 Inference Time: {elapsed:.3f} sec")
    assert elapsed <= 3.0, f"❌ FAIL: Prediction took too long: {elapsed:.2f} seconds"