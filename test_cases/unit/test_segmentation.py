import os
import sys
import numpy as np
import pytest

# Add parent directory to sys.path to resolve segmentation_feature import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation_feature import (
    load_dicom_as_image,
    contrast_enhancement,
    region_growing,
    morphological_operations,
    contour_extraction,
    crop_image_with_contours
)

# Sample evaluation_test image path (update this to match a real training file path in your dataset)
SAMPLE_DICOM = "DDSM/full_mammogram_paths.csv"  # for example, replace this with a real .dcm file path

@pytest.mark.skipif(not os.path.exists(SAMPLE_DICOM), reason="Sample DICOM not found")
def test_load_dicom():
    img = load_dicom_as_image(SAMPLE_DICOM)
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2

def test_contrast_enhancement_on_dummy():
    dummy = np.uint8(np.random.randint(0, 256, (512, 512)))
    enhanced = contrast_enhancement(dummy)
    assert enhanced.shape == dummy.shape
    assert not np.array_equal(dummy, enhanced)

def test_region_growing_output():
    dummy = np.full((100, 100), 100, dtype=np.uint8)
    dummy[50, 50] = 110  # seed is brighter
    result = region_growing(dummy, (50, 50), threshold=15)
    assert result.shape == dummy.shape
    assert result.dtype == np.uint8
    assert result[50, 50] == 255

def test_morphological_operations_on_simple_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255
    cleaned = morphological_operations(mask)
    assert cleaned.shape == mask.shape
    assert cleaned.max() <= 255

def test_contour_extraction_returns_contours():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2 = __import__('cv2')  # avoid import error in environments without OpenCV
    cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)
    contour_img, contours = contour_extraction(mask)
    assert contour_img.shape == mask.shape
    assert len(contours) > 0

def test_crop_image_with_contours_returns_region():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(mask, (20, 20), (60, 60), 255, -1)
    _, contours = contour_extraction(mask)
    cropped = crop_image_with_contours(img, contours)
    assert cropped.ndim == 2
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0