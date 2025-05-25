import numpy as np
import pytest
from skimage.feature import hog
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dummy grayscale image for HOG extraction
@pytest.fixture
def dummy_image():
    image = data.astronaut()
    gray = rgb2gray(image)
    resized = resize(gray, (128, 128), anti_aliasing=True)
    return resized


def test_hog_feature_extraction_output_shape(dummy_image):
    features = hog(dummy_image,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)

    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    assert len(features) > 0


def test_hog_no_nan(dummy_image):
    features = hog(dummy_image,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)

    assert not np.isnan(features).any()