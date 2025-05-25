import sys
import os
import numpy as np
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation_feature import (
    load_dicom_as_image,
    compute_lbp
)

def test_load_dicom_valid(tmp_path):
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import datetime
    import pydicom

    filename = tmp_path / "evaluation_test.dcm"
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelData = (np.ones((10, 10), dtype=np.uint16) * 300).tobytes()
    ds.SamplesPerPixel = 1
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.StudyDate = str(datetime.date.today()).replace("-", "")
    ds.Modality = "MG"
    ds.save_as(str(filename))

    image = load_dicom_as_image(str(filename))
    assert image is not None
    assert image.shape == (10, 10)
    assert image.dtype == np.uint8

def test_load_dicom_invalid():
    result = load_dicom_as_image("non_existent.dcm")
    assert result is None

def test_lbp_output_length_and_sum():
    image = (np.random.rand(64, 64) * 255).astype(np.uint8)
    hist = compute_lbp(image, radius=1, n_points=8)
    assert isinstance(hist, np.ndarray)
    assert hist.shape[0] > 0
    np.testing.assert_almost_equal(np.sum(hist), 1.0, decimal=5)

def test_lbp_uniformity():
    image = np.ones((32, 32), dtype=np.uint8) * 150
    hist = compute_lbp(image)
    assert hist is not None
    assert hist.max() <= 1.0