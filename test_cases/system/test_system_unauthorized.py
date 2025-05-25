import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_NAME = "Unauthorized User"
TEST_NATIONAL_ID = "00000000000"
DICOM_PATH = os.path.abspath("test_samples/test_image.dcm")

@pytest.mark.system
def test_sys02_unauthorized_access_attempt():
    assert os.path.exists(DICOM_PATH), f"❌ Missing DICOM file: {DICOM_PATH}"

    with open(DICOM_PATH, 'rb') as file:
        files = {
            "file": ("test_image.dcm", file, "application/dicom")
        }
        data = {
            "patient_name": TEST_NAME,
            "national_id": TEST_NATIONAL_ID
        }

        response = requests.post(f"{BASE_URL}/process", files=files, data=data)

        # ✅ Expecting 403 Unauthorized since no login session
        assert response.status_code == 403
        assert "Unauthorized" in response.text