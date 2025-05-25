# test_cases/test_unauthorized_process.py

import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
FULL_PATH = "/test_samples/test_image.dcm"  # Adjust path as needed

@pytest.mark.integration
def test_process_without_login():
    assert os.path.exists(FULL_PATH), f"Missing evaluation_test DICOM file at {FULL_PATH}"

    with open(FULL_PATH, 'rb') as f:
        files = {"file": ("test_image.dcm", f, "application/dicom")}
        data = {
            "patient_name": "Test NoLogin",
            "national_id": "00000000000"
        }

        response = requests.post(f"{BASE_URL}/process", files=files, data=data)

    assert response.status_code == 403
    assert "unauthorized" in response.text.lower()