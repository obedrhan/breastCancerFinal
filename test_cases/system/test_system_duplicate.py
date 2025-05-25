# test_cases/system/test_sys06_duplicate_upload.py
import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"
PROCESS_URL = f"{BASE_URL}/process"
LOGIN_URL = f"{BASE_URL}/login"

EMAIL = "testuser@example.com"
PASSWORD = "Testpass123"
PATIENT_NAME = "Duplicate Test"
NATIONAL_ID = "22222222222"
DICOM_FILE = os.path.abspath("test_samples/test_image.dcm")

@pytest.mark.system
def test_sys06_duplicate_upload():
    session = requests.Session()

    # Step 1: Login
    login = session.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
    assert login.status_code == 200, "❌ Login failed"

    for i in range(2):  # Try uploading twice
        with open(DICOM_FILE, 'rb') as f:
            files = {"file": ("test_image.dcm", f, "application/dicom")}
            data = {
                "patient_name": PATIENT_NAME,
                "national_id": NATIONAL_ID
            }
            response = session.post(PROCESS_URL, files=files, data=data)
            assert response.status_code == 200, f"❌ Upload #{i+1} failed"
            print(f"✅ Upload #{i+1} success")