import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"
TEST_DCM = os.path.join("test_samples", "test_image.dcm")  # Update path if needed

EMAIL = "dupuser@example.com"
PASSWORD = "Testpass123"
FULLNAME = "Dup Test"
NATIONAL_ID = "22222222222"
PATIENT_NAME = "Duplicate Patient"

@pytest.fixture(scope="module")
def test_session():
    s = requests.Session()

    # Register (ignore if already registered)
    s.post(f"{BASE_URL}/register", json={
        "fullname": FULLNAME,
        "email": EMAIL,
        "password": PASSWORD
    })

    # Login
    resp = s.post(f"{BASE_URL}/login", json={
        "email": EMAIL,
        "password": PASSWORD
    })
    assert resp.status_code == 200
    return s

def test_duplicate_upload_creates_new_entry(test_session):
    assert os.path.exists(TEST_DCM), "Missing evaluation_test DICOM file"

    def upload_image():
        with open(TEST_DCM, "rb") as f:
            files = {"file": ("test_image.dcm", f, "application/dicom")}
            data = {"patient_name": PATIENT_NAME, "national_id": NATIONAL_ID}
            return test_session.post(f"{BASE_URL}/process", files=files, data=data)

    # First upload
    r1 = upload_image()
    assert r1.status_code == 200

    # Second upload with same ID
    r2 = upload_image()
    assert r2.status_code == 200

    # Verify /view_previous has 2 records
    r3 = test_session.post(f"{BASE_URL}/view_previous", json={"patient_name": PATIENT_NAME})
    assert r3.status_code == 200
    results = r3.json()
    matches = [r for r in results if r["fullname"] == PATIENT_NAME]
    assert len(matches) >= 2, "Duplicate upload should create multiple entries"