import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_DCM = os.path.join("test_samples", "test_image.dcm")  # Update this path if necessary

EMAIL = "missingtest@example.com"
PASSWORD = "Testpass123"
FULLNAME = "Missing Field Tester"


@pytest.fixture(scope="module")
def test_session():
    s = requests.Session()

    # Register
    s.post(f"{BASE_URL}/register", json={
        "fullname": FULLNAME,
        "email": EMAIL,
        "password": PASSWORD
    })

    # Login
    r = s.post(f"{BASE_URL}/login", json={
        "email": EMAIL,
        "password": PASSWORD
    })
    assert r.status_code == 200
    return s


def test_missing_fields_upload(test_session):
    assert os.path.exists(TEST_DCM), "❌ DICOM evaluation_test file not found!"

    # Only upload the file, omit patient_name and national_id
    with open(TEST_DCM, 'rb') as f:
        files = {"file": ("test_image.dcm", f, "application/dicom")}
        response = test_session.post(f"{BASE_URL}/process", files=files)

    assert response.status_code == 400
    assert "Missing required fields" in response.text