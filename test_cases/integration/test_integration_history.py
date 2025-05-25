import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
DICOM_1 = os.path.join("test_samples", "test_image1.dcm")
DICOM_2 = os.path.join("test_samples", "test_image2.dcm")

EMAIL = "historytest@example.com"
PASSWORD = "Testpass123"
PATIENT_NAME = "History Patient"
NATIONAL_ID = "66666666666"

@pytest.fixture(scope="module")
def test_session():
    session = requests.Session()

    # Register (ignore duplicate)
    session.post(f"{BASE_URL}/register", json={
        "fullname": "History Test User",
        "email": EMAIL,
        "password": PASSWORD
    })

    # Login
    login_res = session.post(f"{BASE_URL}/login", json={
        "email": EMAIL,
        "password": PASSWORD
    })
    assert login_res.status_code == 200
    return session

def test_multiple_classifications_returned(test_session):
    for path in [DICOM_1, DICOM_2]:
        assert os.path.exists(path), f"❌ Missing file: {path}"

        with open(path, 'rb') as f:
            files = {"file": (os.path.basename(path), f, "application/dicom")}
            data = {
                "patient_name": PATIENT_NAME,
                "national_id": NATIONAL_ID
            }
            res = test_session.post(f"{BASE_URL}/process", files=files, data=data)
            assert res.status_code == 200, f"Classification failed for: {path}"

    # Check view_previous
    res = test_session.post(f"{BASE_URL}/view_previous", json={
        "patient_name": PATIENT_NAME
    })
    assert res.status_code == 200
    results = res.json()

    # Check that there are at least 2 entries for the same patient
    matched = [r for r in results if r["fullname"] == PATIENT_NAME]
    assert len(matched) >= 2, "Expected at least 2 classification entries"