import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_FILE = os.path.join("test_samples", "invalid_file.txt")

EMAIL = "invalidfield@example.com"
PASSWORD = "Testpass123"
PATIENT_NAME = "Invalid File Test"
NATIONAL_ID = "55555555555"

@pytest.fixture(scope="module")
def test_session():
    session = requests.Session()

    # Register (ignore if already exists)
    session.post(f"{BASE_URL}/register", json={
        "fullname": "Invalid Test",
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

def test_invalid_file_upload(test_session):
    assert os.path.exists(TEST_FILE), "❌ Invalid evaluation_test file not found!"

    with open(TEST_FILE, 'rb') as f:
        files = {"file": ("invalid_file.txt", f, "text/plain")}
        data = {
            "patient_name": PATIENT_NAME,
            "national_id": NATIONAL_ID
        }

        response = test_session.post(f"{BASE_URL}/process", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid file type" in response.text or "error" in response.text