import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"
FULL_PATH = "/test_samples/test_image.dcm"
EMAIL = "testuser2@example.com"
PASSWORD = "Testpass123"
PATIENT_NAME = "Testuser2"
NATIONAL_ID = "INTEG123456"

@pytest.fixture(scope="module")
def test_session():
    session = requests.Session()

    # Register (ignore errors)
    session.post(f"{BASE_URL}/register", json={
        "fullname": "Test User",
        "email": EMAIL,
        "password": PASSWORD
    })

    # Login
    login = session.post(f"{BASE_URL}/login", json={
        "email": EMAIL,
        "password": PASSWORD
    })

    assert login.status_code == 200, f"Login failed: {login.text}"

    # 💡 Manually check session cookie
    print("🔐 Cookies after login:", session.cookies.get_dict())

    return session

def test_process_endpoint(test_session):
    assert os.path.exists(FULL_PATH), f"Missing evaluation_test file at: {FULL_PATH}"

    with open(FULL_PATH, 'rb') as f:
        files = {"file": ("test_image.dcm", f, "application/dicom")}
        data = {
            "patient_name": PATIENT_NAME,
            "national_id": NATIONAL_ID
        }

        response = test_session.post(f"{BASE_URL}/process", files=files, data=data)
        assert response.status_code == 200, f"Failed with: {response.text}"
        result = response.json()
        assert "prediction" in result
        assert "confidence" in result