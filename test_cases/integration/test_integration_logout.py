import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_EMAIL = "logout_test_user@example.com"
TEST_PASSWORD = "Test12345!"
TEST_FULLNAME = "Logout Tester"
TEST_DCM_PATH = os.path.join("test_samples", "test_image.dcm")

@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    # Register user (if not already exists)
    s.post(f"{BASE_URL}/register", json={
        "fullname": TEST_FULLNAME,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })

    # Login user
    login = s.post(f"{BASE_URL}/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })
    assert login.status_code == 200
    return s

def test_logout_and_blocked_access(session):
    # Step 1: Logout
    logout = session.post(f"{BASE_URL}/logout")
    assert logout.status_code == 200
    assert "Logged out" in logout.text or logout.json().get("message")

    # Step 2: Try protected route after logout
    if not os.path.exists(TEST_DCM_PATH):
        pytest.skip("Missing evaluation_test image for /process check")

    with open(TEST_DCM_PATH, 'rb') as f:
        files = {"file": ("test_image.dcm", f, "application/dicom")}
        data = {
            "patient_name": "Logout Dummy",
            "national_id": "999999999"
        }
        response = session.post(f"{BASE_URL}/process", files=files, data=data)

    assert response.status_code == 403
    assert "Unauthorized" in response.text