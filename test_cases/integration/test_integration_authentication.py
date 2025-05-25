import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_EMAIL = "integration_user@example.com"
TEST_PASSWORD = "SecurePass123"
TEST_FULLNAME = "Integration Test User"

@pytest.fixture(scope="module")
def session():
    return requests.Session()

def test_registration_and_login(session):
    # Register user
    register_response = session.post(f"{BASE_URL}/register", json={
        "fullname": TEST_FULLNAME,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })

    # Accept both new registration and "already registered" case
    assert register_response.status_code in [200, 400], f"Register failed: {register_response.text}"
    if register_response.status_code == 400:
        assert "already registered" in register_response.text.lower()

    # Login user
    login_response = session.post(f"{BASE_URL}/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })

    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    assert "Logged in" in login_response.text or "message" in login_response.json()