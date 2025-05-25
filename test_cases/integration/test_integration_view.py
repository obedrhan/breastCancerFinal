import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"
EMAIL = "viewtest@example.com"
PASSWORD = "Testpass123"
PATIENT_NAME = "View Test"
NATIONAL_ID = "99999999999"

@pytest.fixture(scope="module")
def test_session():
    session = requests.Session()

    # Register
    session.post(f"{BASE_URL}/register", json={
        "fullname": PATIENT_NAME,
        "email": EMAIL,
        "password": PASSWORD
    })

    # Login
    response = session.post(f"{BASE_URL}/login", json={
        "email": EMAIL,
        "password": PASSWORD
    })
    assert response.status_code == 200, f"Login failed: {response.text}"
    return session

def test_view_previous_results(test_session):
    response = test_session.post(f"{BASE_URL}/view_previous", json={
        "patient_name": PATIENT_NAME
    })

    assert response.status_code == 200, f"View failed: {response.text}"
    results = response.json()
    assert isinstance(results, list), "Results should be a list"

    if results:
        result = results[0]
        for key in ["prediction", "confidence", "fullname", "date"]:
            assert key in result, f"Missing key: {key}"
