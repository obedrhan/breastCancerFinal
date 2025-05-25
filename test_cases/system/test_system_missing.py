import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
PROCESS_URL = f"{BASE_URL}/process"

@pytest.mark.system
def test_sys03_upload_missing_fields():
    session = requests.Session()

    # First login to get session
    login_res = session.post(f"{BASE_URL}/login", json={
        "email": "testuser@example.com",
        "password": "Testpass123"
    })
    assert login_res.status_code == 200, "Login failed for setup"

    # Prepare only partial form (missing patient_name and file)
    data = {
        "national_id": "12345678900"
        # Missing: patient_name and file
    }

    response = session.post(PROCESS_URL, data=data)

    # Expect 400 Bad Request due to missing fields
    assert response.status_code == 400
    assert "Missing required fields" in response.text