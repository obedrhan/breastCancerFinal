# test_cases/system/test_sys04_unsupported_file.py
import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
PROCESS_URL = f"{BASE_URL}/process"
INVALID_FILE_PATH = os.path.abspath("test_samples/invalid_file.txt")

@pytest.mark.system
def test_sys04_upload_unsupported_file_type():
    session = requests.Session()

    # Login first
    login_res = session.post(f"{BASE_URL}/login", json={
        "email": "testuser@example.com",
        "password": "Testpass123"
    })
    assert login_res.status_code == 200, "Login failed"

    # Prepare form data with unsupported .txt file
    with open(INVALID_FILE_PATH, "rb") as f:
        files = {"file": ("invalid_file.txt", f, "text/plain")}
        data = {
            "patient_name": "Invalid Upload",
            "national_id": "00000000000"
        }

        response = session.post(PROCESS_URL, files=files, data=data)

    # Expect rejection with 400 Bad Request
    assert response.status_code == 400
    assert "Invalid file type" in response.text