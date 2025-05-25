# test_cases/system/test_sys08_ensemble_prediction.py
import os
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
LOGIN_URL = f"{BASE_URL}/login"
PROCESS_URL = f"{BASE_URL}/process"
REGISTER_URL = f"{BASE_URL}/register"

EMAIL = "ensembleuser@example.com"
PASSWORD = "Testpass123"
FULLNAME = "Ensemble User"
PATIENT_NAME = "Patient Ensemble"
NATIONAL_ID = "99001122334"
DICOM_PATH = os.path.abspath("test_samples/test_image.dcm")

@pytest.mark.system
def test_sys08_ensemble_prediction():
    session = requests.Session()

    # Register (ignore duplicate)
    session.post(REGISTER_URL, json={"fullname": FULLNAME, "email": EMAIL, "password": PASSWORD})

    # Login
    res_login = session.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
    assert res_login.status_code == 200, "Login failed"

    # Upload DICOM
    with open(DICOM_PATH, 'rb') as file:
        files = {"file": ("test_image.dcm", file, "application/dicom")}
        data = {"patient_name": PATIENT_NAME, "national_id": NATIONAL_ID}
        response = session.post(PROCESS_URL, files=files, data=data)

    assert response.status_code == 200, f"Prediction failed: {response.text}"
    data = response.json()

    # Check ensemble components
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

    # Optional: log detailed output
    print("✅ Prediction:", data["prediction"])
    print("✅ Confidence:", data["confidence"])

    # Since frontend hides sub-model scores, we can only validate the final ensemble result