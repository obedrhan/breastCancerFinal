# test_cases/system/test_sys05_view_previous.py
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
VIEW_URL = f"{BASE_URL}/view_previous"

@pytest.mark.system
def test_sys05_view_previous_results():
    session = requests.Session()

    # Login first
    login_res = session.post(f"{BASE_URL}/login", json={
        "email": "testuser@example.com",
        "password": "Testpass123"
    })
    assert login_res.status_code == 200, "❌ Login failed"

    # Request previous results (no patient_name filter)
    res = session.post(VIEW_URL, json={})
    assert res.status_code == 200, "❌ View previous results failed"

    data = res.json()
    assert isinstance(data, list), "Response should be a list"
    if data:
        assert "prediction" in data[0]
        assert "confidence" in data[0]
        assert "fullname" in data[0]