# test_cases/system/test_sys07_logout_restriction.py
import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"
LOGIN_URL = f"{BASE_URL}/login"
LOGOUT_URL = f"{BASE_URL}/logout"
VIEW_URL = f"{BASE_URL}/view_previous"

EMAIL = "testuser@example.com"
PASSWORD = "Testpass123"

@pytest.mark.system
def test_sys07_logout_flow_and_restriction():
    session = requests.Session()

    # Step 1: Login
    res_login = session.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
    assert res_login.status_code == 200, "❌ Login failed"

    # Step 2: Access /view_previous (should work)
    res_view_before = session.post(VIEW_URL, json={})
    assert res_view_before.status_code == 200, "❌ View previous failed before logout"

    # Step 3: Logout
    res_logout = session.post(LOGOUT_URL)
    assert res_logout.status_code == 200, "❌ Logout failed"

    # Step 4: Access /view_previous again (should be unauthorized)
    res_view_after = session.post(VIEW_URL, json={})
    assert res_view_after.status_code == 403, "❌ Protected access not blocked after logout"