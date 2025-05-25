import os
import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:5173"
TEST_EMAIL = "testuser@example.com"
TEST_PASSWORD = "Testpass123"
TEST_NAME = "Test User"
TEST_NATIONAL_ID = "12345678900"
DICOM_PATH = os.path.abspath("test_samples/test_image.dcm")

@pytest.mark.system
def test_sys01_classification_flow():
    options = Options()
    # Uncomment to run headlessly:
    # options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 15)

    try:
        # Step 1: Go to Register page
        driver.get(f"{BASE_URL}/register")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "form")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Full Name']").send_keys(TEST_NAME)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Email Address']").send_keys(TEST_EMAIL)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Create Password']").send_keys(TEST_PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(1)

        # Step 2: Go to Login page
        driver.get(f"{BASE_URL}/")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Email']")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Email']").send_keys(TEST_EMAIL)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Password']").send_keys(TEST_PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        # Step 3: Wait for home page
        wait.until(EC.url_contains("/home"))

        # Step 4: Fill upload form
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='National ID']")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='National ID']").send_keys(TEST_NATIONAL_ID)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Patient Full Name']").send_keys("Sample Patient")
        driver.find_element(By.CSS_SELECTOR, "input[type='file']").send_keys(DICOM_PATH)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        # Step 5: Wait for result
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "result-box")))
        assert "Prediction" in driver.page_source
        assert "Confidence" in driver.page_source

    finally:
        driver.quit()