# test_cases/system/test_sys09_segmented_image_rendering.py
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:5173"
EMAIL = "segmenttest@example.com"
PASSWORD = "Testpass123"
NAME = "Segment Tester"
NATIONAL_ID = "98765432100"
DICOM_PATH = os.path.abspath("test_samples/test_image.dcm")

def test_sys09_segmented_image_rendering():
    options = Options()
    # options.add_argument("--headless")  # Uncomment if running in headless mode
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)

    try:
        # Register
        driver.get(f"{BASE_URL}/register")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "form")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Full Name']").send_keys(NAME)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Email Address']").send_keys(EMAIL)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Create Password']").send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(1)

        # Login
        driver.get(BASE_URL)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Email']")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Email']").send_keys(EMAIL)
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Password']").send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        wait.until(EC.url_contains("/home"))

        # Submit DICOM for classification
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='Patient Full Name']").send_keys("Segment Test")
        driver.find_element(By.CSS_SELECTOR, "input[placeholder='National ID']").send_keys(NATIONAL_ID)
        driver.find_element(By.CSS_SELECTOR, "input[type='file']").send_keys(DICOM_PATH)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        # Wait for segmented result
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "result-box")))
        image = driver.find_element(By.CSS_SELECTOR, "img.segmented-image")
        src = image.get_attribute("src")

        assert src.startswith("http://127.0.0.1:5000/uploads/"), "Image not loaded from backend"
        assert src.endswith(".png"), "Segmented image not in PNG format"

    finally:
        driver.quit()