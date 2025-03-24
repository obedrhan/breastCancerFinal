import os
import pandas as pd
import numpy as np
import joblib
import cv2
import pydicom
from collections import deque
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------- SEGMENTATION & FEATURE EXTRACTION ---------------------------- #

def load_dicom_as_image(dicom_path):
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        pixel_array = dicom.pixel_array.astype(np.float32)
        pixel_array -= pixel_array.min()
        pixel_array /= (pixel_array.max() + 1e-6)
        pixel_array *= 255.0
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"❌ Skipping {dicom_path} due to read error: {e}")
        return None

def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.medianBlur(enhanced, 5)
    return enhanced

def morphological_operations(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1000, 1000))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned_image

def region_growing(image, seed_point, threshold=8):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    stack = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        x, y = stack.pop()
        if segmented[y, x] == 0:
            segmented[y, x] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_intensity)) <= threshold:
                        stack.append((nx, ny))
    return segmented

def contour_extraction(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, 255, 2)
    return contour_image, contours

def crop_image_with_contours(original_image, contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_region = original_image[y:y+h, x:x+w]
        return cropped_region
    return original_image

def compute_lbp(image, radius=1, n_points=8):
    image = (image / image.max() * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

def process_test_image(filepath):
    print(f"\n➡️ Segmenting and extracting features from: {filepath}")
    original_image = load_dicom_as_image(filepath)
    if original_image is None:
        return None
    contrast_image = contrast_enhancement(original_image)
    seed_point = (contrast_image.shape[1] // 2, contrast_image.shape[0] // 2)
    region_grown = region_growing(contrast_image, seed_point, threshold=15)
    refined_image = morphological_operations(region_grown)
    _, contours = contour_extraction(refined_image)
    cropped_image = crop_image_with_contours(original_image, contours)
    lbp_hist = compute_lbp(cropped_image)
    return lbp_hist

# ---------------------------- MAIN SCRIPT ---------------------------- #

if __name__ == "__main__":
    BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models"
    test_csv = os.path.join(BASE_DIR, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/full_mammogram_paths.csv")

    df = pd.read_csv(test_csv)
    df = df[df['full_path'].str.contains('Test')].reset_index(drop=True)

    features = []
    labels = []
    paths = []
    skipped_count = 0

    for idx, row in df.iterrows():
        image_path = os.path.join(BASE_DIR, row['full_path'])
        label = row['pathology']

        if not os.path.exists(image_path):
            print(f"⚠️ Skipping missing file: {image_path}")
            skipped_count += 1
            continue

        lbp_hist = process_test_image(image_path)
        if lbp_hist is None:
            skipped_count += 1
            continue

        features.append(lbp_hist)
        labels.append(label)
        paths.append(image_path)

    X_test = np.array(features)

    y_test = []
    for label in labels:
        if "malignant" in label.lower():
            y_test.append("malignant")
        elif "benign" in label.lower():
            y_test.append("benign")
        else:
            y_test.append("normal")

    # Save extracted features to CSV
    lbp_df = pd.DataFrame(features)
    lbp_df.insert(0, "Image Path", paths)
    lbp_df["Label"] = y_test
    lbp_df.to_csv(os.path.join(BASE_DIR, "test_lbp_features.csv"), index=False)
    print("✅ Saved extracted test LBP features to test_lbp_features.csv")

    models = ["knn_model_ddsm.pkl", "svm_model_ddsm.pkl", "random_forest_model_ddsm.pkl"]

    for model_file in models:
        print(f"\n🔍 Loading and evaluating {model_file}...")
        model = joblib.load(os.path.join(BASE_DIR, model_file))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler_ddsm.pkl"))
        X_test_scaled = scaler.transform(X_test)

        print(f"✅ Evaluation Results for {model_file}:")
        y_pred = model.predict(X_test_scaled)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Accuracy:", accuracy_score(y_test, y_pred))

    print(f"\nℹ️ Total skipped images due to errors or missing files: {skipped_count}")
