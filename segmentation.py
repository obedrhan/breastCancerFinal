import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from collections import deque
from skimage.io import imsave

# ------------------------------
# 📁 PATHS
# ------------------------------
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
csv_path = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")
output_folder = os.path.join(BASE_DIR, "segmented_Test_output")
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# 🔧 Image Processing Functions
# ------------------------------
def load_dicom_as_image(dicom_path):
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        pixel_array = dicom.pixel_array.astype(np.float32)
        pixel_array -= pixel_array.min()
        pixel_array /= (pixel_array.max() + 1e-6)
        pixel_array *= 255.0
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"❌ Error reading {dicom_path}: {e}")
        return None

def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.medianBlur(enhanced, 5)
    return enhanced

def region_growing(image, seed_point, threshold=8):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    stack = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

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

def morphological_operations(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1000, 1000))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned_image

def contour_extraction(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, 255, 2)
    return contour_image, contours

def crop_image_with_contours(original_image, contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return original_image[y:y+h, x:x+w]
    return original_image

# ------------------------------
# 🚀 MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        relative_path = row['full_path']
        image_path = os.path.join(BASE_DIR, relative_path)

        if 'Test' not in relative_path:
            continue

        print(f"[{idx + 1}/{len(df)}] Processing: {image_path}")

        if not os.path.exists(image_path):
            print(f"⚠️ Skipping (file not found): {image_path}")
            continue

        if image_path.lower().endswith('.dcm'):
            original_image = load_dicom_as_image(image_path)
            if original_image is None:
                continue

            enhanced = contrast_enhancement(original_image)
            seed_point = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
            grown = region_growing(enhanced, seed_point)
            refined = morphological_operations(grown)
            _, contours = contour_extraction(refined)
            cropped = crop_image_with_contours(original_image, contours)

            # 🔠 Save the image using relative path format as filename
            sanitized_name = relative_path.replace("/", "_")
            out_filename = f"{sanitized_name}_segmented.png"
            out_path = os.path.join(output_folder, out_filename)
            imsave(out_path, cropped)
            print(f"✅ Saved: {out_path}")