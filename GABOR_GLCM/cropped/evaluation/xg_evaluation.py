import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
TEST_CSV = os.path.join(BASE_DIR, "Gabor_GLCM/data/glcm_gabor_features_cropped_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "GABOR_GLCM/models/xgboost_glcm_gabor_cropped.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "GABOR_GLCM/models/scaler_glcm_gabor_xg_cropped.pkl")

# === Load Test Data ===
df_test = pd.read_csv(TEST_CSV)
X_test = df_test.drop(columns=["Image Name", "Label"]).values
y_test = df_test["Label"].values

# === Load Model & Scaler ===
xgb_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Preprocess Test Features ===
X_test_scaled = scaler.transform(X_test)

# === Predict ===
y_pred = xgb_model.predict(X_test_scaled)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

print("✅ XGBoost Evaluation Results")
print(f"Accuracy: {acc:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)