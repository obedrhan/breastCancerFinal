import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === CONFIGURATION ===
TEST_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/data/glcm_gabor_features_cropped_test.csv"
MODEL_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/models"
MODEL_PATH = os.path.join(MODEL_DIR, "knn_gabor_glcm_cropped.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_knn_gabor_glcm_cropped.pkl")

# === LOAD TEST DATA ===
print("📄 Loading evaluation_test dataset...")
df = pd.read_csv(TEST_CSV)
X_test = df.drop(columns=["Image Name", "Label"]).values
y_test = df["Label"].values

# === LOAD MODEL & SCALER ===
print("🔄 Loading trained KNN model and scaler...")
knn_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === SCALE TEST DATA ===
X_test_scaled = scaler.transform(X_test)

# === PREDICT & EVALUATE ===
print("🤖 Making predictions...")
y_pred = knn_model.predict(X_test_scaled)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))