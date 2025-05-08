import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")
FEATURE_CSV = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_cropped_test.csv")  # change path if needed

# === Load test features ===
df = pd.read_csv(FEATURE_CSV)
X_test = df.drop(columns=["Image Name", "Label"]).values
y_test = df["Label"].values

# === Load trained model and scaler ===
model_path = os.path.join(MODEL_DIR, "knn_gabor_glcm_cropped.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler_gabor_glcm_cropped.pkl")  # adjust if your scaler is named differently

knn = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Scale features ===
X_scaled = scaler.transform(X_test)

# === Predict ===
y_pred = knn.predict(X_scaled)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

print("✅ KNN Evaluation Results:")
print(f"\nAccuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)