import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")
FEATURE_CSV = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_segmented_test.csv")  # Change path as needed

# === Load evaluation_test feature data ===
df = pd.read_csv(FEATURE_CSV)
X_test = df.drop(columns=["Image Name", "Label"]).values
y_test = df["Label"].values

# === Load trained SVM model and scaler ===
model_path = os.path.join(MODEL_DIR, "svm_gabor_glcm_segmented.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler_svm_gabor_glcm_segmented.pkl")  # Make sure this matches your saved file

svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Scale features ===
X_scaled = scaler.transform(X_test)

# === Predict with SVM ===
y_pred = svm_model.predict(X_scaled)

# === Evaluation Metrics ===
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

# === Output Results ===
print("✅ SVM Evaluation Results:")
print(f"\nAccuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)