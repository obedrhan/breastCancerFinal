import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")
FEATURE_CSV = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_cropped_test.csv")  # Adjust if needed

# === Load evaluation_test data ===
df = pd.read_csv(FEATURE_CSV)
X_test = df.drop(columns=["Image Name", "Label"]).values
y_test = df["Label"].values

# === Load model and scaler ===
model_path = os.path.join(MODEL_DIR, "rf_gabor_glcm_cropped.pkl")  # Make sure your model is saved with this name
scaler_path = os.path.join(MODEL_DIR, "scaler_rf_gabor_glcm_cropped.pkl")

rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Scale evaluation_test features ===
X_scaled = scaler.transform(X_test)

# === Prediction ===
y_pred = rf_model.predict(X_scaled)

# === Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

# === Output Results ===
print("✅ Random Forest Evaluation Results:")
print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)