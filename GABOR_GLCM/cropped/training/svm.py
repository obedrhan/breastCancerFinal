import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# === CONFIGURATION ===
FEATURE_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/data/glcm_gabor_features_cropped_training.csv"
MODEL_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
print("📦 Loading feature dataset...")
df = pd.read_csv(FEATURE_CSV)

X = df.drop(columns=["Image Name", "Label"]).values
y = df["Label"].values

print(f"✅ Feature shape: X={X.shape}, y={y.shape}")

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SVM + GRID SEARCH ===
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

print("🔍 Starting grid search for SVM...")
svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_scaled, y)

# === BEST MODEL ===
print(f"\n✅ Best parameters: {grid.best_params_}")
print(f"🎯 Best cross-validation accuracy: {grid.best_score_:.4f}")

# === SAVE MODEL + SCALER ===
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, "svm_gabor_glcm_cropped.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_svm_gabor_glcm_cropped.pkl"))
print("💾 SVM model and scaler saved.")