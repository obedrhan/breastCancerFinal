import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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

# === RANDOM FOREST + GRID SEARCH ===
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

print("🔍 Starting grid search for Random Forest...")
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_scaled, y)

# === BEST MODEL ===
print(f"\n✅ Best parameters: {grid.best_params_}")
print(f"🎯 Best cross-validation accuracy: {grid.best_score_:.4f}")

# === SAVE MODEL + SCALER ===
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, "rf_gabor_glcm_cropped.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_rf_gabor_glcm_cropped.pkl"))
print("💾 Random Forest model and scaler saved.")