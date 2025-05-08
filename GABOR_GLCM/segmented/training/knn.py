import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# === CONFIGURATION ===
FEATURE_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/data/glcm_gabor_features_segmented_training.csv"
MODEL_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD FEATURES ===
print("📦 Loading extracted features...")
df = pd.read_csv(FEATURE_CSV)

X = df.drop(columns=["Image Name", "Label"]).values
y = df["Label"].values

print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")

# === FEATURE SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === GRID SEARCH + CROSS VALIDATION ===
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

print("🔍 Performing grid search...")
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_scaled, y)

print(f"\n✅ Best parameters: {grid.best_params_}")
print(f"🎯 Best cross-validation accuracy: {grid.best_score_:.4f}")

# === SAVE MODEL + SCALER ===
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, "knn_gabor_glcm_segmented.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_knn_gabor_glcm_segmented.pkl"))
print("💾 Model and scaler saved.")