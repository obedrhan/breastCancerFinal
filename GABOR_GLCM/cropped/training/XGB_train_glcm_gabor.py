import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
CSV_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/data/glcm_gabor_features_cropped_training.csv"
MODEL_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/Gabor_GLCM/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
df = df.dropna()
df["Label"] = df["Label"].astype(int)

X = df.drop(columns=["Image Name", "Label"]).values
y = df["Label"].values

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === XGBOOST TRAINING ===
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

grid = GridSearchCV(xgb_clf, params, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X_scaled, y)

# === SAVE MODEL AND SCALER ===
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, "xgboost_glcm_gabor_segmented.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_glcm_gabor_xg_segmented.pkl"))

print("✅ Model and scaler saved.")
print(f"✅ Best Parameters: {grid.best_params_}")