import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# === Configuration ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_segmented_training.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load features and labels ===
print("📂 Loading feature file...")
df = pd.read_csv(CSV_PATH)

print("🔎 Raw label values:", df["Label"].unique())
df = df[df["Label"].isin([0, 1])]
X = df.drop(columns=["Image Name", "Label"]).values
y = df["Label"].values
print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.")

# === Feature scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Random Forest training with GridSearch ===
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"]
}

print("🤖 Training Random Forest with grid search...")
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_scaled, y)

print("🏆 Best Params:", grid.best_params_)
print("✅ Best CV Accuracy:", grid.best_score_)

# === Save model and scaler ===
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, "rf_gabor_glcm_segmented.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_rf_gabor_glcm_segmented.pkl"))
print(f"💾 Model and scaler saved to: {MODEL_DIR}")