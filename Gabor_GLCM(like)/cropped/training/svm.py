import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
FEATURE_CSV = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_cropped_training.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_gabor_glcm_cropped.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_svm_gabor_glcm_cropped.pkl")

# === Load and preprocess data ===
def load_data(path):
    df = pd.read_csv(path)
    print(f"📊 Loaded feature shape: {df.shape}")

    X = df.drop(columns=["Image Name", "Label"]).values
    y = df["Label"].values

    return X, y

# === Train and optimize SVM ===
def train_optimized_svm(X, y):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# === Main ===
if __name__ == "__main__":
    print("📦 Loading feature data...")
    X, y = load_data(FEATURE_CSV)

    print("🧪 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training SVM...")
    svm_model = train_optimized_svm(X_scaled, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ SVM model saved to: {MODEL_PATH}")
    print(f"✅ Scaler saved to: {SCALER_PATH}")
