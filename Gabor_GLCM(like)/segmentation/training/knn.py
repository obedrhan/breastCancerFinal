import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# === Load and preprocess Gabor + GLCM features ===
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded dataset shape: {df.shape}")

    if "Label" not in df.columns:
        raise ValueError("CSV must contain a 'Label' column with 0 (benign) and 1 (malignant) values.")

    X = df.drop(columns=["Image Name", "Label"]).values
    y = df["Label"].values

    return X, y

# === Train KNN with grid search ===
def train_knn(X, y):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)

    print(f"✅ Best Parameters: {grid.best_params_}")
    print(f"✅ Best Cross-Validation Accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_

# === Main block ===
if __name__ == "__main__":
    BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
    CSV_FILE = os.path.join(BASE_DIR, "Gabor_GLCM(like)/data/glcm_gabor_features_segmented_training.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "Gabor_GLCM(like)/models")

    print("📥 Loading features...")
    X, y = load_data(CSV_FILE)

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training KNN classifier...")
    knn_model = train_knn(X_scaled, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(knn_model, os.path.join(MODEL_DIR, "knn_gabor_glcm_segmented.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_gabor_glcm_segmented.pkl"))
    print("💾 Model and scaler saved.")
