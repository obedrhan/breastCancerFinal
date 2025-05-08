import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(features_file):
    data = pd.read_csv(features_file)
    print(f"📂 Loaded data shape: {data.shape}")

    # Debug print: see what labels look like before mapping
    print("\n🧪 Unique original label values:")
    print(data["Label"].unique())

    def normalize_label(label):
        if isinstance(label, str):
            label = label.strip().lower()
            if "malignant" in label:
                return 1
            elif "benign" in label:
                return 0
        elif isinstance(label, (int, float)):
            if label in [0, 1]:
                return int(label)
        return -1

    data["Label"] = data["Label"].apply(normalize_label)

    print("\n🔍 Label value counts (after normalization):")
    print(data["Label"].value_counts(dropna=False))

    data = data[data["Label"] != -1]

    print(f"\n✅ Final dataset size: X={data.shape[0]}, Features={data.shape[1] - 2}")

    X = data.drop(columns=["Image Name", "Label"]).values
    y = data["Label"].values
    return X, y


def train_optimized_knn(X, y):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/glcm_features_mahotas.csv"

    print("📦 Loading GLCM feature dataset...")
    X, y = load_data(features_file)

    print("🧪 Scaling dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training KNN on GLCM features...")
    knn_model = train_optimized_knn(X_scaled, y)

    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(knn_model, os.path.join(model_dir, "knn_cropped_glcm.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_knn_glcm_cropped.pkl"))

    print("✅ KNN model and scaler saved for GLCM features.")