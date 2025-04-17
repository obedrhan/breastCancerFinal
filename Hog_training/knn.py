import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(features_file):
    data = pd.read_csv(features_file)

    # Normalize labels
    def normalize_label(label):
        if isinstance(label, str):
            label = label.lower()
            if "malignant" in label:
                return "malignant"
            elif "benign" in label or "benign_without_callback" in label:
                return "benign"
            else:
                return "unknown"
        elif label in [0, 1]:
            return "benign" if label == 0 else "malignant"
        else:
            return "unknown"

    data["Label"] = data["Label"].apply(normalize_label)
    data = data[data["Label"] != "unknown"]

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])  # 'benign' → 0, 'malignant' → 1

    X = data.drop(columns=["Image Name", "Label"]).values
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/hog_features_segmented.csv"

    print("📦 Loading full HOG feature dataset...")
    X, y = load_data(features_file)

    # Scale all features
    print("🧪 Scaling full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training KNN on full dataset...")
    knn_model = train_optimized_knn(X_scaled, y)

    # Save model + scaler
    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/HOG"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(knn_model, os.path.join(model_dir, "knn_model_hog.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_knn_hog.pkl"))

    print("✅ KNN model and scaler saved for HOG features.")