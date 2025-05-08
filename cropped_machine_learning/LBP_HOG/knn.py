import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(features_file):
    data = pd.read_csv(features_file)
    print("📑 Columns:", data.columns)

    # Normalize the label (from the last column)
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

    # Use the last label column for safety
    data["Label_Clean"] = data.iloc[:, -1].apply(normalize_label)
    data = data[data["Label_Clean"] != "unknown"]

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(data["Label_Clean"])  # 'benign' → 0, 'malignant' → 1

    # Drop all non-numeric columns
    non_numeric_cols = [col for col in data.columns if data[col].dtype == 'object']
    numeric_df = data.drop(columns=non_numeric_cols + ["Label_Clean"])

    X = numeric_df.astype(float).values
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/combined_hog_lbp_features_cropped.csv"

    print("📦 Loading combined HOG + LBP features...")
    X, y = load_data(features_file)

    # Scale full dataset
    print("🧪 Scaling full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training KNN on full dataset...")
    knn_model = train_optimized_knn(X_scaled, y)

    # Save model and scaler
    model_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models"
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(knn_model, os.path.join(model_path, "knn_lbp_hog_cropped.pkl"))
    joblib.dump(scaler, os.path.join(model_path, "scaler_knn_hog_lbp_cropped.pkl"))

    print("✅ KNN model and scaler saved for combined HOG + LBP features!")