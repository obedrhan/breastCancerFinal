import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(features_file):
    data = pd.read_csv(features_file)

    # Normalize labels (string → 'benign' / 'malignant')
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

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])  # 'benign' → 0, 'malignant' → 1

    X = data.drop(columns=["Image Name", "Label"]).values
    return X, y

def train_random_forest(X, y):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = "/cropped_machine_learning/data/hog_features_cropped_training.csv"

    print("📥 Loading HOG dataset...")
    X, y = load_data(features_file)

    print("🧪 Scaling full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🌲 Training Random Forest on full dataset...")
    rf_model = train_random_forest(X_scaled, y)

    model_dir = "/cropped_machine_learning/models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(model_dir, "random_forest_cropped_hog.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_rf_hog_cropped.pkl"))

    print("✅ Random Forest model and scaler (HOG) saved.")