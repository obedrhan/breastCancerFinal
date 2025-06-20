import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV


def load_data(features_file):
    """
    Load combined LBP+HOG feature CSV and return feature matrix X, labels y, and LabelEncoder.
    """
    df = pd.read_csv(features_file)

    # Normalize pathology labels
    def normalize_pathology(p):
        p = str(p).strip().lower()
        if "malignant" in p:
            return "malignant"
        elif "benign" in p:
            return "benign"
        else:
            return "unknown"

    df["Pathology"] = df["Pathology"].apply(normalize_pathology)
    df = df[df["Pathology"] != "unknown"].reset_index(drop=True)

    # Encode labels (benign -> 0, malignant -> 1)
    le = LabelEncoder()
    y = le.fit_transform(df["Pathology"])

    # Drop non-feature columns
    X = df.drop(columns=["Image Name", "Label", "Pathology"]).values
    return X, y, le


def train_random_forest(X, y):
    """
    Train a Random Forest with GridSearchCV on combined features.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X, y)

    print(f"✅ RF Best Params: {grid_search.best_params_}")
    print(f"📊 RF Best CV Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    # Path to combined LBP+HOG feature CSV
    features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/lbp_hog_mass_features_training.csv"
    )

    print("📂 Loading and preprocessing dataset…")
    X, y, label_encoder = load_data(features_file)

    print("🔍 Scaling features…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training optimized Random Forest on combined features…")
    rf_model = train_random_forest(X_scaled, y)

    # Directory to save model
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )
    os.makedirs(model_dir, exist_ok=True)

    print("💾 Saving model, scaler, and label encoder…")
    joblib.dump(rf_model, os.path.join(model_dir, "rf_combined_lbp_hog.pkl"))
    joblib.dump(scaler,   os.path.join(model_dir, "scaler_rf_combined.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder_pathology.pkl"))

    print("✅ Done! Random Forest (LBP+HOG) pipeline saved.")