import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

def load_data(features_file):
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

    # Encode benign→0, malignant→1
    le = LabelEncoder()
    y = le.fit_transform(df["Pathology"])

    # Drop non-feature cols
    X = df.drop(columns=["Image Name", "Label", "Pathology"]).values
    return X, y, le

def train_optimized_rf(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X, y)

    print(f"✅ RF Best Parameters: {grid_search.best_params_}")
    print(f"📊 RF CV Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/lbp_features_mass_training_cropped.csv"
    )
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )
    os.makedirs(model_dir, exist_ok=True)

    print("📂 Loading and preprocessing dataset…")
    X, y, label_encoder = load_data(features_file)

    print("🔍 Scaling features…")
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("🤖 Training optimized Random Forest model…")
    rf_model = train_optimized_rf(X_scaled, y)

    print("💾 Saving model, scaler, and label encoder…")
    joblib.dump(rf_model, os.path.join(model_dir, "rf_cropped_lbp.pkl"))
    joblib.dump(scaler,  os.path.join(model_dir, "scaler_rf_lbp_cropped.pkl"))
    joblib.dump(label_encoder,
                 os.path.join(model_dir, "label_encoder_pathology.pkl"))

    print("✅ Done! Random Forest pipeline saved.")