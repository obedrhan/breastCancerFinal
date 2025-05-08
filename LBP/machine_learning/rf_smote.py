import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(features_file):
    """
    Load features and normalize labels (benign vs malignant).
    """
    data = pd.read_csv(features_file)

    def normalize_label(label):
        label = label.lower()
        if "malignant" in label:
            return "malignant"
        elif "benign" in label or "callback" in label:
            return "benign"
        else:
            return "unknown"

    data["Label"] = data["Label"].apply(normalize_label)
    data = data[data["Label"] != "unknown"]

    # Label encoding (benign → 0, malignant → 1)
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])

    X = data.drop(columns=["Image Name", "Label"]).values
    return X, y

def train_random_forest(X, y):
    """
    Train Random Forest with hyperparameter tuning.
    """
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_with_labels.csv"
    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP"

    print("📦 Loading full dataset...")
    X, y = load_data(features_file)

    print("📊 Scaling all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔄 Applying SMOTE to balance dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("🌲 Training Random Forest on resampled data...")
    rf_model = train_random_forest(X_resampled, y_resampled)

    print("💾 Saving model and scaler...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest_lbp_smote.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_rf_lbp_smote.pkl"))
    print("✅ Random Forest model and scaler saved.")