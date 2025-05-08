import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def load_data(features_file):
    data = pd.read_csv(features_file)
    print("📑 Columns:", data.columns)

    # Normalize label (last column) including 'BENIGN_WITHOUT_CALLBACK'
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

    data["Label_Clean"] = data.iloc[:, -1].apply(normalize_label)
    data = data[data["Label_Clean"] != "unknown"]

    # Encode labels to 0/1
    le = LabelEncoder()
    y = le.fit_transform(data["Label_Clean"])

    # Drop all non-numeric columns
    non_numeric_cols = [col for col in data.columns if data[col].dtype == 'object']
    numeric_df = data.drop(columns=non_numeric_cols + ["Label_Clean"])

    X = numeric_df.astype(float).values
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/combined_hog_lbp_features.csv"

    print("📥 Loading full dataset...")
    X, y = load_data(features_file)

    print("🧪 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔍 Applying PCA (retain 95% variance)...")
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    print("🌲 Training Random Forest...")
    rf_model = train_random_forest(X_pca, y)

    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP"
    os.makedirs(model_dir, exist_ok=True)

    print("💾 Saving model, scaler, and PCA...")
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest_hog_lbp_pca.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_rf_hog_lbp_pca.pkl"))
    joblib.dump(pca, os.path.join(model_dir, "pca_rf_hog_lbp.pkl"))

    print("✅ Random Forest model, scaler, and PCA saved successfully.")