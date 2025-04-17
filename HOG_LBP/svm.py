import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

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

def train_optimized_svm(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }

    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/combined_hog_lbp_features.csv"

    print("📥 Loading full dataset...")
    X, y = load_data(features_file)

    print("🧪 Scaling full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🚀 Training SVM on full dataset...")
    svm_model = train_optimized_svm(X_scaled, y)

    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(svm_model, os.path.join(model_dir, "svm_model_hog_lbp.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_svm_hog_lbp.pkl"))

    print("✅ SVM model and scaler (HOG) saved!")