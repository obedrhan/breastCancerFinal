import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import os

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

    # Encode labels to 0/1
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])

    X = data.drop(columns=["Image Name", "Label"]).values
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/glcm_features_mahotas.csv"

    print("📥 Loading GLCM dataset...")
    X, y = load_data(features_file)

    print("🔄 Scaling full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🚀 Training SVM on GLCM features...")
    svm_model = train_optimized_svm(X_scaled, y)

    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(svm_model, os.path.join(model_dir, "svm_cropped_glcm.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_svm_glcm_cropped.pkl"))
    print("✅ SVM model and scaler (GLCM) saved.")