import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

def load_data(features_file):
    """
    Load HOG feature CSV and return feature matrix X, labels y, and fitted LabelEncoder.
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


def train_optimized_svm(X, y):
    """
    Perform GridSearchCV to find best SVM hyperparameters.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(probability=True)
    # Use single process to avoid multiprocessing tracker errors
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X, y)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"📊 Best CV Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    # Path to your HOG feature CSV
    features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/hog_features_mass_training_cropped.csv"
    )

    print("📂 Loading and preprocessing dataset…")
    X, y, label_encoder = load_data(features_file)

    print("🔍 Scaling features…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training optimized SVM model…")
    svm_model = train_optimized_svm(X_scaled, y)

    # Directory to save trained models
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )
    os.makedirs(model_dir, exist_ok=True)

    print("💾 Saving model, scaler, and label encoder…")
    joblib.dump(svm_model, os.path.join(model_dir, "svm_hog.pkl"))
    joblib.dump(scaler,    os.path.join(model_dir, "scaler_svm_hog.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder_pathology_hog.pkl"))

    print("✅ Done! SVM (HOG) pipeline saved.")