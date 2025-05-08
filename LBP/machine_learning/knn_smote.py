import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

def load_data(features_file):
    data = pd.read_csv(features_file)

    # Normalize labels
    def normalize_label(label):
        if isinstance(label, str):
            label = label.strip().lower()
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
    y = le.fit_transform(data["Label"])

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
    print(f"📊 Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM//data/lbp_features_with_labels.csv"

    print("📂 Loading and preprocessing dataset...")
    X, y = load_data(features_file)

    print("🔍 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔄 Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("🤖 Training optimized KNN model with GridSearch...")
    knn_model = train_optimized_knn(X_resampled, y_resampled)

    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/Segmented_deep_learning"
    os.makedirs(model_dir, exist_ok=True)

    print("💾 Saving model and scaler...")
    joblib.dump(knn_model, os.path.join(model_dir, "knn_lbp_smote.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_knn_lbp_smote.pkl"))
    print("✅ Done! Optimized KNN model saved.")