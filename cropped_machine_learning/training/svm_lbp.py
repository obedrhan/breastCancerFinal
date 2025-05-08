import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def load_data(features_file):
    data = pd.read_csv(features_file)

    # Normalize labels (handle 'BENIGN_WITHOUT_CALLBACK')
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

    # Extract features and labels
    X = data.drop(columns=["Image Name", "Label"]).values
    y = data["Label"].values
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
    features_file = "/cropped_machine_learning/data/lbp_features_cropped_training.csv"

    print("📥 Loading dataset...")
    X, y = load_data(features_file)

    print("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Training SVM with GridSearch...")
    svm_model = train_optimized_svm(X_scaled, y)

    print("💾 Saving model and scaler...")
    joblib.dump(svm_model, "/cropped_machine_learning/models/svm_lbp_cropped.pkl")
    joblib.dump(scaler, "/cropped_machine_learning/models/scaler_svm_lbp_cropped.pkl")
    print("✅ SVM model and scaler saved!")