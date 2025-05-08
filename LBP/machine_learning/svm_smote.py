import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

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

    # Encode labels (benign → 0, malignant → 1)
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
    features_file = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_with_labels.csv"

    print("📥 Loading dataset...")
    X, y = load_data(features_file)

    print("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔄 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("🤖 Training SVM with GridSearch...")
    svm_model = train_optimized_svm(X_resampled, y_resampled)

    print("💾 Saving model and scaler...")
    os.makedirs("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP", exist_ok=True)
    joblib.dump(svm_model, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models//LBP/svm_lbp_smote.pkl")
    joblib.dump(scaler, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP/scaler_svm_lbp_smote.pkl")
    print("✅ SVM model and scaler saved!")