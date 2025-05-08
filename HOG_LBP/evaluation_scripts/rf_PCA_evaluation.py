import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# === Config ===
TEST_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test/combined_hog_lbp_features.csv"
SCALER_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/scaler_rf_hog_lbp_pca.pkl"
PCA_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/pca_rf_hog_lbp.pkl"
MODEL_PATH = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/HOG_LBP/random_forest_hog_lbp_pca.pkl"

# === Load and preprocess test data ===
def load_test_data(features_file):
    data = pd.read_csv(features_file)
    print("📑 Loaded test CSV with columns:", data.columns)

    # Normalize labels
    def normalize_label(label):
        if isinstance(label, str):
            label = label.lower()
            if "malignant" in label:
                return "malignant"
            elif "benign" in label or "callback" in label or "without" in label:
                return "benign"
            else:
                return "unknown"
        elif label in [0, 1]:
            return "benign" if label == 0 else "malignant"
        else:
            return "unknown"

    # Find the label column (last one assumed here)
    label_col = data.columns[-1]
    data["Label_Clean"] = data[label_col].apply(normalize_label)

    # Filter unknown labels
    data = data[data["Label_Clean"] != "unknown"]

    # Encode string labels
    le = LabelEncoder()
    y = le.fit_transform(data["Label_Clean"])

    # Drop non-numeric columns
    non_numeric_cols = [col for col in data.columns if data[col].dtype == 'object']
    X = data.drop(columns=non_numeric_cols + ["Label_Clean"]).astype(float).values

    return X, y

# === Evaluation Pipeline ===
if __name__ == "__main__":
    print("📥 Loading test features...")
    X_test, y_test = load_test_data(TEST_CSV)

    print("🔄 Loading scaler, PCA, and model...")
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    model = joblib.load(MODEL_PATH)

    print("⚙️ Scaling and transforming test data...")
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)

    print("🤖 Predicting...")
    y_pred = model.predict(X_pca)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")