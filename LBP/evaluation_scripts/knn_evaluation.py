import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_test_data(features_file):
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

if __name__ == "__main__":
    # === Path Setup ===
    BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
    TEST_CSV = os.path.join(BASE_DIR, "test/test_lbp_features.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models/LBP/knn_lbp_smote.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "models/LBP/scaler_knn_lbp_smote.pkl")

    # === Load Test Data ===
    print("📂 Loading LBP test features...")
    X_test, y_test = load_test_data(TEST_CSV)

    # === Load Scaler and Model ===
    print("🔁 Loading saved model and scaler...")
    scaler = joblib.load(SCALER_PATH)
    knn_model = joblib.load(MODEL_PATH)

    # === Scale Test Data ===
    X_test_scaled = scaler.transform(X_test)

    # === Predict and Evaluate ===
    print("🧠 Evaluating KNN model...")
    y_pred = knn_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))