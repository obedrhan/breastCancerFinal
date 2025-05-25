import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === Configuration ===
features_file = "/evaluation_test/test_lbp_features.csv"
model_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP/random_forest_lbp_smote.pkl"
scaler_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP/scaler_rf_lbp_smote.pkl"

# === Load and preprocess evaluation_test data ===
def load_data(path):
    df = pd.read_csv(path)

    def normalize_label(label):
        label = label.lower()
        if "malignant" in label:
            return "malignant"
        elif "benign" in label or "callback" in label:
            return "benign"
        else:
            return "unknown"

    df["Label"] = df["Label"].apply(normalize_label)
    df = df[df["Label"] != "unknown"]

    X = df.drop(columns=["Image Name", "Label"]).values
    y = df["Label"].values
    return X, y

# === Main Evaluation ===
if __name__ == "__main__":
    print("📥 Loading evaluation_test features...")
    X_test, y_test = load_data(features_file)

    print("🧪 Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("⚙️ Scaling evaluation_test features...")
    X_test_scaled = scaler.transform(X_test)

    print("🔐 Encoding labels...")
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)

    print("🔍 Predicting...")
    y_pred = model.predict(X_test_scaled)

    print("\n📊 Evaluation Results:")
    acc = accuracy_score(y_test_encoded, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%")

    print("\n📄 Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred))