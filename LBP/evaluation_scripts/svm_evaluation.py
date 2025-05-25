import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_test_data(test_csv_path):
    df = pd.read_csv(test_csv_path)

    # Normalize labels
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

    X_test = df.drop(columns=["Image Name", "Label"]).values
    y_test = df["Label"].values
    return X_test, y_test

if __name__ == "__main__":
    test_features_csv = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/evaluation_test/test_lbp_features.csv"
    model_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP/svm_lbp_smote.pkl"
    scaler_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/LBP/scaler_svm_lbp_smote.pkl"

    print("📥 Loading evaluation_test features...")
    X_test, y_test = load_test_data(test_features_csv)

    print("🔄 Loading scaler and model...")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    print("⚙️ Scaling evaluation_test features...")
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    print("🔐 Encoding labels...")
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)

    print("📊 Predicting...")
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("\n✅ Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    print(f"✅ Accuracy: {accuracy_score(y_test_encoded, y_pred) * 100:.2f}%")

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred))