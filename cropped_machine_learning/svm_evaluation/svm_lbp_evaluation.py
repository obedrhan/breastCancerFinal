import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

def load_test_data(test_csv_path):
    df = pd.read_csv(test_csv_path)

    # Normalize labels (include 'BENIGN_WITHOUT_CALLBACK')
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
    # Paths
    test_features_csv = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/lbp_features_cropped_test.csv"
    model_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models/svm_lbp_cropped.pkl"
    scaler_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models//scaler_svm_lbp_cropped.pkl"

    print("📥 Loading test features...")
    X_test, y_test = load_test_data(test_features_csv)

    print("🔄 Loading scaler and model...")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Preprocess test data
    X_test_scaled = scaler.transform(X_test)

    print("📊 Predicting...")
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")