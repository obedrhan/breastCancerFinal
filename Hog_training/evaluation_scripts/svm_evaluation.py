import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_test_data(csv_path):
    data = pd.read_csv(csv_path)

    # Normalize label names
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

    le = LabelEncoder()
    y = le.fit_transform(data["Label"])
    X = data.drop(columns=["Image Name", "Label"]).values
    return X, y

if __name__ == "__main__":
    # Paths
    test_csv = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test/hog_features_segmented.csv"
    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/HOG"

    print("📥 Loading test HOG features...")
    X_test, y_test = load_test_data(test_csv)

    print("🔄 Loading scaler and SVM model...")
    scaler = joblib.load(os.path.join(model_dir, "scaler_svm_hog.pkl"))
    svm_model = joblib.load(os.path.join(model_dir, "svm_model_hog.pkl"))

    print("📊 Evaluating...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")