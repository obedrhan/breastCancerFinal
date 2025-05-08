import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_test_data(features_file):
    data = pd.read_csv(features_file)
    print("📑 Loaded test CSV with columns:", data.columns)

    # Normalize labels
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

    data["Label_Clean"] = data.iloc[:, -1].apply(normalize_label)
    data = data[data["Label_Clean"] != "unknown"]

    # Encode to binary
    le = LabelEncoder()
    y = le.fit_transform(data["Label_Clean"])

    # Drop non-numeric columns
    non_numeric_cols = [col for col in data.columns if data[col].dtype == 'object']
    X = data.drop(columns=non_numeric_cols + ["Label_Clean"]).astype(float).values

    return X, y

if __name__ == "__main__":
    test_csv = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/combined_hog_lbp_features_cropped_test.csv"

    print("📥 Loading test features...")
    X_test, y_test = load_test_data(test_csv)

    print("🔄 Loading scaler and model...")
    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models"
    svm_model = joblib.load(os.path.join(model_dir, "svm_model_hog_lbp_cropped.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_svm_hog_lbp_cropped.pkl"))

    print("🧪 Scaling test data...")
    X_test_scaled = scaler.transform(X_test)

    print("📊 Predicting...")
    y_pred = svm_model.predict(X_test_scaled)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")