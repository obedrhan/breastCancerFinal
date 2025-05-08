import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

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

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])  # 'benign' → 0, 'malignant' → 1

    X = data.drop(columns=["Image Name", "Label"]).values
    return X, y

if __name__ == "__main__":
    test_csv = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/glcm_features_mahotas_test.csv"
    model_dir = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models"

    print("📥 Loading GLCM test features...")
    X_test, y_test = load_test_data(test_csv)

    print("🔄 Loading GLCM scaler and model...")
    scaler = joblib.load(os.path.join(model_dir, "scaler_knn_glcm_cropped.pkl"))
    knn_model = joblib.load(os.path.join(model_dir, "knn_cropped_glcm.pkl"))

    print("📊 Predicting with GLCM features...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = knn_model.predict(X_test_scaled)

    print("\n✅ Classification Report (GLCM):")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")