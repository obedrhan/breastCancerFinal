import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# === Configuration ===
features_file = ("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/data/lbp_features_cropped_test.csv")
model_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models/random_forest_lbp_cropped.pkl"
scaler_path = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/models/scaler_rf_lbp_cropped.pkl"

# === Load and preprocess test data ===
def load_data(path):
    df = pd.read_csv(path)

    def normalize_label(label):
        label = label.lower()
        if "malignant" in label:
            return "malignant"
        elif "benign" in label:
            return "benign"
        elif "callback" in label:
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
    print("📥 Loading test features...")
    X_test, y_test = load_data(features_file)

    print("🧪 Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("⚙️ Scaling test features...")
    X_test_scaled = scaler.transform(X_test)

    print("🔍 Predicting...")
    y_pred = model.predict(X_test_scaled)

    print("\n📊 Evaluation Results:")
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%\n")

    print("📝 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))