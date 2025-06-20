import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def load_test_data(features_file):
    df = pd.read_csv(features_file)

    # Normalize pathology labels
    def normalize_pathology(p):
        p = str(p).strip().lower()
        if "malignant" in p:
            return "malignant"
        elif "benign" in p:
            return "benign"
        else:
            return "unknown"

    df["Pathology"] = df["Pathology"].apply(normalize_pathology)
    df = df[df["Pathology"] != "unknown"].reset_index(drop=True)

    # Encode labels
    le = LabelEncoder().fit(df["Pathology"])
    y_test = le.transform(df["Pathology"])

    # Drop non-feature columns
    X_test = df.drop(columns=["Image Name", "Label", "Pathology"]).values
    return X_test, y_test, le

if __name__ == "__main__":
    # === Update paths as needed ===
    test_features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/lbp_hog_mass_features_test.csv"
    )
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )
    # =============================

    print("📂 Loading test data…")
    X_test, y_test, label_encoder = load_test_data(test_features_file)

    print("🔄 Loading scaler and RF combined model…")
    scaler   = joblib.load(os.path.join(model_dir, "scaler_rf_combined.pkl"))
    rf_model = joblib.load(os.path.join(model_dir, "rf_combined_lbp_hog.pkl"))

    print("🔍 Scaling test features…")
    X_test_scaled = scaler.transform(X_test)

    print("⚡ Predicting with Random Forest (combined features)…")
    y_pred  = rf_model.predict(X_test_scaled)
    y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    auc    = roc_auc_score(y_test, y_proba)

    print(f"\n🎯 RF Combined Test Accuracy: {acc:.4f}")
    print(f"\n🗂 RF Combined Confusion Matrix:\n{cm}")
    print(f"\n📄 RF Combined Classification Report:\n{report}")
    print(f"📈 RF Combined ROC AUC: {auc:.4f}")