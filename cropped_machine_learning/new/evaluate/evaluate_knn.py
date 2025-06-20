import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

def load_test_data(features_file):
    """
    Load and preprocess the test‐set CSV.
    Returns:
      X_test: feature array
      y_test: integer labels
      label_encoder: fitted LabelEncoder for Pathology
    """
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

    # Encode benign→0, malignant→1
    label_encoder = LabelEncoder().fit(df["Pathology"])
    y_test = label_encoder.transform(df["Pathology"])

    # Drop non‐feature columns
    X_test = df.drop(columns=["Image Name", "Label", "Pathology"]).values
    return X_test, y_test, label_encoder

def main():
    # === UPDATE THESE PATHS ===
    test_features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/lbp_features_mass_cropped_test.csv"
    )
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )
    # ===========================

    # Load test data
    print("📂 Loading test set…")
    X_test, y_test, label_encoder = load_test_data(test_features_file)

    # Load pipeline artifacts
    print("🔄 Loading scaler and model…")
    scaler       = joblib.load(os.path.join(model_dir, "scaler_knn_lbp_cropped.pkl"))
    knn_model    = joblib.load(os.path.join(model_dir, "knn_cropped_lbp.pkl"))

    # Scale features
    print("🔍 Scaling test features…")
    X_test_scaled = scaler.transform(X_test)

    # Predict
    print("⚡ Running predictions…")
    y_pred  = knn_model.predict(X_test_scaled)
    y_proba = knn_model.predict_proba(X_test_scaled)[:, 1]  # P(malignant)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = None

    # Print results
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print(f"\n🗂 Confusion Matrix:\n{cm}")
    print(f"\n📄 Classification Report:\n{report}")
    if auc is not None:
        print(f"📈 ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()