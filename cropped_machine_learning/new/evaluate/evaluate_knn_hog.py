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

    # Drop non-feature cols
    X_test = df.drop(columns=["Image Name", "Label", "Pathology"]).values
    return X_test, y_test, le


if __name__ == "__main__":
    # Update paths as needed
    test_features_file = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_data/hog_features_mass_test_cropped.csv"
    )
    model_dir = (
        "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/cropped_machine_learning/new/new_models"
    )

    print("📂 Loading test data…")
    X_test, y_test, label_encoder = load_test_data(test_features_file)

    print("🔄 Loading scaler and KNN model…")
    scaler    = joblib.load(os.path.join(model_dir, "scaler_knn_hog.pkl"))
    knn_model = joblib.load(os.path.join(model_dir, "knn_hog.pkl"))

    print("🔍 Scaling test features…")
    X_test_scaled = scaler.transform(X_test)

    print("⚡ Predicting with KNN…")
    y_pred  = knn_model.predict(X_test_scaled)
    y_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    auc    = roc_auc_score(y_test, y_proba)

    print(f"\n🎯 KNN Test Accuracy: {acc:.4f}")
    print(f"\n🗂 KNN Confusion Matrix:\n{cm}")
    print(f"\n📄 KNN Classification Report:\n{report}")
    print(f"📈 KNN ROC AUC: {auc:.4f}")