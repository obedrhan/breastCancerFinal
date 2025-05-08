import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Load test data ===
X_test = pd.read_csv("X_pca_test.csv").values
y_test = pd.read_csv("y_test.csv")["Label"].values

# === Normalize labels ===
def normalize_label(label):
    if isinstance(label, str):
        label = label.strip().upper()
        if "MALIGNANT" in label:
            return 1
        elif "BENIGN" in label or "BENIGN_WITHOUT_CALLBACK" in label:
            return 0
    elif label in [0, 1]:
        return label
    return -1  # Invalid

y_test = pd.Series(y_test).apply(normalize_label).values

# === Filter invalid entries ===
valid_idx = y_test != -1
X_test = X_test[valid_idx]
y_test = y_test[valid_idx]

# === Load trained model ===
model = joblib.load("models/xgboost_model_combined_pca.pkl")

# === Predict ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))