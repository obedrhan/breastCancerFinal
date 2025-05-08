import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Load combined features ===
df = pd.read_csv("data/combined_features_training.csv")
X = df.drop(columns=["Image Name", "Label"]).values
y = df["Label"].values

# === Standardize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Apply PCA to retain 95% of variance ===
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === Save preprocessed features and models ===
joblib.dump(scaler, "scaler_test.pkl")
joblib.dump(pca, "pca_test.pkl")

pd.DataFrame(X_pca).to_csv("X_pca_test.csv", index=False)
pd.DataFrame(y, columns=["Label"]).to_csv("y_test.csv", index=False)

print("✅ Scaler, PCA, and preprocessed datasets saved successfully.")
print(f"🎯 Reduced feature dimensions from {X.shape[1]} to {X_pca.shape[1]}")