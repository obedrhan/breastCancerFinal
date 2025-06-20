import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ───────── CONFIGURATION ───────────────────────────────────────────
FEATURES_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/unet/hog_lbp_features_with_pathology_unet.csv"
OUT_DIR      = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/unet"
SCALER_FILE  = os.path.join(OUT_DIR, "scaler_hog_lbp_knn.pkl")
MODEL_FILE   = os.path.join(OUT_DIR, "knn_hog_lbp_model.pkl")
SUMMARY_CSV  = os.path.join(OUT_DIR, "knn_hog_lbp_summary.csv")
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
print(f"▶️ Output dir: {OUT_DIR}")

# ───────── LOAD AND PREPARE FEATURES ─────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
# normalize pathology labels
df['pathology'] = df['pathology'].astype(str).str.lower().replace({
    'benign_without_callback': 'benign',
    'benign': 'benign',
    'malignant': 'malignant'
})
# encode target
df['target'] = df['pathology'].map({'benign': 0, 'malignant': 1})
# drop unknown labels
unknown_count = df['target'].isnull().sum()
if unknown_count > 0:
    print(f"⚠️ Dropping {unknown_count} rows with unknown pathology labels")
    df = df[df['target'].notnull()].copy()

# ───────── SPLIT TRAIN/TEST ─────────────────────────────────────────
mask_train = df['full_path'].str.contains('training', case=False, na=False)
mask_test  = df['full_path'].str.contains('test',     case=False, na=False)
train_df = df[mask_train].copy()
test_df  = df[mask_test].copy()
print(f"▶️ Training samples: {len(train_df)}; Test samples: {len(test_df)}")

# select feature columns
tfeature_cols = [c for c in df.columns if c.startswith('lbp_') or c.startswith('hog_')]
X_train_full = train_df[tfeature_cols].values
y_train_full = train_df['target'].values
X_test       = test_df[tfeature_cols].values
y_test       = test_df['target'].values

# ───────── VALIDATION SPLIT ────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)
print(f"▶️ Split {len(X_train)} train / {len(X_val)} validation")

# ───────── STANDARDIZE FEATURES ────────────────────────────────────
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
# save scaler
joblib.dump(scaler, SCALER_FILE)
print(f"✔️ Saved scaler to {SCALER_FILE}")

# ───────── GRID SEARCH HYPERPARAMETERS ─────────────────────────────
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=1)
print("▶️ Performing grid search on training subset...")
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"✅ Best params: {grid.best_params_}")

# ───────── EVALUATE ON VALIDATION ─────────────────────────────────
y_val_pred = best.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"▶️ Validation Accuracy: {val_acc:.4f}")
print(classification_report(y_val, y_val_pred, target_names=['benign','malignant']))

# ───────── FINAL EVALUATION ON TEST ────────────────────────────────
y_test_pred = best.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"▶️ Test Accuracy: {test_acc:.4f}")
print(classification_report(y_test, y_test_pred, target_names=['benign','malignant']))

# ───────── SAVE MODEL & SUMMARY ───────────────────────────────────
joblib.dump(best, MODEL_FILE)
print(f"✔️ Saved KNN model to {MODEL_FILE}")

results = {
    'best_params': [grid.best_params_],
    'validation_accuracy': val_acc,
    'test_accuracy': test_acc
}
res_df = pd.DataFrame(results)
res_df.to_csv(SUMMARY_CSV, index=False)
print(f"✔️ Summary written to {SUMMARY_CSV}")