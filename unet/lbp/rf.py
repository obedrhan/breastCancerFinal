import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ───────── CONFIGURATION ───────────────────────────────────────────
FEATURES_CSV = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/unet/lbp_features_with_pathology_unet.csv"
OUT_DIR      = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/unet"
MODEL_FILE   = os.path.join(OUT_DIR, "rf_model.pkl")
SUMMARY_CSV  = os.path.join(OUT_DIR, "rf_summary.csv")
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
print(f"▶️ Output dir: {OUT_DIR}")

# ───────── LOAD FEATURES ───────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
# normalize pathology labels
df['pathology'] = df['pathology'].str.lower().replace({
    'benign_without_callback': 'benign',
    'benign': 'benign',
    'malignant': 'malignant'
})
# encode target
df['target'] = df['pathology'].map({'benign': 0, 'malignant': 1})
# drop unknowns
df = df[df['target'].notnull()].copy()

# ───────── SPLIT TRAIN/TEST ─────────────────────────────────────────
mask_train = df['full_path'].str.contains('training', case=False, na=False)
mask_test  = df['full_path'].str.contains('test',     case=False, na=False)
train_df = df[mask_train]
test_df  = df[mask_test]
print(f"▶️ Training samples: {len(train_df)}; Test samples: {len(test_df)}")

feature_cols     = [c for c in df.columns if c.startswith('lbp_')]
X_train_full = train_df[feature_cols].values
y_train_full = train_df['target'].values
X_test       = test_df[feature_cols].values
y_test       = test_df['target'].values

# ───────── VALIDATION SPLIT ────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)
print(f"▶️ Split {len(X_train)} train / {len(X_val)} validation")

# ───────── GRID SEARCH HYPERPARAMETERS ─────────────────────────────
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}
rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid = GridSearchCV(
    rf, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1
)
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
print(f"✔️ Saved RF model to {MODEL_FILE}")

results = {
    'best_params': [grid.best_params_],
    'validation_accuracy': val_acc,
    'test_accuracy': test_acc
}
res_df = pd.DataFrame(results)
res_df.to_csv(SUMMARY_CSV, index=False)
print(f"✔️ Summary written to {SUMMARY_CSV}")
