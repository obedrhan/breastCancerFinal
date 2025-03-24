import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def load_data(features_file):
    """
    Load LBP features and normalize labels (benign vs malignant).
    """
    data = pd.read_csv(features_file)

    # Normalize labels
    def normalize_label(label):
        label = label.lower()
        if "malignant" in label:
            return "malignant"
        elif "benign" in label:
            return "benign"
        else:
            return "unknown"

    data["Label"] = data["Label"].apply(normalize_label)
    data = data[data["Label"] != "unknown"]

    X = data.drop(columns=["Image Name", "Label"]).values
    y = data["Label"].values
    return X, y

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = "/data/lbp_features_with_labels.csv"

    print("Loading dataset...")
    X, y = load_data(features_file)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize (optional for RF but good for consistency)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training Random Forest model with hyperparameter tuning...")
    rf_model = train_random_forest(X_train, y_train)

    print("Saving the trained model and scaler...")
    joblib.dump(rf_model, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/random_forest_model_ddsm.pkl")
    joblib.dump(scaler, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/scaler_rf.pkl")
    print("✅ Random Forest model and scaler saved.")