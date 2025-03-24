import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def load_data(features_file):
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

    # Drop ID columns
    X = data.drop(columns=["Image Name", "Label"]).values
    y = data["Label"].values
    return X, y

def train_optimized_svm(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }

    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    features_file = "/data/lbp_features_with_labels.csv"


    print("Loading dataset...")
    X, y = load_data(features_file)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training SVM with expanded GridSearch...")
    svm_model = train_optimized_svm(X_train, y_train)

    print("Saving the trained model and scaler...")
    joblib.dump(svm_model, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/svm_model_ddsm.pkl")
    joblib.dump(scaler, "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/models/scaler_svm.pkl")
    print("✅ SVM model and scaler saved!")