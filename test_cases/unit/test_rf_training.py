import pytest
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HOG_LBP.randomforest import load_data, train_random_forest

@pytest.fixture
def dummy_feature_csv(tmp_path):
    csv_path = tmp_path / "dummy_features.csv"
    df = pd.DataFrame({
        "Feature_1": np.random.rand(10),
        "Feature_2": np.random.rand(10),
        "Image Name": [f"img_{i}" for i in range(10)],
        "Label": ["benign", "malignant"] * 5
    })
    df.to_csv(csv_path, index=False)
    return csv_path

def test_load_data_and_train(tmp_path, dummy_feature_csv):
    # Load and check shape
    X, y = load_data(str(dummy_feature_csv))
    assert X.shape[0] == 10
    assert X.shape[1] == 2  # 2 features
    assert len(y) == 10
    assert set(y) <= {0, 1}

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = train_random_forest(X_scaled, y)
    assert isinstance(model, RandomForestClassifier)

    # Save to temp directory
    joblib.dump(model, tmp_path / "rf_model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")

    assert os.path.exists(tmp_path / "rf_model.pkl")
    assert os.path.exists(tmp_path / "scaler.pkl")