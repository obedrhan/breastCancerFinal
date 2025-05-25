import os
import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def dummy_hog_lbp_csv(tmp_path):
    hog_data = pd.DataFrame(np.random.rand(5, 10))
    hog_data.insert(0, "Image Name", [f"img_{i}" for i in range(5)])
    hog_path = tmp_path / "hog.csv"
    hog_data.to_csv(hog_path, index=False)

    lbp_data = pd.DataFrame(np.random.rand(5, 5))
    lbp_data.insert(0, "Image Name", [f"img_{i}" for i in range(5)])
    lbp_path = tmp_path / "lbp.csv"
    lbp_data.to_csv(lbp_path, index=False)

    return hog_path, lbp_path, tmp_path / "combined.csv"

def test_combine_features_success(dummy_hog_lbp_csv):
    hog_path, lbp_path, combined_path = dummy_hog_lbp_csv

    hog_df = pd.read_csv(hog_path)
    lbp_df = pd.read_csv(lbp_path)

    # Ensure matching lengths
    assert hog_df.shape[0] == lbp_df.shape[0]

    # Combine
    combined_df = pd.concat([hog_df, lbp_df.drop(columns=["Image Name"])], axis=1)
    combined_df.to_csv(combined_path, index=False)

    # Verify output
    assert os.path.exists(combined_path)
    df_out = pd.read_csv(combined_path)
    assert df_out.shape[0] == 5
    assert df_out.shape[1] == hog_df.shape[1] + lbp_df.shape[1] - 1  # Drop duplicate "Image Name"