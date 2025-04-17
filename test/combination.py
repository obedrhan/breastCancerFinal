import pandas as pd
import numpy as np

# Load the features
hog_df = pd.read_csv('/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test/hog_features_segmented.csv')
lbp_df = pd.read_csv('/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test/test_lbp_features.csv')

# Optional: check shapes and index match
assert hog_df.shape[0] == lbp_df.shape[0], "Row counts do not match!"
assert list(hog_df.index) == list(lbp_df.index), "Indices do not match!"

# Combine them horizontally
combined_df = pd.concat([hog_df, lbp_df], axis=1)

# Save the combined features
combined_df.to_csv('combined_hog_lbp_features.csv', index=False)

print("✅ Combined feature file saved as 'combined_hog_lbp_features.csv'")