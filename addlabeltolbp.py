import pandas as pd

# Load your LBP feature CSV (generated previously)
lbp_df = pd.read_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_segmented.csv")

# Load the original CSV containing paths + labels
train_csv = pd.read_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv")

# Step 1: Filter only the 'Training' samples (since LBP script only processed training)
train_df_filtered = train_csv[train_csv['full_path'].str.contains('Training')].reset_index(drop=True)

# Step 2: Extract labels from train_df_filtered
# Assuming the label column is like 'mass shape' or 'pathology' or similar
train_df_filtered['Label'] = train_df_filtered['pathology']  # Adjust this column name if needed

# Step 3: Merge by row order
lbp_df['Label'] = train_df_filtered['Label']

# Step 4: Save to new CSV
lbp_df.to_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_with_labels.csv", index=False)
print("✅ Labels added and saved to lbp_features_with_labels.csv")