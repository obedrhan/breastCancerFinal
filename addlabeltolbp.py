import pandas as pd

lbp_df = pd.read_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_segmented.csv")

train_csv = pd.read_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/full_mammogram_paths.csv")

train_df_filtered = train_csv[train_csv['full_path'].str.contains('Training')].reset_index(drop=True)

train_df_filtered['Label'] = train_df_filtered['pathology']

lbp_df['Label'] = train_df_filtered['Label']

lbp_df.to_csv("/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/lbp_features_with_labels.csv", index=False)
print("✅ Labels added and saved to lbp_features_with_labels.csv")