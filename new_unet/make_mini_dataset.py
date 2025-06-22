import random
import pickle
import pandas as pd
from train3 import CSV_PATH, DDSM_ROOT, PATCH_SIZE, MammogramPatchDataset


def main(output_path='mini_indices.pkl', subset_size=500):
    # Read CSV and select 'mass' training samples
    df = pd.read_csv(CSV_PATH)
    df = df[df['full_path'].str.contains('mass', case=False)]
    df = df[df['full_path'].str.contains('training', case=False)].reset_index(drop=True)

    # Create full dataset (with augment=True to enable dynamic neg sampling)
    full_ds = MammogramPatchDataset(df, DDSM_ROOT, PATCH_SIZE, augment=True)
    total = len(full_ds)

    # Determine mini subset size
    mini_size = min(subset_size, total)
    print(f"Toplam örnek: {total}, Mini set boyutu: {mini_size}")

    # Randomly sample indices
    mini_indices = random.sample(range(total), mini_size)

    # Save to pickle
    with open(output_path, 'wb') as f:
        pickle.dump(mini_indices, f)
    print(f"{mini_size} örneklik mini indeks '{output_path}' dosyasına kaydedildi.")


if __name__ == '__main__':
    main()