import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"

class MammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform
        self.dataframe["pathology"] = self.dataframe["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
        self.dataframe["label"] = self.dataframe["pathology"].map({"BENIGN": 0, "MALIGNANT": 1})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        relative_path = self.dataframe.iloc[idx]['full_path']
        img_path = os.path.join(BASE_DIR, relative_path)
        label = self.dataframe.iloc[idx]['label']

        try:
            dicom = pydicom.dcmread(img_path, force=True)
            if 'PixelData' not in dicom:
                raise ValueError("Missing PixelData")
            img_array = dicom.pixel_array.astype(np.float32)
            img_array -= img_array.min()
            img_array /= (img_array.max() + 1e-6)
            img_array *= 255.0
            image = Image.fromarray(img_array.astype(np.uint8)).convert("L")
        except Exception as e:
            # Skip faulty image by picking another one
            new_idx = (idx + 1) % len(self.dataframe)
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_dataloaders(df, batch_size=16):
    train_df = df[df['full_path'].str.contains("Training")].reset_index(drop=True)
    test_df = df[df['full_path'].str.contains("Test")].reset_index(drop=True)

    train_dataset = MammogramDataset(train_df, transform=transform)
    test_dataset = MammogramDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader