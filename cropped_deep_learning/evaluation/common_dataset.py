# common_dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import pydicom
import numpy as np

BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/roi_cropped_with_pathology.csv")

class CroppedMammoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.copy()
        self.transform = transform
        self.df['pathology'] = self.df['pathology'].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
        self.df['label'] = self.df['pathology'].map({"BENIGN": 0, "MALIGNANT": 1})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]

        try:
            dicom = pydicom.dcmread(image_path)
            img = dicom.pixel_array.astype(np.float32)
            img -= img.min()
            img /= (img.max() + 1e-6)
            img *= 255.0
            img = Image.fromarray(img.astype(np.uint8)).convert("L")
        except:
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            img = self.transform(img)
        return img, label