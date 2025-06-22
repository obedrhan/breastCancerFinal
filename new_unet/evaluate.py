import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

# ───────── CONFIG ───────────────────────────────────────────
DDSM_ROOT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH    = os.path.join(DDSM_ROOT, "data/final_full_roi.csv")
# List your trained checkpoint(s) here
CHECKPOINTS = [os.path.join(DDSM_ROOT, "new_unet/patch_ds_attention_unet5.pth")]
BEST_THR    = 0.65
BATCH_SIZE  = 4
PATCH_SIZE  = 592
MODEL_SIZE  = 592
DEVICE      = torch.device("cuda") if torch.cuda.is_available() else \
              torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ───────── IMPORT TRAINED MODEL CLASS ────────────────────────
from train3 import DSAttentionUNet

# ───────── IMPORT TRAINED MODEL CLASS ────────────────────────
from train2 import DSAttentionUNet

# ───────── DATASET ───────────────────────────────────────────
class TestPatchDataset(Dataset):
    def __init__(self, df):
        self.recs = []
        # Pre-validate and skip corrupted DICOMs
        for _, r in df.iterrows():
            path = r['full_path']
            if 'mass' in path.lower() and 'test' in path.lower():
                full = path if os.path.isabs(path) else os.path.join(DDSM_ROOT, path)
                roi  = r['roi_path']
                roi  = roi if os.path.isabs(roi) else os.path.join(DDSM_ROOT, roi)
                if os.path.exists(full) and os.path.exists(roi):
                    try:
                        # Attempt to load pixel data to catch corruption
                        _ = pydicom.dcmread(full, force=True).pixel_array
                        _ = pydicom.dcmread(roi, force=True).pixel_array
                        self.recs.append((full, roi))
                    except Exception:
                        # Skip corrupted or unreadable files
                        continue
        print(f"🔍 Loaded {len(self.recs)} valid test samples (skipped corrupted/invalid)")
        self.size = PATCH_SIZE
        self.recs = []
        for _, r in df.iterrows():
            path = r['full_path']
            if 'mass' in path.lower() and 'test' in path.lower():
                full = path if os.path.isabs(path) else os.path.join(DDSM_ROOT, path)
                roi  = r['roi_path']
                roi  = roi if os.path.isabs(roi) else os.path.join(DDSM_ROOT, roi)
                if os.path.exists(full) and os.path.exists(roi):
                    self.recs.append((full, roi))
        self.size = PATCH_SIZE

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        full, roi = self.recs[idx]
        # attempt to read DICOM, skip if corrupted
        try:
            img = pydicom.dcmread(full, force=True).pixel_array.astype(np.float32)
            mask = (pydicom.dcmread(roi, force=True).pixel_array > 0).astype(np.uint8)
        except Exception as e:
            # skip this corrupted sample by moving to the next one
            new_idx = (idx + 1) % len(self.recs)
            return self.__getitem__(new_idx)
        # normalize
        img = (img - img.min())/(img.max()-img.min()+1e-8)
        # compute center crop and padding
        ys, xs = np.nonzero(mask)
        yc, xc = int(ys.mean()), int(xs.mean())
        half = self.size // 2
        H, W = img.shape
        y0, y1 = max(0, yc-half), min(H, yc+half)
        x0, x1 = max(0, xc-half), min(W, xc+half)
        patch_img = img[y0:y1, x0:x1]
        patch_mask = mask[y0:y1, x0:x1]
        top, left = max(0, half-yc), max(0, half-xc)
        bottom = max(0, yc+half-H); right = max(0, xc+half-W)
        patch_img = cv2.copyMakeBorder(patch_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        patch_mask = cv2.copyMakeBorder(patch_mask, top, bottom, left, right, cv2.BORDER_CONSTANT)
        # CLAHE
        u8 = (patch_img*255).astype(np.uint8)
        patch_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(u8)/255.0
        # to PIL and resize
        pil_i = Image.fromarray((patch_img*255).astype(np.uint8))
        pil_m = Image.fromarray((patch_mask*255).astype(np.uint8))
        pil_i = pil_i.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
        pil_m = pil_m.resize((MODEL_SIZE, MODEL_SIZE), Image.NEAREST)
        return to_tensor(pil_i), (to_tensor(pil_m)>0.5).float()

# ───────── EVALUATION ────────────────────────────────────────
def evaluate_ensemble(models, loader, device, thr):
    total_d, count = 0.0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            bs = imgs.shape[0]
            prob_sum = torch.zeros_like(masks)
            for m in models:
                # main forward returns (out_main, aux4, aux3)
                out_main, *_ = m(imgs)
                p = torch.sigmoid(out_main)
                # original
                prob_sum += p
                # horizontal flip TTA
                p_flip = torch.sigmoid(m(torch.flip(imgs, dims=[3]))[0])
                prob_sum += torch.flip(p_flip, dims=[3])
            prob_avg = prob_sum / (len(models)*2)
            preds = (prob_avg > thr).float()
            inter = (preds * masks).sum((1,2,3))
            union= preds.sum((1,2,3)) + masks.sum((1,2,3))
            total_d += ((2*inter + 1e-6)/(union+1e-6)).sum().item()
            count   += bs
    return total_d / count

# ───────── MAIN ──────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    dataset = TestPatchDataset(df)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    models = []
    for ckpt in CHECKPOINTS:
        if os.path.exists(ckpt):
            model = DSAttentionUNet().to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            model.eval()
            models.append(model)
        else:
            print(f"⚠️ Checkpoint not found: {ckpt}")
    if not models:
        raise RuntimeError("No valid checkpoints loaded.")

    dice = evaluate_ensemble(models, loader, DEVICE, BEST_THR)
    print(f"Ensembled Test Dice (TTA): {dice:.4f}")