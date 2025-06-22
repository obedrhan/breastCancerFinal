import os
import random
import time
import pickle
import numpy as np
import pandas as pd
import cv2
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
from contextlib import nullcontext

# ───────── CONFIG ────────────────────────────────────────────────
DDSM_ROOT       = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH        = os.path.join(DDSM_ROOT, "data/final_full_roi.csv")
MODEL_OUT       = os.path.join(DDSM_ROOT, "new_unet/patch_ds_attention_unet6s.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = DEVICE.type == "cuda"

# Hyperparameters (no argparse)
USE_MINI        = False             # set True to use mini subset
MINI_INDICES    = 'mini_indices.pkl'
PATCH_SIZE      = 592
MODEL_SIZE      = 592
BATCH_SIZE      = 4
MAX_EPOCHS      = 30
LR              = 1e-4
PATIENCE        = 5
TRAIN_VAL_SPLIT = 0.2
SEED            = 42
GAMMA           = 0.3
AUX_WEIGHTS     = [0.6, 0.3, 0.1]
THRESH_INIT     = 0.5

# ───────── HELPERS ───────────────────────────────────────────────
def crop_patch(img: np.ndarray, mask: np.ndarray, size: int):
    ys, xs = np.nonzero(mask)
    yc, xc = int(ys.mean()), int(xs.mean())
    half = size // 2
    H, W = img.shape
    y0, y1 = max(0, yc-half), min(H, yc+half)
    x0, x1 = max(0, xc-half), min(W, xc+half)
    patch = img[y0:y1, x0:x1]
    mpatch = mask[y0:y1, x0:x1]
    top = half - yc if yc<half else 0
    left = half - xc if xc<half else 0
    bottom = (yc+half - H) if yc+half>H else 0
    right = (xc+half - W) if xc+half>W else 0
    patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mpatch = cv2.copyMakeBorder(mpatch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return patch, mpatch

class ExpLogDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, gamma=GAMMA):
        super().__init__(); self.smooth, self.gamma = smooth, gamma
    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        inter = (pred * targets).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2*inter + self.smooth)/(union + self.smooth)
        return (-torch.log(dice)).pow(self.gamma).mean()

class MammogramPatchDataset(Dataset):
    def __init__(self, df, root, patch_size, augment=False):
        self.recs = []
        self.root = root
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.color_jitter = T.ColorJitter(0.2,0.2)

        # Pre‐validate every pair and only keep the ones that load successfully
        for _, r in df.iterrows():
            full = r['full_path']
            roi  = r['roi_path']
            full_p = full if os.path.isabs(full) else os.path.join(root, full)
            roi_p  = roi  if os.path.isabs(roi)  else os.path.join(root, roi)
            if not (os.path.exists(full_p) and os.path.exists(roi_p)):
                continue
            try:
                _ = pydicom.dcmread(full_p, force=True).pixel_array
                _ = pydicom.dcmread(roi_p,  force=True).pixel_array
            except Exception:
                # skip corrupted/unreadable files
                continue
            self.recs.append((full_p, roi_p))

        if len(self.recs) == 0:
            raise RuntimeError("No valid (full, roi) pairs found in MammogramPatchDataset!")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        ip, mp = self.recs[idx]
        # ... rest of your data‐loading, augmentation, etc. unchanged ...
        ip = ip if os.path.isabs(ip) else os.path.join(self.root, ip)
        mp = mp if os.path.isabs(mp) else os.path.join(self.root, mp)
        img = pydicom.dcmread(ip).pixel_array.astype(np.float32)
        img = (img - img.min())/(img.max()-img.min()+1e-8)
        mask = (pydicom.dcmread(mp).pixel_array>0).astype(np.uint8)
        if self.augment and random.random()<0.5:
            H,W = img.shape
            while True:
                y = random.randint(0,H-self.patch_size)
                x = random.randint(0,W-self.patch_size)
                m = mask[y:y+self.patch_size, x:x+self.patch_size]
                if m.sum()==0:
                    p_img, p_mask = img[y:y+self.patch_size, x:x+self.patch_size], m
                    break
        else:
            p_img, p_mask = crop_patch(img, mask, self.patch_size)
        u8 = (p_img*255).astype(np.uint8)
        p_img = cv2.createCLAHE(2.0,(8,8)).apply(u8)/255.0
        pil_i = Image.fromarray((p_img*255).astype(np.uint8))
        pil_m = Image.fromarray((p_mask*255).astype(np.uint8))
        if self.augment:
            angle = random.uniform(-30,30)
            trans = (random.uniform(-0.1,0.1)*self.patch_size,
                     random.uniform(-0.1,0.1)*self.patch_size)
            scale = random.uniform(0.9,1.1)
            shear = random.uniform(-10,10)
            pil_i = TF.affine(pil_i, angle, trans, scale, [shear],
                              interpolation=InterpolationMode.BILINEAR)
            pil_m = TF.affine(pil_m, angle, trans, scale, [shear],
                              interpolation=InterpolationMode.NEAREST)
            pil_i = self.color_jitter(pil_i)
        pil_i = pil_i.resize((MODEL_SIZE,MODEL_SIZE), Image.BILINEAR)
        pil_m = pil_m.resize((MODEL_SIZE,MODEL_SIZE), Image.NEAREST)
        return self.to_tensor(pil_i), (np.array(pil_m)>127).astype(np.float32)

# ───────── TRAIN & EVAL ─────────────────────────────────────────
def train():
    df = pd.read_csv(CSV_PATH)
    df = df[df['full_path'].str.contains('mass',False)].reset_index(drop=True)
    trainval = df[df['full_path'].str.contains('training',False)]
    test_df  = df[df['full_path'].str.contains('test',False)]
    tr_df, vl_df = train_test_split(trainval, test_size=TRAIN_VAL_SPLIT, random_state=SEED)

    if USE_MINI:
        full_tr = MammogramPatchDataset(tr_df, DDSM_ROOT, PATCH_SIZE, augment=True)
        with open(MINI_INDICES, 'rb') as f:
            mini_idx = pickle.load(f)
        tr_ds = Subset(full_tr, mini_idx)
    else:
        tr_ds = MammogramPatchDataset(tr_df, DDSM_ROOT, PATCH_SIZE, augment=True)
    vl_ds = MammogramPatchDataset(vl_df, DDSM_ROOT, PATCH_SIZE)
    ts_ds = MammogramPatchDataset(test_df, DDSM_ROOT, PATCH_SIZE)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    ts_ld = DataLoader(ts_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1).to(DEVICE)
    for p in model.encoder.parameters(): p.requires_grad=False
    for n,p in model.encoder.named_parameters():
        if n.startswith('layer3') or n.startswith('layer4'): p.requires_grad=True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    sched = OneCycleLR(opt, max_lr=5e-4, steps_per_epoch=len(tr_ld), epochs=MAX_EPOCHS, pct_start=0.1)
    scaler = GradScaler(enabled=use_amp)
    loss_fn = ExpLogDiceLoss()

    best, wait = 0.0, 0
    for e in range(1, MAX_EPOCHS+1):
        # choose appropriate autocast context for GPU; no-op on MPS/CPU
        amp_ctx = autocast(device_type='cuda', enabled=use_amp and DEVICE.type=='cuda') if use_amp else nullcontext()
        model.train(); total_loss = 0.0
        for xb, yb in tqdm(tr_ld, desc=f'Epoch {e}/{MAX_EPOCHS}'):
            xb, yb = xb.to(DEVICE), yb.unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            # automatically use mixed-precision when on CUDA
            with amp_ctx:
                logits = model(xb)
            loss = loss_fn(logits, yb)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            total_loss += loss.item() * xb.size(0)
        # validation
        model.eval(); dice = 0.0
        with torch.no_grad():
            for xb, yb in vl_ld:
                xb, yb = xb.to(DEVICE), yb.unsqueeze(1).to(DEVICE)
                preds = (torch.sigmoid(model(xb)) > THRESH_INIT).float()
                inter = (preds * yb).sum((1,2,3)); uni = preds.sum((1,2,3)) + yb.sum((1,2,3))
                dice += ((2*inter+1e-6)/(uni+1e-6)).sum().item()
        dice /= len(vl_ds)
        print(f'Epoch {e} | Loss: {total_loss/len(tr_ds):.4f} | Val Dice: {dice:.4f}')
        if dice > best:
            best, wait = dice, 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'Early stopping at epoch {e}')
                break

if __name__ == '__main__':
    train()
