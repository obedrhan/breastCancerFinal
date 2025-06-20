import os
import random
import time
import numpy as np
import pandas as pd
import cv2
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ───────── CONFIG ────────────────────────────────────────────────
DDSM_ROOT       = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH        = os.path.join(DDSM_ROOT, "data/final_full_roi.csv")
MODEL_OUT       = os.path.join(DDSM_ROOT, "new_unet/patch_ds_attention_unet2.pth")

# Device setup
DEVICE          = torch.device("cuda") if torch.cuda.is_available() else \
                  torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
use_amp         = (DEVICE.type == "cuda")

# Hyperparameters
BATCH_SIZE      = 4
PATCH_SIZE      = 592
MODEL_SIZE      = 592
LR              = 1e-4
MAX_EPOCHS      = 30
PATIENCE        = 3
TRAIN_VAL_SPLIT = 0.2
SEED            = 42
GAMMA           = 0.3
AUX_WEIGHTS     = [0.6, 0.3, 0.1]
THRESHOLD_INIT  = 0.5
MIN_OBJ_SIZE    = 50

# ───────── HELPERS ───────────────────────────────────────────────
def crop_patch(img: np.ndarray, mask: np.ndarray, size: int):
    ys, xs = np.nonzero(mask)
    yc, xc = int(ys.mean()), int(xs.mean())
    half = size // 2
    H, W = img.shape
    y0, y1 = yc - half, yc + half
    x0, x1 = xc - half, xc + half
    pad_top    = max(0, -y0);    pad_left   = max(0, -x0)
    pad_bottom = max(0, y1 - H); pad_right  = max(0, x1 - W)
    y0_cl = max(0, y0); y1_cl = min(H, y1)
    x0_cl = max(0, x0); x1_cl = min(W, x1)
    patch_img  = img[y0_cl:y1_cl, x0_cl:x1_cl]
    patch_mask = mask[y0_cl:y1_cl, x0_cl:x1_cl]
    patch_img = cv2.copyMakeBorder(patch_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    patch_mask= cv2.copyMakeBorder(patch_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return patch_img, patch_mask

# ───────── MODEL & LOSS ─────────────────────────────────────────
class ExpLogDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, gamma=GAMMA):
        super().__init__(); self.smooth = smooth; self.gamma = gamma
    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        inter = (pred * targets).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2*inter + self.smooth) / (union + self.smooth)
        return (-torch.log(dice)).pow(self.gamma).mean()

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(True),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class DSAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1,64)
        self.enc2 = ConvBlock(64,128)
        self.enc3 = ConvBlock(128,256)
        self.enc4 = ConvBlock(256,512)
        self.pool = nn.MaxPool2d(2)
        self.center = ConvBlock(512,1024)
        self.up4 = nn.ConvTranspose2d(1024,512,2,2); self.att4 = SEBlock(512); self.dec4 = ConvBlock(1024,512)
        self.up3 = nn.ConvTranspose2d(512,256,2,2); self.att3 = SEBlock(256); self.dec3 = ConvBlock(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,2); self.att2 = SEBlock(128); self.dec2 = ConvBlock(256,128)
        self.up1 = nn.ConvTranspose2d(128,64,2,2);                     self.dec1 = ConvBlock(128,64)
        self.outc = nn.Conv2d(64,1,1)
        self.aux4 = nn.Conv2d(512,1,1)
        self.aux3 = nn.Conv2d(256,1,1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c  = self.center(self.pool(e4))
        u4,a4 = self.up4(c), self.att4(e4); d4 = self.dec4(torch.cat([u4,a4],1))
        u3,a3 = self.up3(d4), self.att3(e3); d3 = self.dec3(torch.cat([u3,a3],1))
        u2,a2 = self.up2(d3), self.att2(e2); d2 = self.dec2(torch.cat([u2,a2],1))
        u1    = self.up1(d2);             d1 = self.dec1(torch.cat([u1,e1],1))
        out = self.outc(d1)
        aux4 = nn.functional.interpolate(self.aux4(d4), scale_factor=8, mode='bilinear', align_corners=False)
        aux3 = nn.functional.interpolate(self.aux3(d3), scale_factor=4, mode='bilinear', align_corners=False)
        return out, aux4, aux3

# ───────── DATASET & AUGMENTATIONS ─────────────────────────────────
class MammogramPatchDataset(Dataset):
    def __init__(self, df, root, patch_size, augment=False):
        self.recs = []
        for _, r in df.iterrows():
            ip, mp = r['full_path'], r['roi_path']
            ip = ip if os.path.isabs(ip) else os.path.join(root, ip)
            mp = mp if os.path.isabs(mp) else os.path.join(root, mp)
            if os.path.exists(ip) and os.path.exists(mp):
                try:
                    _ = pydicom.dcmread(ip, force=True).pixel_array
                    _ = pydicom.dcmread(mp, force=True).pixel_array
                    self.recs.append((ip, mp))
                except:
                    continue
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        ip, mp = self.recs[idx]
        img = pydicom.dcmread(ip, force=True).pixel_array.astype(np.float32)
        img = (img - img.min())/(img.max()-img.min()+1e-8)
        m   = pydicom.dcmread(mp, force=True).pixel_array; mask = (m>0).astype(np.uint8)
        pimg, pmask = crop_patch(img, mask, self.patch_size)
        u8 = (pimg*255).astype(np.uint8)
        pimg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(u8)/255.0
        pil_i = Image.fromarray((pimg*255).astype(np.uint8))
        pil_m = Image.fromarray((pmask*255).astype(np.uint8))
        if self.augment:
            angle = random.uniform(-30, 30)
            translations = (random.uniform(-0.1,0.1)*self.patch_size, random.uniform(-0.1,0.1)*self.patch_size)
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            pil_i = TF.affine(pil_i, angle=angle, translate=translations, scale=scale, shear=[shear], interpolation=TF.InterpolationMode.BILINEAR)
            pil_m = TF.affine(pil_m, angle=angle, translate=translations, scale=scale, shear=[shear], interpolation=TF.InterpolationMode.NEAREST)
            pil_i = self.color_jitter(pil_i)
        pil_i = pil_i.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
        pil_m = pil_m.resize((MODEL_SIZE, MODEL_SIZE), Image.NEAREST)
        return self.to_tensor(pil_i), (np.array(pil_m)>127).astype(np.float32)

# ───────── TRAINING & EVALUATION ─────────────────────────────────
def train():
    df = pd.read_csv(CSV_PATH)
    df = df[df['full_path'].str.contains('mass', case=False)].reset_index(drop=True)
    trainval = df[df['full_path'].str.contains('training', case=False)]
    test_df  = df[df['full_path'].str.contains('test', case=False)]
    tr_df, vl_df = train_test_split(trainval, test_size=TRAIN_VAL_SPLIT, random_state=SEED)
    tr_ds = MammogramPatchDataset(tr_df, DDSM_ROOT, PATCH_SIZE, augment=True)
    vl_ds = MammogramPatchDataset(vl_df, DDSM_ROOT, PATCH_SIZE)
    ts_ds = MammogramPatchDataset(test_df, DDSM_ROOT, PATCH_SIZE)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    ts_ld = DataLoader(ts_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DSAttentionUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    loss_fn = ExpLogDiceLoss()
    scaler = GradScaler(enabled=use_amp)

    best_val, wait = 0.0, 0
    for epoch in range(1, MAX_EPOCHS+1):
        t0 = time.time()
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(tr_ld, desc="Training"):
            imgs = imgs.to(DEVICE); masks = masks.unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                out, aux4, aux3 = model(imgs)
                l_main = loss_fn(out, masks)
                l4 = loss_fn(aux4, masks)
                l3 = loss_fn(aux3, masks)
                loss = AUX_WEIGHTS[0]*l_main + AUX_WEIGHTS[1]*l4 + AUX_WEIGHTS[2]*l3
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(tr_ds)

        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in vl_ld:
                imgs = imgs.to(DEVICE); masks = masks.unsqueeze(1).to(DEVICE)
                out, _, _ = model(imgs)
                preds = (torch.sigmoid(out) > THRESHOLD_INIT).float()
                inter = (preds * masks).sum((1,2,3))
                union = preds.sum((1,2,3)) + masks.sum((1,2,3))
                val_dice += ((2*inter + 1e-6) / (union + 1e-6)).sum().item()
        val_dice /= len(vl_ds)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {time.time()-t0:.0f}s")

        if val_dice > best_val:
            best_val, wait = val_dice, 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Threshold sweep on validation set
    best_thr, best_score = THRESHOLD_INIT, best_val
    for thr in np.linspace(0.3, 0.7, 9):
        score = 0; count=0
        with torch.no_grad():
            for imgs, masks in vl_ld:
                imgs = imgs.to(DEVICE); masks = masks.unsqueeze(1).to(DEVICE)
                out, _, _ = model(imgs)
                preds = (torch.sigmoid(out) > thr).float()
                inter = (preds * masks).sum((1,2,3))
                union = preds.sum((1,2,3)) + masks.sum((1,2,3))
                score += ((2*inter + 1e-6) / (union + 1e-6)).sum().item()
                count += imgs.size(0)
        score /= count
        if score > best_score:
            best_score, best_thr = score, thr
    print(f"Optimal threshold: {best_thr:.2f} → Val Dice={best_score:.4f}")

    # Final test evaluation
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    model.eval()
    test_dice = 0
    with torch.no_grad():
        for imgs, masks in ts_ld:
            imgs = imgs.to(DEVICE); masks = masks.unsqueeze(1).to(DEVICE)
            out, _, _ = model(imgs)
            preds = (torch.sigmoid(out) > best_thr).float()
            inter = (preds * masks).sum((1,2,3))
            union = preds.sum((1,2,3)) + masks.sum((1,2,3))
            test_dice += ((2*inter + 1e-6) / (union + 1e-6)).sum().item()
    test_dice /= len(ts_ds)
    print(f"Test Dice: {test_dice:.4f}")

if __name__ == '__main__':
    train()
