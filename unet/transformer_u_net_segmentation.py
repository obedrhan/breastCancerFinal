import os
import random
import time
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ───────── CONFIGURATION ───────────────────────────────────────────
DDSM_ROOT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH    = os.path.join(DDSM_ROOT, "data/full_with_correct_roi.csv")
MODEL_OUT   = os.path.join(DDSM_ROOT, "unet_attention_final.pth")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use MPS on Apple Silicon if available
if DEVICE.type != 'cuda' and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🐎 Using MPS (GPU) backend")
elif DEVICE.type == 'cpu':
    print("⚠️  No GPU device found, using CPU (slow)")

BATCH_SIZE  = 4
IMG_SIZE    = (512, 512)
LR          = 1e-4
EPOCHS      = 50
TEST_SPLIT  = 0.2
SEED        = 42

# ───────── LOSS ────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        inter = (preds_flat * targets_flat).sum()
        return 1 - ((2*inter + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth))

# ───────── DATASET ─────────────────────────────────────────────────
class MammogramSegDataset(Dataset):
    def __init__(self, df, root, img_col, mask_col, size, augment=False):
        self.root      = root
        self.img_col   = img_col
        self.mask_col  = mask_col
        self.size      = size
        self.augment   = augment
        self.to_tensor = T.ToTensor()

        df_clean = df.copy()
        df_clean[img_col]  = df_clean[img_col].astype(str).str.strip()
        df_clean[mask_col] = df_clean[mask_col].astype(str).str.strip()
        self.records = df_clean.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records.iloc[idx]
        img_rel  = os.path.normpath(rec[self.img_col].strip())
        mask_rel = os.path.normpath(rec[self.mask_col].strip())
        img_path  = img_rel  if os.path.isabs(img_rel)  else os.path.join(self.root, img_rel)
        mask_path = mask_rel if os.path.isabs(mask_rel) else os.path.join(self.root, mask_rel)
        # ROI fallback
        if not os.path.exists(mask_path):
            alt_rel  = os.path.join("DDSM_IMAGES", "CBIS-DDSM", mask_rel)
            alt_path = os.path.join(self.root, alt_rel)
            if os.path.exists(alt_path):
                mask_path = alt_path
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return None
        try:
            if img_path.lower().endswith('.dcm'):
                img = pydicom.dcmread(img_path, force=True).pixel_array.astype(np.float32)
            else:
                img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            img = (img - img.min())/(img.max()-img.min()+1e-8)
            if mask_path.lower().endswith('.dcm'):
                m = pydicom.dcmread(mask_path, force=True).pixel_array.astype(np.float32)
                mask = (m>0).astype(np.float32)
            else:
                mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)/255.0
        except Exception:
            return None
        img_pil  = Image.fromarray((img*255).astype(np.uint8))
        mask_pil = Image.fromarray((mask*255).astype(np.uint8))
        if self.augment:
            if random.random() < 0.5:
                img_pil  = ImageOps.mirror(img_pil)
                mask_pil = ImageOps.mirror(mask_pil)
            angle = random.uniform(-15,15)
            img_pil  = img_pil.rotate(angle, resample=Image.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
        img_pil  = img_pil.resize(self.size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.size, Image.NEAREST)
        img_t    = self.to_tensor(img_pil)
        mask_t   = self.to_tensor(mask_pil)
        return img_t, mask_t

# ───────── COLLATE FUNCTION ─────────────────────────────────────────
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)

# ───────── ATTENTION U-NET ─────────────────────────────────────────
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g,F_int,1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l,F_int,1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu= nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=ConvBlock(1,64); self.enc2=ConvBlock(64,128)
        self.enc3=ConvBlock(128,256); self.enc4=ConvBlock(256,512)
        self.pool=nn.MaxPool2d(2); self.center=ConvBlock(512,1024)
        self.att4=AttentionBlock(512,512,256); self.att3=AttentionBlock(256,256,128); self.att2=AttentionBlock(128,128,64)
        self.up4=nn.ConvTranspose2d(1024,512,2,2); self.dec4=ConvBlock(1024,512)
        self.up3=nn.ConvTranspose2d(512,256,2,2); self.dec3=ConvBlock(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,2); self.dec2=ConvBlock(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,2); self.dec1=ConvBlock(128,64)
        self.outc=nn.Conv2d(64,1,1)
    def forward(self,x):
        c1,p1=self.enc1(x),self.pool(self.enc1(x))
        c2,p2=self.enc2(p1),self.pool(self.enc2(p1))
        c3,p3=self.enc3(p2),self.pool(self.enc3(p2))
        c4,p4=self.enc4(p3),self.pool(self.enc4(p3))
        c5=self.center(p4)
        u4=self.up4(c5); a4=self.att4(u4,c4); d4=self.dec4(torch.cat([u4,a4],1))
        u3=self.up3(d4); a3=self.att3(u3,c3); d3=self.dec3(torch.cat([u3,a3],1))
        u2=self.up2(d3); a2=self.att2(u2,c2); d2=self.dec2(torch.cat([u2,a2],1))
        u1=self.up1(d2); d1=self.dec1(torch.cat([u1,c1],1))
        return self.outc(d1)

# ───────── TRAINING PIPELINE ────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(SEED)
    df = pd.read_csv(CSV_PATH)
    test_df      = df[df['full_path'].str.contains('Test_', case=False)].reset_index(drop=True)
    train_val_df = df[df['full_path'].str.contains('Training_', case=False)].reset_index(drop=True)

    if 'pathology' in train_val_df.columns:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=TEST_SPLIT,
            random_state=SEED,
            stratify=train_val_df['pathology']
        )
    else:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=TEST_SPLIT,
            random_state=SEED
        )

    train_ds = MammogramSegDataset(train_df, DDSM_ROOT, 'full_path', 'correct_roi_path', IMG_SIZE, augment=True)
    val_ds   = MammogramSegDataset(val_df,   DDSM_ROOT, 'full_path', 'correct_roi_path', IMG_SIZE)
    test_ds  = MammogramSegDataset(test_df,  DDSM_ROOT, 'full_path', 'correct_roi_path', IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model     = AttentionUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    bce       = nn.BCEWithLogitsLoss()
    dice      = DiceLoss()

    best_dice = 0.0
    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        start = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            if imgs.numel() == 0: continue
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce(logits, masks) + dice(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        duration = time.time() - start
        print(f"Training epoch took {duration:.1f}s")

        model.eval()
        val_dice = 0.0
        count    = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                if imgs.numel() == 0: continue
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = torch.sigmoid(model(imgs))
                inter = (preds * masks).sum()
                d = (2*inter + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
                val_dice += d.item() * imgs.size(0)
                count    += imgs.size(0)
        val_dice = val_dice / count if count else 0
        scheduler.step(val_dice)
        print(f"Epoch {epoch}/{EPOCHS} Train Loss: {train_loss/len(train_ds):.4f}, Val Dice: {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  Saved best model (Dice={best_dice:.4f})")

    print(f"Training complete. Best Val Dice: {best_dice:.4f}")

    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    model.eval()
    test_dice = 0.0
    count     = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            if imgs.numel() == 0: continue
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = torch.sigmoid(model(imgs))
            inter = (preds * masks).sum()
            d = (2*inter + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
            test_dice += d.item() * imgs.size(0)
            count      += imgs.size(0)
    test_dice = test_dice / count if count else 0
    print(f"Final Test-set Dice: {test_dice:.4f}")