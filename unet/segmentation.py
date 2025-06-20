import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pydicom
from PIL import Image

# ───────── CONFIGURATION ─────────────────────────────────────────
DDSM_ROOT    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
FULL_CSV     = os.path.join(DDSM_ROOT, "data/full_mammogram_paths.csv")
MODEL_PATH   = os.path.join(DDSM_ROOT, "transformer_unet.pth")
OUTPUT_DIR   = os.path.join(DDSM_ROOT, "eval_crops")
IMG_SIZE     = (256, 256)
THRESHOLD    = 0.2  # for final mask binarization

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"▶️ Output dir: {OUTPUT_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶️ Device: {device}")

# ───────── MODEL DEFINITION ────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_c, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, 1)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    def forward(self, x):
        h = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        h = x
        x = self.norm2(x)
        return x + self.mlp(x)

class TransformerUNet(nn.Module):
    def __init__(self, in_c=1, out_c=2, dim=256, heads=8, depth=4):
        super().__init__()
        self.conv1, self.pool1 = ConvBlock(in_c, 64), nn.MaxPool2d(2)
        self.conv2, self.pool2 = ConvBlock(64, 128), nn.MaxPool2d(2)
        self.conv3, self.pool3 = ConvBlock(128, 256), nn.MaxPool2d(2)
        self.conv4             = ConvBlock(256, dim)
        self.patch_embed       = PatchEmbedding(dim, dim)
        self.trans_blocks      = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.up3, self.dec3    = nn.ConvTranspose2d(dim, 256, 2, 2), ConvBlock(dim + 256, 256)
        self.up2, self.dec2    = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(128 + 128, 128)
        self.up1, self.dec1    = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(64 + 64, 64)
        self.classifier        = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        c1 = self.conv1(x); p1 = self.pool1(c1)
        c2 = self.conv2(p1); p2 = self.pool2(c2)
        c3 = self.conv3(p2); p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        t, H, W = self.patch_embed(c4)
        t = self.trans_blocks(t)
        t = t.transpose(1, 2).view(-1, c4.size(1), H, W)
        u3 = self.up3(t); d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, c1], dim=1))
        return self.classifier(d1)

# load the trained model
model = TransformerUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ───────── HELPER FUNCTIONS ────────────────────────────────────────
def load_mammogram(path):
    dcm = pydicom.dcmread(path, force=True)
    arr = dcm.pixel_array.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def preprocess_breast(img):
    u8 = (img * 255).astype(np.uint8)
    _, thr = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean  = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(clean, dtype=bool)
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(clean, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask.astype(bool)

# ───────── MAIN PIPELINE ──────────────────────────────────────────
df = pd.read_csv(FULL_CSV)
full_col = next(c for c in df.columns if 'full' in c.lower())

for idx, row in df.iterrows():
    rel_path = row[full_col]
    full_path = os.path.join(DDSM_ROOT, rel_path)
    # handle missing/corrupt DICOMs
    try:
        img = load_mammogram(full_path)
    except Exception as e:
        print(f"❌ Could not load image {rel_path}: {e}")
        continue

    # crop out background
    mask_bkg = preprocess_breast(img)
    ys, xs  = np.where(mask_bkg)
    if ys.size == 0:
        continue
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cropped_img = img[y0:y1, x0:x1]

    # inference
    small = cv2.resize(
        (cropped_img * 255).astype(np.uint8),
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0
    tensor = torch.from_numpy(small).unsqueeze(0).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logits = model(tensor)
        prob   = torch.softmax(logits, dim=1)[0,1].cpu().numpy()

    # binary mask
    mask_up = cv2.resize(
        (prob * 255).astype(np.uint8),
        (cropped_img.shape[1], cropped_img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    _, mask_bin = cv2.threshold(mask_up, int(255 * THRESHOLD), 255, cv2.THRESH_BINARY)

    # lesion bounding box
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No lesion found for {rel_path}")
        continue
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # prepare output directory
    rel_dir = os.path.splitext(rel_path)[0]
    out_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    # save crop, mask, segmented_image
    crop_uint8 = (cropped_img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, 'crop.png'), crop_uint8)
    cv2.imwrite(os.path.join(out_dir, 'mask.png'), mask_bin)
    seg = (cropped_img * (mask_bin>0)).astype(np.float32)
    seg_uint8 = (seg * 255).astype(np.uint8)
    seg_crop  = seg_uint8[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(out_dir, 'segmented_image.png'), seg_crop)

    print(f"Processed {idx+1}/{len(df)}: {rel_path}")

print("✅ All images processed.")