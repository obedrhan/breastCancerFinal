import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

# ───────── Configuration ─────────────────────────
DDSM_ROOT    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODEL_PATH   = os.path.join(DDSM_ROOT, "transformer_unet.pth")
ROI_CSV      = os.path.join(DDSM_ROOT, "data/roi_cropped_with_pathology.csv")
FULL_CSV     = os.path.join(DDSM_ROOT, "data/full_mammogram_paths.csv")
OUTPUT_DIR   = os.path.join(DDSM_ROOT, "eval_crops")
IMG_SIZE     = (256, 256)
THRESHOLD    = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"▶️  Output dir: {OUTPUT_DIR}")

# ───────── Model Definition ─────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
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
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        return self.norm(x), H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
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
        self.conv1, self.pool1 = ConvBlock(in_c,64),  nn.MaxPool2d(2)
        self.conv2, self.pool2 = ConvBlock(64,128),   nn.MaxPool2d(2)
        self.conv3, self.pool3 = ConvBlock(128,256),  nn.MaxPool2d(2)
        self.conv4             = ConvBlock(256,dim)
        self.patch_embed       = PatchEmbedding(dim, dim)
        self.trans_blocks      = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.up3, self.dec3    = nn.ConvTranspose2d(dim,256,2,2), ConvBlock(dim+256,256)
        self.up2, self.dec2    = nn.ConvTranspose2d(256,128,2,2), ConvBlock(128+128,128)
        self.up1, self.dec1    = nn.ConvTranspose2d(128,64,2,2),  ConvBlock(64+64,64)
        self.classifier        = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        c1, p1 = self.conv1(x), self.pool1(self.conv1(x))
        c2, p2 = self.conv2(p1), self.pool2(self.conv2(p1))
        c3, p3 = self.conv3(p2), self.pool3(self.conv3(p2))
        c4     = self.conv4(p3)
        t,H,W  = self.patch_embed(c4)
        t      = self.trans_blocks(t)
        t      = t.transpose(1,2).view(-1, c4.size(1), H, W)
        u3 = self.up3(t);  d3 = self.dec3(torch.cat([u3,c3],1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2,c2],1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1,c1],1))
        return self.classifier(d1)

# ───────── Load model ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶️  Device: {device}")
model = TransformerUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✔️  Model loaded from {MODEL_PATH}")

# ───────── Read CSVs ─────────────────────────────
roi_df  = pd.read_csv(ROI_CSV)
full_df = pd.read_csv(FULL_CSV)
print(f"▶️  ROI entries: {len(roi_df):,}, full entries: {len(full_df):,}")

# build `pair_key` for masks: strip off the "-ROI mask images-.../1-2.dcm"
roi_df['pair_key'] = roi_df['full_path'] \
    .str.rsplit('-ROI mask images', n=1).str[0]

# restrict to TEST full‐image cases and build their `pair_key`
full_df = full_df[ full_df['full_path'].str.contains('Test_') ].copy()
full_df['pair_key'] = full_df['full_path'] \
    .str.rsplit('-full mammogram images', n=1).str[0]

print(f"▶️  Test‐set full images: {len(full_df):,}")

# ───────── Inference + Cropping + Accuracy ──────
total_correct = 0
total_pixels  = 0
processed     = 0

for idx, row in enumerate(full_df.itertuples(), start=1):
    rel_full = row.full_path
    key = rel_full.replace('/', '_')
    print(f"\n[{idx}/{len(full_df)}] {rel_full}")

    full_path = os.path.join(DDSM_ROOT, rel_full)
    try:
        dcm = pydicom.dcmread(full_path, force=True)
        full = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        print(f"   ❌ DICOM load failed: {e}")
        continue

    # normalize & get shape
    full = (full - full.min())/(full.max()+1e-8)
    Hf, Wf = full.shape

    # resize to network input
    small = np.array(
        Image.fromarray((full*255).astype(np.uint8))
             .resize(IMG_SIZE, Image.Resampling.BILINEAR),
        dtype=np.float32
    ) / 255.0

    inp = torch.from_numpy(small).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        prob   = torch.softmax(logits, dim=1)[0,1].cpu().numpy()

    print(f"   ✅ inference: max={prob.max():.3f}, mean={prob.mean():.3f}")

    # up-sample & binarize
    mask_up = cv2.resize((prob*255).astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(mask_up, int(255*THRESHOLD), 255, cv2.THRESH_BINARY)

    # find largest contour
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("   ⚠️  no lesion detected — skipping")
        continue
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    print(f"   🩺 crop box x={x},y={y},w={w},h={h}")

    # crop the full image & predicted mask
    crop_img  = full[y:y+h, x:x+w]
    crop_pred = mask_bin[y:y+h, x:x+w]

    # look up the GT ROI mask by matching pair_key
    pk = row.pair_key
    match = roi_df[roi_df['pair_key']==pk]
    if match.empty:
        print("   ⚠️  no ROI mask CSV entry — skipping")
        continue

    gt_rel = match.iloc[0].full_path
    gt_path = os.path.join(DDSM_ROOT, gt_rel)
    if not os.path.exists(gt_path):
        print("   ⚠️  GT file missing on disk — skipping")
        continue

    try:
        gt_dcm = pydicom.dcmread(gt_path, force=True)
        gt = (gt_dcm.pixel_array>0).astype(np.uint8)*255
    except Exception as e:
        print(f"   ❌ GT DICOM load failed: {e}")
        continue

    gt_crop = gt[y:y+h, x:x+w]

    # compute pixel‐wise accuracy
    correct = np.sum((crop_pred>0)==(gt_crop>0))
    total   = crop_pred.size
    total_correct += correct
    total_pixels  += total
    processed     += 1

    # save crops
    Image.fromarray((crop_img*255).astype(np.uint8))\
         .save(os.path.join(OUTPUT_DIR, f"{key}_crop.png"))
    Image.fromarray(crop_pred)\
         .save(os.path.join(OUTPUT_DIR, f"{key}_pred.png"))
    print("   💾 saved crop & prediction")

# ───────── Report ───────────────────────────────
mean_acc = (total_correct/total_pixels*100) if total_pixels else 0.0
print(f"\n✅ Processed {processed}/{len(full_df)} cases")
print(f"▶️  Mean pixel accuracy: {mean_acc:.2f}%")