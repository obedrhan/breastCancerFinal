import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from PIL import Image
import pandas as pd

# User configuration
CSV_PATH       = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/data/roi_cropped_with_pathology.csv"
DDSM_ROOT      = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
OUTPUT_PATH    = os.path.join(DDSM_ROOT, "transformer_unet.pth")
EPOCHS         = 30
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-4
IMG_SIZE       = (256, 256)

# 1. Read CSV and build image–mask pairs
df = pd.read_csv(CSV_PATH)
df['pair_key'] = df['full_path'].apply(lambda p: p.rsplit('-', 1)[0])

df_img = df[df['label']=="cropped"][['pair_key','image_path','relative_path','pathology']] \
           .rename(columns={'image_path':'img_path'})
df_mask = df[df['label']=="roi"][['pair_key','image_path']] \
            .rename(columns={'image_path':'mask_path'})
df_pairs = pd.merge(df_img, df_mask, on='pair_key', how='inner')

# 2. Model definitions
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
        self.conv1, self.pool1 = ConvBlock(in_c,64), nn.MaxPool2d(2)
        self.conv2, self.pool2 = ConvBlock(64,128), nn.MaxPool2d(2)
        self.conv3, self.pool3 = ConvBlock(128,256), nn.MaxPool2d(2)
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

# 3. Dataset
class MammoPatchDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img_p = row['img_path'].strip()
        mask_p= row['mask_path'].strip()
        img_p = os.path.normpath(img_p)
        mask_p= os.path.normpath(mask_p)
        if not os.path.isabs(img_p):
            img_p = os.path.join(DDSM_ROOT, img_p)
        if not os.path.isabs(mask_p):
            mask_p = os.path.join(DDSM_ROOT, mask_p)
        if idx < 5:
            print("Checking:", img_p, os.path.exists(img_p), mask_p, os.path.exists(mask_p))
        try:
            if img_p.lower().endswith('.dcm'):
                img = pydicom.dcmread(img_p).pixel_array.astype(np.float32)
            else:
                img = np.array(Image.open(img_p).convert('L'), dtype=np.float32)
            img = (img - img.min())/(img.max()+1e-8)
            img = np.array(Image.fromarray((img*255).astype(np.uint8))
                           .resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)/255.0

            if mask_p.lower().endswith('.dcm'):
                m = pydicom.dcmread(mask_p).pixel_array.astype(np.float32)
                mask = (m>0).astype(np.float32)
            else:
                mask = np.array(Image.open(mask_p).convert('L'), dtype=np.float32)/255.0
            mask = np.array(Image.fromarray((mask*255).astype(np.uint8))
                            .resize(IMG_SIZE, Image.NEAREST), dtype=np.float32)/255.0

            return torch.from_numpy(img)[None], torch.from_numpy(mask)[None]
        except:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)

# 4. Split train/val/test
train_val = df_pairs[df_pairs.relative_path.str.contains('Training_')]
test_set  = df_pairs[df_pairs.relative_path.str.contains('Test_')]
train_df  = train_val.sample(frac=0.8, random_state=42)
val_df    = train_val.drop(train_df.index)

loaders = {
    'train': DataLoader(MammoPatchDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn),
    'val':   DataLoader(MammoPatchDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn),
    'test':  DataLoader(MammoPatchDataset(test_set), batch_size=1,            shuffle=False, collate_fn=collate_fn),
}

# 5. Train/Val/Test loop
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = TransformerUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
best_val  = float('inf')

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss, train_count = 0.0, 0
    for imgs, masks in loaders['train']:
        if imgs.numel()==0: continue
        imgs, masks = imgs.to(device), masks.long().to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks.squeeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*imgs.size(0)
        train_count += imgs.size(0)
    train_loss = train_loss/train_count if train_count else 0.0

    model.eval()
    val_loss, val_count = 0.0, 0
    with torch.no_grad():
        for imgs, masks in loaders['val']:
            if imgs.numel()==0: continue
            imgs, masks = imgs.to(device), masks.long().to(device)
            val_loss += criterion(model(imgs), masks.squeeze(1)).item()*imgs.size(0)
            val_count += imgs.size(0)
    val_loss = val_loss/val_count if val_count else 0.0

    print(f"Epoch {epoch}/{EPOCHS} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), OUTPUT_PATH)
        print("  Saved best model.")

# Test accuracy
model.load_state_dict(torch.load(OUTPUT_PATH, map_location=device))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, masks in loaders['test']:
        if imgs.numel()==0: continue
        imgs, masks = imgs.to(device), masks.long().to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == masks.squeeze(1)).sum().item()
        total   += preds.numel()
accuracy = (correct/total*100) if total else 0.0
print(f"Test Pixel Accuracy: {accuracy:.2f}%")