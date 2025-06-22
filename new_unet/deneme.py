import os
import cv2
import torch
import numpy as np
import pydicom
from skimage.filters import threshold_otsu
from skimage.morphology import closing, remove_small_objects, disk
from torch import nn

# ───────── CONFIG ────────────────────────────────────────────────
DDSM_ROOT   = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
MODEL_PATH  = os.path.join(DDSM_ROOT, "new_unet/patch_ds_attention_unet5.pth")
IMG_DICOM   = os.path.join(DDSM_ROOT, "some_full_mammogram.dcm")  # ← your DICOM here
OUT_DIR     = os.path.join(DDSM_ROOT, "new_unet/segment_outputs")
PATCH_SIZE  = 592
MODEL_SIZE  = 592
THR         = 0.65
MIN_AREA    = 5000
DEVICE      = torch.device("cuda" if torch.cuda.is_available()
                           else "mps"   if torch.backends.mps.is_available()
                           else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)


# ───────── MODEL ─────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(True),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

class DSAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = ConvBlock(1,64),  ConvBlock(64,128)
        self.enc3, self.enc4 = ConvBlock(128,256),ConvBlock(256,512)
        self.pool = nn.MaxPool2d(2)
        self.center  = ConvBlock(512,1024)
        self.up4, self.att4 = nn.ConvTranspose2d(1024,512,2,2), SEBlock(512)
        self.dec4     = ConvBlock(1024,512)
        self.up3, self.att3 = nn.ConvTranspose2d(512,256,2,2), SEBlock(256)
        self.dec3     = ConvBlock(512,256)
        self.up2, self.att2 = nn.ConvTranspose2d(256,128,2,2), SEBlock(128)
        self.dec2     = ConvBlock(256,128)
        self.up1      = nn.ConvTranspose2d(128,64,2,2)
        self.dec1     = ConvBlock(128,64)
        self.outc     = nn.Conv2d(64,1,1)

    def forward(self, x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))
        e4=self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        u4,a4 = self.up4(c), self.att4(e4)
        d4 = self.dec4(torch.cat([u4,a4],1))
        u3,a3 = self.up3(d4), self.att3(e3)
        d3 = self.dec3(torch.cat([u3,a3],1))
        u2,a2 = self.up2(d3), self.att2(e2)
        d2 = self.dec2(torch.cat([u2,a2],1))
        u1    = self.up1(d2)
        d1    = self.dec1(torch.cat([u1,e1],1))
        return self.outc(d1)


# ───────── PRE/POST──────────────────────────────────────────────
def compute_breast_mask(img):
    thr = threshold_otsu(img)
    m = img > thr
    m = closing(m, footprint=disk(25))
    return remove_small_objects(m, min_size=50000)

def sliding_window_inference(img, model):
    H,W = img.shape
    stride = PATCH_SIZE // 2
    pad_h = ( ( (H-stride)//stride + 1 )*stride + PATCH_SIZE - H ) % stride
    pad_w = ( ( (W-stride)//stride + 1 )*stride + PATCH_SIZE - W ) % stride
    img_p = np.pad(img, ((0,pad_h),(0,pad_w)), mode="constant")
    Hp,Wp = img_p.shape

    prob_sum   = np.zeros_like(img_p, dtype=np.float32)
    prob_count = np.zeros_like(img_p, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, Hp-PATCH_SIZE+1, stride):
            for x in range(0, Wp-PATCH_SIZE+1, stride):
                patch = img_p[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                u8    = (patch*255).astype(np.uint8)
                patch = cv2.createCLAHE(2.0,(8,8)).apply(u8)/255.0
                patch = cv2.resize(patch, (MODEL_SIZE,MODEL_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
                t = torch.from_numpy(patch[None,None].astype(np.float32)).to(DEVICE)
                out = torch.sigmoid(model(t))[0,0].cpu().numpy()
                out = cv2.resize(out, (PATCH_SIZE,PATCH_SIZE),
                                 interpolation=cv2.INTER_LINEAR)

                prob_sum  [y:y+PATCH_SIZE, x:x+PATCH_SIZE]   += out
                prob_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE]   += 1

    full_prob = prob_sum / (prob_count + 1e-6)
    return full_prob[:H, :W]


# ───────── MAIN ───────────────────────────────────────────────────
def main():
    # load image
    ds  = pydicom.dcmread(IMG_DICOM, force=True)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min())/(img.max()-img.min()+1e-8)

    # load model + strip aux keys
    model = DSAttentionUNet().to(DEVICE)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    for k in list(sd):
        if k.startswith("aux"):
            sd.pop(k)
    model.load_state_dict(sd)

    # breast mask
    breast_m = compute_breast_mask(img)

    # infer
    prob = sliding_window_inference(img, model)

    # binarize + postprocess
    mask = (prob > THR)
    mask &= breast_m
    mask = remove_small_objects(mask, min_size=MIN_AREA)

    # save full mask & prob
    np.save(os.path.join(OUT_DIR, "full_probability.npy"), prob)
    cv2.imwrite(os.path.join(OUT_DIR, "full_mask.png"),
                (mask*255).astype(np.uint8))

    # tight crop
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop_img  = img[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    cv2.imwrite(os.path.join(OUT_DIR, "lesion_crop.png"),
                (crop_img*255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT_DIR, "lesion_mask.png"),
                (crop_mask*255).astype(np.uint8))

    # overlay
    bg = (img*255).astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(bg)
    red[:,:,2] = mask.astype(np.uint8)*255
    overlay = cv2.addWeighted(bg, 1.0, red, 0.5, 0)
    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay.png"), overlay)

    print("Segmentation complete – outputs in", OUT_DIR)

if __name__=="__main__":
    main()