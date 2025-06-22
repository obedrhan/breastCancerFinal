import os
import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from torchvision.transforms import InterpolationMode
from PIL import Image
from new_unet import DSAttentionUNet  # your model definition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")
MODEL_PATH = "patch_ds_attention_unet5.pth"

# ───────── UTILS ────────────────────────────────────────
def load_image(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min())/(img.max()-img.min()+1e-8)
    return img

def compute_breast_mask(img):
    """Coarse breast mask via Otsu + morphological closing."""
    u8 = (img*255).astype(np.uint8)
    thr = threshold_otsu(u8)
    m = (u8 >= thr).astype(np.uint8)
    # close small holes
    m = closing(m, selem=disk(25))
    return m

def remove_small_and_keep_largest(bin_mask, min_size=1000):
    lbl = label(bin_mask)
    out = np.zeros_like(bin_mask)
    props = regionprops(lbl)
    # remove small
    props = [p for p in props if p.area >= min_size]
    if not props:
        return out
    # keep only largest
    largest = max(props, key=lambda p: p.area).label
    out[lbl == largest] = 1
    return out

def sliding_window_predict(model, full_img, breast_mask,
                           patch_size=592, overlap=0.5):
    H, W = full_img.shape
    stride = int(patch_size*(1-overlap))
    # weight window (cosine) to blend seams
    wx = np.hanning(patch_size)[:,None]
    wy = np.hanning(patch_size)[None,:]
    weight = wx * wy

    prob_map  = np.zeros((H,W), dtype=np.float32)
    weight_map= np.zeros((H,W), dtype=np.float32)

    for y in tqdm(range(0, H, stride), desc="Patching Y"):
        for x in range(0, W, stride):
            y1 = min(y+patch_size, H)
            x1 = min(x+patch_size, W)
            y0 = y1-patch_size
            x0 = x1-patch_size
            # skip if no breast in this patch
            if breast_mask[y0:y1, x0:x1].sum() == 0:
                continue

            # prep input
            patch = full_img[y0:y1, x0:x1]
            p = (patch*255).astype(np.uint8)
            pil = Image.fromarray(p)
            pil = pil.resize((patch_size,patch_size), Image.BILINEAR)
            t = torch.from_numpy(np.array(pil)[None,None]/255.0).to(DEVICE).float()

            # forward
            with torch.no_grad():
                out,_,_ = model(t)
                p_out = torch.sigmoid(out)[0,0].cpu().numpy()

            # resize back to original patch
            p_out = cv2.resize(p_out, (patch_size,patch_size), interpolation=cv2.INTER_LINEAR)
            prob_map[y0:y1, x0:x1]    += p_out * weight
            weight_map[y0:y1, x0:x1]  += weight

    # normalize
    prob_map /= (weight_map + 1e-8)
    return prob_map

# ───────── MAIN ─────────────────────────────────────────
def main():
    # load
    model = DSAttentionUNet().to(DEVICE)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # path to your DICOM
    dicom_path = "path/to/your/test.dcm"
    img = load_image(dicom_path)

    # 1) breast mask
    breast_m = compute_breast_mask(img)

    # 2) sliding-window UNet
    prob = sliding_window_predict(model, img, breast_m,
                                  patch_size=592, overlap=0.5)

    # 3) threshold more aggressively
    bin_mask = (prob > 0.75).astype(np.uint8)
    # zero-out outside breast
    bin_mask *= breast_m

    # 4) clean up
    bin_mask = remove_small_and_keep_largest(bin_mask, min_size=2000)

    # save outputs
    cv2.imwrite("full_probability.png",   (prob*255).astype(np.uint8))
    cv2.imwrite("full_mask.png",          bin_mask*255)

    # overlay
    rgb = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgb[bin_mask==1] = (0,0,255)  # red overlay
    cv2.imwrite("overlay.png", rgb)

if __name__ == "__main__":
    main()