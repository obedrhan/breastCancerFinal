#!/usr/bin/env python3
import os
import cv2
import torch
import pydicom
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
from skimage.morphology import remove_small_objects
from skimage.measure   import label, regionprops
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ───────── CONFIG ────────────────────────────────────────────────
DICOM_PATH  = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test_samples/test_image.dcm"
MODEL_PATH  = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/new_unet/patch_ds_attention_unet6s.pth"
MODEL_SIZE  = 592       # must match your training config
THRESHOLD   = 0.5
MIN_AREA_PX = 500       # drop tiny speckles
DEVICE      = torch.device("cuda" if torch.cuda.is_available()
                           else "mps"  if torch.backends.mps.is_available()
                           else "cpu")

# ───────── HELPERS ────────────────────────────────────────────────
def load_and_preprocess(dcm_path, size):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    # normalize
    img = (img - img.min()) / (img.ptp() + 1e-8)
    # CLAHE
    u8 = (img*255).astype(np.uint8)
    img_eq = cv2.createCLAHE(2.0,(8,8)).apply(u8) / 255.0
    # resize for model
    pil = Image.fromarray((img_eq*255).astype(np.uint8))
    pil = pil.resize((size,size), Image.BILINEAR)
    tensor = T.ToTensor()(pil).unsqueeze(0).to(DEVICE)  # 1×1×size×size
    return dcm, img, tensor

def postprocess_mask(prob_map, orig_shape):
    # binarize + remove small
    m = prob_map > THRESHOLD
    m = remove_small_objects(m, min_size=MIN_AREA_PX)
    # keep only the largest connected component
    lbl   = label(m)
    props = regionprops(lbl)
    if not props:
        return np.zeros(orig_shape, bool)
    props.sort(key=lambda r: r.area, reverse=True)
    lesion = lbl == props[0].label
    # upsample to full resolution
    return cv2.resize(
        lesion.astype(np.uint8),
        (orig_shape[1], orig_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

def extract_bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()

# ───────── MAIN ─────────────────────────────────────────────────
def main():
    # load + preprocess
    dcm, orig, inp = load_and_preprocess(DICOM_PATH, MODEL_SIZE)
    H, W = orig.shape

    # load model
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 1,
        classes         = 1
    ).to(DEVICE)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # inference
    with torch.no_grad():
        out = model(inp)                   # 1×1×M×M
        prob = torch.sigmoid(out)[0,0]     # M×M
    prob_full = cv2.resize(
        prob.cpu().numpy(),
        (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    # postprocess
    mask = postprocess_mask(prob_full, (H, W))
    n_px = mask.sum()
    if hasattr(dcm, "PixelSpacing"):
        ps = dcm.PixelSpacing
        area_mm2 = n_px * float(ps[0]) * float(ps[1])
    else:
        area_mm2 = np.nan

    print(f"Lesion: {n_px} px   /   {area_mm2:.1f} mm²")

    # extract bounding box
    x0,y0,x1,y1 = extract_bbox(mask)
    crop_img  = orig[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    # prepare overlay (red lesion on gray)
    gray = (orig*255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[mask, :] = [0, 0, 255]  # BGR red

    # save outputs
    os.makedirs("predictions", exist_ok=True)
    cv2.imwrite("predictions/full_mask.png", (mask*255).astype(np.uint8))
    cv2.imwrite("predictions/overlay.png", overlay)
    cv2.imwrite("predictions/lesion_crop.png", (crop_img*255).astype(np.uint8))

    # display side by side
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(orig, cmap="gray");      axs[0].set_title("Original")
    axs[1].imshow(mask, cmap="gray");      axs[1].set_title("Predicted Mask")
    axs[2].imshow(crop_img, cmap="gray");  axs[2].set_title("Lesion Crop")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()