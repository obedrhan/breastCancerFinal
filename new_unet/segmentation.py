import os
import numpy as np
import cv2
import pydicom
import torch
from PIL import Image
import torchvision.transforms as T
from skimage.morphology import remove_small_objects, opening, disk

# ───────── CONFIG ────────────────────────────────────────────────
# Initialize paths directly here:
INPUT_PATH    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test_samples/test_image2.dcm"
OUTPUT_DIR    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/test_samples/test_mask2.dcm"
WEIGHTS_PATH  = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM/transformer_unet.pth"

DEVICE       = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
IMG_SIZE     = 512        # model input / output size
THRESHOLD    = 0.10       # binarization cutoff
MIN_OBJ_SIZE = 50         # drop tiny islands
PADDING      = 20         # pixels around lesion box

# ───────── MODEL IMPORT ─────────────────────────────────────────────
from train_unet import DSAttentionUNet

def load_model(path):
    model = DSAttentionUNet().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ───────── IMAGE I/O & BREAST CROP ─────────────────────────────────
def load_image(path):
    if path.lower().endswith('.dcm'):
        d = pydicom.dcmread(path, force=True)
        arr = d.pixel_array.astype(np.float32)
        return (arr - arr.min())/(arr.max() - arr.min() + 1e-8)
    else:
        img = np.array(Image.open(path).convert('L'), dtype=np.float32)
        return img/255.0

def breast_crop(img):
    """
    Otsu threshold + opening to find the breast region
    Returns: (crop, (bx,by,bw,bh)) in original coords
    """
    u8 = (img*255).astype(np.uint8)
    _, thr = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    clean = opening(thr, disk(15))
    cnts, _ = cv2.findContours(clean.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback: whole image
        H,W = img.shape
        return img, (0,0,W,H)
    cnt = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w], (x,y,w,h)

# ───────── INFERENCE ON BREAST CROP ────────────────────────────────
def predict_on_crop(model, crop):
    Hc,Wc = crop.shape
    pil = Image.fromarray((crop*255).astype(np.uint8))
    small = pil.resize((IMG_SIZE,IMG_SIZE), Image.BILINEAR)
    tensor = T.ToTensor()(small).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out, *_ = model(tensor)
        prob_small = torch.sigmoid(out)[0,0].cpu().numpy()
    # back to original crop resolution
    prob = cv2.resize(prob_small, (Wc,Hc), interpolation=cv2.INTER_LINEAR)
    return prob

# ───────── POST-PROCESS & LESION BOX ───────────────────────────────
def post_process(prob):
    """
    1) threshold
    2) remove tiny specks
    3) find all CCs
    4) drop any CC that touches the crop border
    5) pick the largest remaining CC
    """
    binm = prob > THRESHOLD
    clean = remove_small_objects(binm, min_size=MIN_OBJ_SIZE)
    cnts, _ = cv2.findContours(clean.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    H,W = clean.shape
    inside = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # skip if it touches any edge of the crop
        if x==0 or y==0 or x+w==W or y+h==H:
            continue
        inside.append((c, cv2.contourArea(c)))

    # if none left after border filter, fall back to largest overall
    if not inside:
        inside = [(c, cv2.contourArea(c)) for c in cnts]

    # pick the contour with max area
    lesion_cnt, _ = max(inside, key=lambda tup: tup[1])
    x,y,w,h = cv2.boundingRect(lesion_cnt)
    return (x,y,w,h), clean.astype(np.uint8)

# ───────── CROP, MAP BACK & SAVE ───────────────────────────────────
def crop_and_save(img, mask, bbox, breast_box):
    bx,by,_,_ = breast_box
    x,y,w,h = bbox

    # apply padding inside the crop
    x0 = max(0, x - PADDING)
    y0 = max(0, y - PADDING)
    x1 = min(mask.shape[1], x + w + PADDING)
    y1 = min(mask.shape[0], y + h + PADDING)

    # map back to full‐image coordinates
    gx0, gy0 = bx + x0, by + y0
    gx1, gy1 = bx + x1, by + y1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) overlay with green box
    overlay = (img*255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (gx0,gy0), (gx1,gy1), (0,255,0), 2)
    Image.fromarray(overlay).save(os.path.join(OUTPUT_DIR,'overlay.png'))

    # 2) full‐size mask
    mask_full = np.zeros_like(img, np.uint8)
    mask_full[by:by+mask.shape[0], bx:bx+mask.shape[1]] = mask
    Image.fromarray((mask_full*255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR,'mask.png'))

    # 3) just the lesion crop
    lesion_crop = img[gy0:gy1, gx0:gx1]
    Image.fromarray((lesion_crop*255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR,'lesion_crop.png'))

# ───────── MAIN ───────────────────────────────────────────────────
def main():
    print(f"Loading model from {WEIGHTS_PATH}...")
    model = load_model(WEIGHTS_PATH)

    print(f"Reading image {INPUT_PATH}...")
    img = load_image(INPUT_PATH)

    print("Cropping to breast area...")
    breast, breast_box = breast_crop(img)

    print("Running segmentation on breast crop...")
    prob = predict_on_crop(model, breast)

    print("Post-processing mask to find lesion bbox...")
    bbox, clean = post_process(prob)
    if bbox is None:
        print("No lesion detected.")
        return

    print(f"Lesion bbox in crop coords: {bbox}")
    crop_and_save(img, clean, bbox, breast_box)
    print("Done.  Outputs saved in:", OUTPUT_DIR)

if __name__=='__main__':
    main()