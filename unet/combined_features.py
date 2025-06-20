import os
import glob
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import local_binary_pattern, hog

# ───────── CONFIGURATION ───────────────────────────────────────────
DDSM_ROOT    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
FULL_CSV     = os.path.join(DDSM_ROOT, "data/full_mammogram_paths.csv")
CROPS_ROOT   = os.path.join(DDSM_ROOT, "eval_crops")
OUT_CSV      = os.path.join(DDSM_ROOT, "unet/features_all.csv")

# LBP parameters
P, R = 8, 1
LBP_METHOD = 'uniform'
N_BINS = P + 2

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PPC = (8, 8)
HOG_CPB = (2, 2)
HOG_BN = 'L2-Hys'
HOG_RESIZE = (128, 128)

# GLCM quantization levels
glc_levels = 16

def compute_glcm_feats(roi, distances=[1,2,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=glc_levels):
    """
    Compute averaged GLCM features (contrast, homogeneity, energy, correlation) over given distances and angles.
    """
    # quantize ROI to `levels` gray-levels
    maxv = roi.max() or 1
    q = np.floor_divide(roi.astype(float) * (levels-1), maxv).astype(int)

    I, J = np.ogrid[0:levels, 0:levels]
    feats = {'contrast': [], 'homogeneity': [], 'energy': [], 'correlation': []}

    for d in distances:
        for theta in angles:
            dy = int(round(d * np.sin(theta)))
            dx = int(round(d * np.cos(theta)))
            # vertical shift
            if dy >= 0:
                mat1 = q[dy:, :]
                mat2 = q[:-dy, :]
            else:
                mat1 = q[:dy, :]
                mat2 = q[-dy:, :]
            # horizontal shift
            if dx >= 0:
                m1 = mat1[:, dx:]
                m2 = mat2[:, :-dx]
            else:
                m1 = mat1[:, :dx]
                m2 = mat2[:, -dx:]
            # skip invalid slices
            if m1.size == 0 or m2.size == 0 or m1.shape != m2.shape:
                continue
            # build GLCM
            pairs = (m1 * levels + m2).ravel()
            glcm = np.bincount(pairs, minlength=levels*levels).reshape((levels, levels)).astype(float)
                        # symmetrize and normalize
            glcm = glcm + glcm.T
            S = glcm.sum()
            if S > 0:
                glcm /= S
            # compute stats
            contrast = ((I-J)**2 * glcm).sum()
            homogeneity = (glcm / (1. + np.abs(I-J))).sum()
            energy = (glcm**2).sum()
            mu_i = (I * glcm).sum(axis=1)
            mu_j = (J * glcm).sum(axis=0)
            sigma_i = np.sqrt(((I - mu_i[:,None])**2 * glcm).sum(axis=1))
            sigma_j = np.sqrt(((J - mu_j[None,:])**2 * glcm).sum(axis=0))
            denom = sigma_i[:,None] * sigma_j[None,:]
            # compute correlation safely
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_mat = ((I - mu_i[:,None]) * (J - mu_j[None,:]) * glcm) / denom
            correlation = np.nan_to_num(corr_mat).sum()
            denom = sigma_i[:,None] * sigma_j[None,:]
            # safe correlation: ignore invalid divisions
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_mat = np.where(
                    denom > 0,
                    ((I - mu_i[:,None]) * (J - mu_j[None,:]) * glcm) / denom,
                    0.0
                )
            correlation = np.nansum(corr_mat)

            contrast = ((I-J)**2 * glcm).sum()
            homogeneity = (glcm / (1. + np.abs(I-J))).sum()
            energy = (glcm**2).sum()
            mu_i = (I * glcm).sum(axis=1)
            mu_j = (J * glcm).sum(axis=0)
            sigma_i = np.sqrt(((I - mu_i[:,None])**2 * glcm).sum(axis=1))
            sigma_j = np.sqrt(((J - mu_j[None,:])**2 * glcm).sum(axis=0))
            denom = sigma_i[:,None] * sigma_j[None,:]
            corr_mat = np.where(
                denom > 0,
                ((I - mu_i[:,None]) * (J - mu_j[None,:]) * glcm) / denom,
                0.0
            )
            correlation = corr_mat.sum()

            feats['contrast'].append(contrast)
            feats['homogeneity'].append(homogeneity)
            feats['energy'].append(energy)
            feats['correlation'].append(correlation)

    # average results
    return {
        'glcm_contrast': np.mean(feats['contrast']),
        'glcm_homogeneity': np.mean(feats['homogeneity']),
        'glcm_energy': np.mean(feats['energy']),
        'glcm_correlation': np.mean(feats['correlation'])
    }
    return {
        'glcm_contrast': np.mean(feats['contrast']),
        'glcm_homogeneity': np.mean(feats['homogeneity']),
        'glcm_energy': np.mean(feats['energy']),
        'glcm_correlation': np.mean(feats['correlation'])
    }


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    full_col = next(c for c in df.columns if 'full' in c.lower())
    df['pathology'] = df['pathology'].str.lower().replace({
        'benign_without_callback':'benign',
        'benign':'benign',
        'malignant':'malignant'
    })
    return dict(zip(df[full_col], df['pathology']))


def extract_roi(seg_path):
    img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    mask = img>0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img[y0:y1+1, x0:x1+1]


if __name__ == '__main__':
    label_map = load_labels(FULL_CSV)
    rows = []
    pattern = os.path.join(CROPS_ROOT, '**', 'segmented_image.png')
    for seg_path in glob.glob(pattern, recursive=True):
        rel_dir = os.path.relpath(os.path.dirname(seg_path), CROPS_ROOT)
        rel_full = rel_dir + '.dcm'
        pathology = label_map.get(rel_full)
        if pathology is None:
            continue
        roi = extract_roi(seg_path)
        if roi is None:
            continue
        feat = {'full_path': rel_full, 'seg_path': seg_path, 'pathology': pathology}
        # LBP
        lbp = local_binary_pattern(roi, P, R, LBP_METHOD).astype(int)
        lbp_vals = lbp[roi>0]
        hist, _ = np.histogram(lbp_vals, bins=np.arange(0, N_BINS+1), density=True)
        for i, v in enumerate(hist): feat[f'lbp_{i}'] = v
        # HOG
        roi_r = cv2.resize(roi, HOG_RESIZE, interpolation=cv2.INTER_LINEAR)
        hog_v = hog(roi_r, orientations=HOG_ORIENTATIONS,
                    pixels_per_cell=HOG_PPC, cells_per_block=HOG_CPB,
                    block_norm=HOG_BN, feature_vector=True)
        for i, v in enumerate(hog_v): feat[f'hog_{i}'] = v
        # GLCM
        glcm_feats = compute_glcm_feats(roi)
        feat.update(glcm_feats)
        # Wavelet LBP
        _, (_ , _ , HH) = pywt.dwt2(roi, 'db2')
        wlbp = local_binary_pattern(HH.astype(np.uint8), P, R, LBP_METHOD).astype(int)
        wvals = wlbp[HH>0]
        whist, _ = np.histogram(wvals, bins=np.arange(0, N_BINS+1), density=True)
        for i, v in enumerate(whist): feat[f'wlbp_{i}'] = v
        rows.append(feat)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"✔️ Extracted {len(df_out)} feature vectors → {OUT_CSV}")
