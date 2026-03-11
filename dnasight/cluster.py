from skimage.measure import find_contours
from scipy.ndimage import label as ndi_label
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma  # For masking background
from scipy.ndimage import gaussian_filter, label, center_of_mass
from skimage.morphology import dilation, disk
from skimage.segmentation import random_walker
from tqdm import tqdm
from skimage.feature import blob_log
from skimage.morphology import remove_small_objects
import glob
import numpy.ma as ma

from scipy.ndimage import gaussian_filter, label as ndi_label, center_of_mass, binary_fill_holes
from skimage.restoration import denoise_bilateral

from dnasight.shared import load_annotated_mask

import json
from skimage.segmentation import expand_labels, find_boundaries
from skimage.measure import label as sk_label, regionprops
from skimage.transform import resize
from skimage.color import label2rgb
from skimage import measure
import re


def binary_random_walker_segmentation(
    img,
    sigma=1,
    threshold_factor=1.5,
    dilation_foreground=5,
    dilation_background=10,
    beta=90
):
    """
    1) Blur (sigma)
    2) Simple threshold on mean * factor to get seed mask
    3) Dilate foreground & background to build markers (0=BG, 1=FG, -1=unknown)
    4) Random walker (binary mode) -> 0 or 1 segmentation
    """
    # (optional) light denoise before thresholding; helps noisy AFM
    # comment out if not needed:
    # img = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=3, channel_axis=None)
    # img = (img * 255).astype(np.uint8)

    # 1) blur
    blurred = gaussian_filter(img, sigma=sigma)

    # 2) threshold
    thr = threshold_factor * float(np.mean(blurred))
    seed = blurred > thr

    # 3) markers
    fg = dilation(seed, disk(dilation_foreground)).astype(np.uint8)
    bg = (~dilation(seed, disk(dilation_background)).astype(bool)).astype(np.uint8)

    markers = np.full_like(blurred, fill_value=-1, dtype=np.int32)
    markers[bg == 1] = 0
    markers[fg == 1] = 1

    # 4) random walker (binary mode)
    rw = random_walker(blurred, markers, beta=beta, mode='bf')
    return (rw == 1).astype(np.uint8)  # 0/1


def segment_image(
    image_path: str,
    output_folder: str,
    next_global_id: int,
    sigma=1,
    threshold_factor=1.5,
    dilation_foreground=5,
    dilation_background=10,
    beta=90,
    min_area=200,
):
    """
    Segments a single TIF (grayscale), assigns **GLOBAL** cluster IDs,
    filters small clusters, saves:
      - <name>_segmentation.npy  (binary 0/1)
      - <name>_segmented.png     (overlay with global IDs)
    Returns:
      cluster_rows, next_global_id

      where cluster_rows is a list of dicts:
        {
          'global_cluster_id', 'file', 'local_id',
          'centroid_x', 'centroid_y', 'intensity', 'area'
        }
    """
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return [], next_global_id

    # binary (0/1)
    mask = binary_random_walker_segmentation(
        img,
        sigma=sigma,
        threshold_factor=threshold_factor,
        dilation_foreground=dilation_foreground,
        dilation_background=dilation_background,
        beta=beta,
    )

    # fill small holes (optional)
    mask = binary_fill_holes(mask).astype(np.uint8)

    # 8-connected components via scipy.ndimage.label
    structure8 = np.ones((3, 3), dtype=int)
    labeled_local, num_local = ndi_label(mask.astype(np.uint8), structure=structure8)

    cluster_rows = []

    # Remove too-small regions from the labeled map in-place (so overlay looks clean)
    for local_id in range(1, num_local + 1):
        region_mask = (labeled_local == local_id)
        area = int(region_mask.sum())
        if area < min_area:
            labeled_local[region_mask] = 0  # drop

    # Re-label after filtering (so labels are contiguous for overlay color)
    labeled_local, num_local = ndi_label((labeled_local > 0).astype(np.uint8), structure=structure8)

    # Build rows + assign **GLOBAL** IDs
    for local_id in range(1, num_local + 1):
        region_mask = (labeled_local == local_id)
        area = int(region_mask.sum())
        if area <= 0:
            continue

        intensity = float(img[region_mask].sum())
        cy, cx = center_of_mass(region_mask)
        if np.isnan(cx) or np.isnan(cy):
            cx = cy = np.nan

        cluster_rows.append({
            "global_cluster_id": int(next_global_id),
            "file": base,
            "local_id": int(local_id),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "intensity": intensity,
            "area": area
        })
        next_global_id += 1

    # Save the *filtered, relabeled* INT map so local_id matches CSV
    npy_filename = f"{stem}_segmentation.npy"
    np.save(os.path.join(output_folder, npy_filename), labeled_local.astype(np.int32))


    # Overlay: show relabeled_local with transparent background
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    masked = ma.masked_where(labeled_local == 0, labeled_local)
    ax.imshow(masked, cmap="jet", alpha=0.4)
    ax.set_title(f"Segmentation: {base}")
    ax.axis("off")

    # Drop text labels (global IDs) slightly offset so they don't sit on centroids
    offset = 15
    for r in cluster_rows:
        x, y = r["centroid_x"], r["centroid_y"]
        if not (np.isnan(x) or np.isnan(y)):
            ax.text(
                x + offset, y + offset,
                str(r["global_cluster_id"]),
                color="yellow",
                fontsize=6,
                ha="center", va="center",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none")
            )

    png_filename = f"{stem}_segmented.png"
    plt.savefig(os.path.join(output_folder, png_filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return cluster_rows, next_global_id


def process_folder(
    input_folder: str,
    output_folder: str,
    sigma=1,
    threshold_factor=1.5,
    dilation_foreground=5,
    dilation_background=10,
    beta=90,
    min_area=200
):
    """
    Processes all .tif images. Assigns **GLOBAL** cluster IDs (monotonic)
    across the entire folder. Writes:
      - *_segmentation.npy per image
      - *_segmented.png per image (overlay with global IDs)
      - segmentation_results.csv (has global_cluster_id)
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith((".tif", ".tiff"))]

    image_files.sort()

    all_rows = []
    next_global_id = 1

    for fname in tqdm(image_files, desc="Processing Images", unit="file"):
        image_path = os.path.join(input_folder, fname)
        rows, next_global_id = segment_image(
            image_path=image_path,
            output_folder=output_folder,
            next_global_id=next_global_id,
            sigma=sigma,
            threshold_factor=threshold_factor,
            dilation_foreground=dilation_foreground,
            dilation_background=dilation_background,
            beta=beta,
            min_area=min_area
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "global_cluster_id", "file", "local_id",
        "centroid_x", "centroid_y", "intensity", "area"
    ])
    csv_path = os.path.join(output_folder, "segmentation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone! {len(df)} clusters kept. Results: {csv_path}")

    return df, csv_path


### FOR SMALLER PSF LIKE BLOBS
def _basename_noext(p): 
    return os.path.splitext(os.path.basename(p))[0]

def _draw_disk(mask, cy, cx, r):
    """Rasterize a filled disk into 'mask' in-place."""
    H, W = mask.shape
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(H, int(np.ceil (cy + r)) + 1)
    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(W, int(np.ceil (cx + r)) + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= ((yy - cy)**2 + (xx - cx)**2) <= (r*r)

def _annulus_stats(img, cy, cx, r_in, r_out):
    """Mean/std inside an annulus around (cy,cx). Falls back if thin."""
    H, W = img.shape
    y0 = max(0, int(np.floor(cy - r_out)))
    y1 = min(H, int(np.ceil (cy + r_out)) + 1)
    x0 = max(0, int(np.floor(cx - r_out)))
    x1 = min(W, int(np.ceil (cx + r_out)) + 1)
    if y0 >= y1 or x0 >= x1:
        return np.nan, np.nan, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr2 = (yy - cy)**2 + (xx - cx)**2
    ann = (rr2 <= r_out*r_out) & (rr2 >= r_in*r_in)
    vals = img[y0:y1, x0:x1][ann]
    if vals.size < 10:
        return np.nan, np.nan, vals.size
    return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0), vals.size

# ---------- PSF-first blob segmentation (no RW) ----------
def psf_blob_mask(
    img,
    *,
    # optional light smoothing
    sigma_pre=0.3,
    # LoG scale-space
    log_min_sigma=1.2,
    log_max_sigma=6.0,
    log_num_sigma=12,
    threshold_rel=0.5,   # relative to max intensity (NOT LoG response)
    overlap=0.9,
    # per-blob SNR gating based on annulus background
    k_radius=1.5,         # drawn mask radius = k_radius * sigma
    snr_min=4.3,          # require (peak - bg_mean) / (bg_std + eps) >= snr_min
    bg_inner=6.0,         # annulus inner radius (pixels)
    bg_outer=10.0,        # annulus outer radius (pixels)
    # cleanup
    min_obj_area_px=10
):
    """
    Returns a binary mask with small disks placed at LoG-detected peaks
    that pass an SNR gate relative to a local annulus background.
    """
    img = img.astype(np.float32, copy=False)
    if sigma_pre and sigma_pre > 0:
        work = gaussian_filter(img, sigma=sigma_pre)
    else:
        work = img

    # LoG blob centers; skimage returns (y, x, sigma)
    blobs = blob_log(work, min_sigma=log_min_sigma, max_sigma=log_max_sigma,
                     num_sigma=log_num_sigma, threshold=1e-12, overlap=overlap)

    if blobs.size == 0:
        return np.zeros_like(img, dtype=bool)

    # intensity cutoff for initial filtering (relative to image max)
    img_max = float(work.max()) if work.size else 0.0
    min_peak_abs = threshold_rel * img_max

    H, W = img.shape
    mask = np.zeros((H, W), dtype=bool)
    eps = 1e-8

    for (cy, cx, s) in blobs:
        cyf, cxf = float(cy), float(cx)
        # Robust local peak: take max in a small core disk (radius ~ sigma)
        r_core = max(1.0, float(s))
        y0 = max(0, int(np.floor(cyf - r_core)))
        y1 = min(H, int(np.ceil (cyf + r_core)) + 1)
        x0 = max(0, int(np.floor(cxf - r_core)))
        x1 = min(W, int(np.ceil (cxf + r_core)) + 1)
        core_vals = work[y0:y1, x0:x1]
        if core_vals.size == 0:
            continue
        peak_val = float(core_vals.max())

        if peak_val < min_peak_abs:
            continue

        # local background via annulus
        bg_mean, bg_std, n_bg = _annulus_stats(work, cyf, cxf, bg_inner, bg_outer)
        if not np.isfinite(bg_mean) or not np.isfinite(bg_std) or n_bg < 10:
            # if we cannot estimate bg reliably, be conservative and skip
            continue

        snr = (peak_val - bg_mean) / (bg_std + eps)
        if snr < snr_min:
            continue

        # accept; rasterize small disk
        r_draw = max(1.0, k_radius * float(s))
        _draw_disk(mask, cyf, cxf, r_draw)

    if min_obj_area_px and min_obj_area_px > 1:
        mask = remove_small_objects(mask, min_size=int(min_obj_area_px))

    return mask

# ---------- per-image processing (assign GLOBAL IDs) ----------
def segment_circular_small_with_globals(
    image_path,
    output_folder,
    next_global_id,
    *,
    # LoG+SNR controls (focus on these)
    sigma_pre=0.5,
    log_min_sigma=1.0,
    log_max_sigma=4.0,
    log_num_sigma=8,
    threshold_rel=0.03,
    overlap=0.5,
    k_radius=1.5,
    snr_min=5.0,
    bg_inner=5.0,
    bg_outer=12.0,
    min_obj_area_px=6,
    # legacy shape filters (optional; keep permissive so LoG+SNR does the heavy lifting)
    min_area=6,
    max_area=4000,
    min_circularity=0.2,
    max_eccentricity=0.98,
    min_solidity=0.10
):
    """
    Returns:
      rows (list[dict]), next_global_id (int)
    Saves:
      <stem>_segmentation.npy   (INT-LABELED mask with LOCAL IDs)
      <stem>_segmented.png      (overlay with GLOBAL IDs)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], next_global_id

    # PSF-friendly mask (no RW)
    mask = psf_blob_mask(
        img,
        sigma_pre=sigma_pre,
        log_min_sigma=log_min_sigma,
        log_max_sigma=log_max_sigma,
        log_num_sigma=log_num_sigma,
        threshold_rel=threshold_rel,
        overlap=overlap,
        k_radius=k_radius,
        snr_min=snr_min,
        bg_inner=bg_inner,
        bg_outer=bg_outer,
        min_obj_area_px=min_obj_area_px
    )

    labeled = sk_label(mask, connectivity=2).astype(np.int32, copy=False)
    props = regionprops(labeled, intensity_image=img)

    keep_labels, kept_rows = set(), []
    for rp in props:
        area = float(rp.area)
        if area < min_area or area > max_area:
            continue

        perim = float(rp.perimeter) if rp.perimeter > 0 else np.nan
        circ = (4.0 * np.pi * area / (perim ** 2)) if perim and np.isfinite(perim) and perim > 0 else 0.0
        ecc  = float(getattr(rp, "eccentricity", np.nan))
        sol  = float(getattr(rp, "solidity", np.nan))

        # keep permissive; the SNR gate already trimmed most false positives
        if (circ >= min_circularity) and (np.isnan(ecc) or ecc <= max_eccentricity) and (np.isnan(sol) or sol >= min_solidity):
            keep_labels.add(rp.label)
            cy, cx = rp.centroid
            intensity_sum = float(rp.intensity_image[rp.image].sum()) 
            kept_rows.append({
                "local_id": int(rp.label),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "area": area,
                "perimeter": perim if np.isfinite(perim) else np.nan,
                "circularity": circ,
                "eccentricity": ecc,
                "solidity": sol,
                "equivalent_diameter": float(rp.equivalent_diameter),
                "intensity": intensity_sum
            })

    kept_mask_local = np.where(np.isin(labeled, list(keep_labels)), labeled, 0).astype(np.int32, copy=False)

    # Assign GLOBAL IDs
    rows_out = []
    base = _basename_noext(image_path)
    for row in kept_rows:
        g = int(next_global_id)
        rows_out.append({
            "global_cluster_id": g,
            "file": os.path.basename(image_path),
            "local_id": row["local_id"],
            "centroid_x": row["centroid_x"],
            "centroid_y": row["centroid_y"],
            "intensity": row["intensity"],
            "area": row["area"]
        })
        next_global_id += 1

    # Save INT-LABELED mask + overlay
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f"{base}_segmentation.npy"), kept_mask_local)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    masked = ma.masked_where(kept_mask_local == 0, kept_mask_local)
    ax.imshow(masked, cmap='jet', alpha=0.4)
    ax.set_title(f"PSF-like clusters (LoG+SNR): {os.path.basename(image_path)}")
    ax.axis('off')

    for r in rows_out:
        ax.text(
            r["centroid_x"] + 10, r["centroid_y"] + 10,
            str(r["global_cluster_id"]),
            color='yellow', fontsize=6, ha='center', va='center',
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none')
        )

    plt.savefig(os.path.join(output_folder, f"{base}_segmented.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rows_out, next_global_id

# ---------- folder processor (writes segmentation_results.csv) ----------
def process_folder_circular_small(
    input_folder,
    output_folder,
    *,
    # core knobs to tune (start here)
    sigma_pre=0.5,
    log_min_sigma=1.0,
    log_max_sigma=4.0,
    log_num_sigma=8,
    threshold_rel=0.04,   # up to be stricter; try 0.05–0.1 if still too many
    overlap=0.5,
    k_radius=1.5,
    snr_min=5.0,          # up to be stricter; 6–8 if needed
    bg_inner=6.0,
    bg_outer=14.0,
    min_obj_area_px=6,
    # secondary (shape) filters
    min_area=6,
    max_area=2000,
    min_circularity=0.3,
    max_eccentricity=0.98,
    min_solidity=0.10
):
    """
    Produces:
      - <stem>_segmentation.npy (INT-labeled kept mask using LOCAL IDs)
      - <stem>_segmented.png    (overlay with GLOBAL IDs)
      - segmentation_results.csv with columns:
            global_cluster_id, file, local_id, centroid_x, centroid_y, intensity, area
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    files.sort()

    all_rows, next_global_id = [], 1

    for fname in tqdm(files, desc="Detecting PSF-like clusters", unit="file"):
        fpath = os.path.join(input_folder, fname)
        try:
            rows, next_global_id = segment_circular_small_with_globals(
                image_path=fpath,
                output_folder=output_folder,
                next_global_id=next_global_id,
                sigma_pre=sigma_pre,
                log_min_sigma=log_min_sigma,
                log_max_sigma=log_max_sigma,
                log_num_sigma=log_num_sigma,
                threshold_rel=threshold_rel,
                overlap=overlap,
                k_radius=k_radius,
                snr_min=snr_min,
                bg_inner=bg_inner,
                bg_outer=bg_outer,
                min_obj_area_px=min_obj_area_px,
                min_area=min_area,
                max_area=max_area,
                min_circularity=min_circularity,
                max_eccentricity=max_eccentricity,
                min_solidity=min_solidity
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"{fname}: {e}")
            continue

    df = pd.DataFrame(all_rows, columns=[
        "global_cluster_id","file","local_id","centroid_x","centroid_y","intensity","area"
    ])
    csv_path = os.path.join(output_folder, "segmentation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone! {len(df)} clusters kept. Results: {csv_path}")

    return df, csv_path




def _find_raw_image(stem, filename_in_csv, csv_dir, extra_roots=None):
    """
    Try several sensible places to locate the raw image referenced by `file` in the CSV.
    Returns an absolute path or None.
    """
    exts = ('.tif', '.tiff', '.png', '.jpg')
    candidates = []

    # 1) exactly what CSV says (absolute or relative)
    if filename_in_csv:
        candidates.append(filename_in_csv)
        candidates.append(os.path.join(csv_dir, os.path.basename(filename_in_csv)))

    # 2) parent of csv_dir
    parent = os.path.dirname(csv_dir)
    for ext in exts:
        candidates.append(os.path.join(parent, stem + ext))

    # 3) siblings often used
    sibs = ["input", "images", "raw", "imgs"]
    for sib in sibs:
        for ext in exts:
            candidates.append(os.path.join(parent, sib, stem + ext))

    # 4) any user-provided roots
    for root in (extra_roots or []):
        for ext in exts:
            candidates.append(os.path.join(root, stem + ext))

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    # 5) last-ditch: small recursive search in parent
    try:
        for ext in exts:
            hits = glob.glob(os.path.join(parent, "**", stem + ext), recursive=True)
            if hits:
                return os.path.abspath(hits[0])
    except Exception:
        pass
    return None


def _find_segmentation_npy(stem, csv_dir):
    """
    Find the segmentation npy saved by your code:
      <stem>_segmentation.npy  or  <stem>_kept_seg.npy
    """
    candidates = [
        os.path.join(csv_dir, f"{stem}_segmentation.npy"),
        os.path.join(csv_dir, f"{stem}_kept_seg.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # last-ditch: look one level up (sometimes results live with PNGs)
    parent = os.path.dirname(csv_dir)
    for c in [os.path.join(parent, f"{stem}_segmentation.npy"),
              os.path.join(parent, f"{stem}_kept_seg.npy")]:
        if os.path.isfile(c):
            return c
    return None


def _load_labeled_seg(seg_path):
    """
    Accepts either a binary mask (0/1/bool) or an int-labeled array.
    Returns an int32 labeled map (0=bg, 1..K = local_id).
    """
    arr = np.load(seg_path)
    if arr.dtype == bool or np.unique(arr).size <= 2:
        lab, _ = ndi_label(arr.astype(bool))
        return lab.astype(np.int32, copy=False)
    # already labeled
    return arr.astype(np.int32, copy=False)


def normalize_cluster_metrics(
    csv_path: str,
    pixelsize_csv: str | None,
    output_path: str,
    *,
    image_roots: list[str] | None = None   # optional: extra places to look for raw images
):
    """
    Normalize + locally background-correct cluster metrics using the ring
    between 2px and 5px dilations of each cluster mask.

    Input CSV must come from your segmentation pipeline and contain:
      ['global_cluster_id','file','local_id','intensity','area', ...]
    """
    df = pd.read_csv(csv_path)

    # ---- required columns
    required = {"global_cluster_id", "file", "local_id", "area", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    # Keep local_id (needed to select the right region),
    # but expose global id as 'cluster_id' as the primary id.
    df = df.copy()
    df.rename(columns={"global_cluster_id": "cluster_id"}, inplace=True)

    # pixel-space columns
    df["area_px"] = pd.to_numeric(df["area"], errors="coerce")
    df["summed_intensity_px"] = pd.to_numeric(df["intensity"], errors="coerce")

    # NEW: initialize edge-touch flag
    df["touches_edge_cluster"] = False

    # ---- optional pixel size merge
    have_valid_px_table = False
    if pixelsize_csv and os.path.isfile(pixelsize_csv):
        try:
            px_df = pd.read_csv(pixelsize_csv)
            cols = {c.lower(): c for c in px_df.columns}
            fcol = next((cols[c] for c in ("filename","file","image","name") if c in cols), None)
            pcol = next((cols[c] for c in ("pixelsize","pixel_size_nm","pixel_size","nm_per_pixel") if c in cols), None)
            if fcol and pcol:
                px_df = px_df[[fcol, pcol]].copy()
                px_df.columns = ["filename", "pixelsize"]
                px_df["filename"] = px_df["filename"].astype(str).apply(lambda p: os.path.splitext(os.path.basename(p))[0])
                px_df["pixelsize"] = pd.to_numeric(px_df["pixelsize"], errors="coerce")

                df["__stem"] = df["file"].astype(str).apply(lambda p: os.path.splitext(os.path.basename(p))[0])
                df = df.merge(px_df, left_on="__stem", right_on="filename", how="left").drop(columns=["filename"])
                have_valid_px_table = df["pixelsize"].notna().any()

                if have_valid_px_table:
                    df["cluster_area_nm2"] = df["area_px"] * (df["pixelsize"] ** 2)
                    df["summed_intensity_per_nm2"] = df["summed_intensity_px"] / df["cluster_area_nm2"]
                    df["summed_intensity_per_nm"]  = df["summed_intensity_px"] / df["pixelsize"]
        except Exception as e:
            print(f"[normalize_cluster_metrics] Pixel-size merge warning: {e}")

    # ---- local background: 2px - 5px ring
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    se2, se5 = disk(2), disk(5)

    df["bg_ring_mean"] = np.nan
    df["bg_corrected_summed_intensity_px"] = np.nan
    if have_valid_px_table:
        df["bg_corrected_summed_intensity_per_nm2"] = np.nan
        df["bg_corrected_summed_intensity_per_nm"]  = np.nan

    # group per image to avoid reloading repeatedly
    for file_name, sub in df.groupby("file"):
        stem = os.path.splitext(os.path.basename(file_name))[0]

        # locate raw image
        raw_path = _find_raw_image(stem, file_name, csv_dir, extra_roots=image_roots)
        if not raw_path:
            print(f"[normalize_cluster_metrics] raw image not found for '{file_name}'")
            continue
        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"[normalize_cluster_metrics] could not read raw image: {raw_path}")
            continue

        # locate segmentation (npy saved by your pipeline)
        seg_path = _find_segmentation_npy(stem, csv_dir)
        if not seg_path:
            print(f"[normalize_cluster_metrics] segmentation npy not found for '{file_name}'")
            continue
        labeled = _load_labeled_seg(seg_path)

        # process each cluster row for this file via its LOCAL id
        for ridx in sub.index:
            lid = df.at[ridx, "local_id"]
            if not np.isfinite(lid):
                continue
            lid = int(lid)
            comp = (labeled == lid)
            if not comp.any():
                # fall back: if labels were re-indexed later, skip
                continue

            # NEW: mark if the component touches any image edge (original mask)
            # (Assumes labeled and raw share the same shape.)
            try:
                touches = bool(
                    comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any()
                )
                df.at[ridx, "touches_edge_cluster"] = touches
            except Exception:
                # if shapes are unexpected, leave as False and keep going
                pass

            # ring = (dilate 5) minus (dilate 2)
            dil2 = dilation(comp, se2)
            dil5 = dilation(comp, se5)
            ring = np.logical_and(dil5, np.logical_not(dil2))
            if not ring.any():
                # very tight packing; skip bg correction for this row
                continue

            bg_mean = float(raw[ring].mean())
            area_px = float(df.at[ridx, "area_px"]) if np.isfinite(df.at[ridx, "area_px"]) else np.nan
            sum_px  = float(df.at[ridx, "summed_intensity_px"]) if np.isfinite(df.at[ridx, "summed_intensity_px"]) else np.nan
            if not (np.isfinite(bg_mean) and np.isfinite(area_px) and np.isfinite(sum_px)):
                continue

            corrected_sum = sum_px - bg_mean * area_px
            df.at[ridx, "bg_ring_mean"] = bg_mean
            df.at[ridx, "bg_corrected_summed_intensity_px"] = corrected_sum

            if have_valid_px_table:
                pxsz = df.at[ridx, "pixelsize"] if "pixelsize" in df.columns else np.nan
                area_nm2 = df.at[ridx, "cluster_area_nm2"] if "cluster_area_nm2" in df.columns else np.nan
                if np.isfinite(area_nm2) and area_nm2 > 0:
                    df.at[ridx, "bg_corrected_summed_intensity_per_nm2"] = corrected_sum / area_nm2
                if np.isfinite(pxsz) and pxsz > 0:
                    df.at[ridx, "bg_corrected_summed_intensity_per_nm"] = corrected_sum / pxsz

    # ---- select output columns
    base_cols = [
        "cluster_id","file",
        "area_px","summed_intensity_px",
        "bg_ring_mean","bg_corrected_summed_intensity_px",
        "touches_edge_cluster",   # NEW: include in output
    ]
    opt_cols = []
    if "centroid_x" in df.columns: opt_cols.append("centroid_x")
    if "centroid_y" in df.columns: opt_cols.append("centroid_y")
    if have_valid_px_table:
        opt_cols += [
            "pixelsize","cluster_area_nm2",
            "summed_intensity_per_nm2","summed_intensity_per_nm",
            "bg_corrected_summed_intensity_per_nm2","bg_corrected_summed_intensity_per_nm",
        ]
    out_cols = [c for c in base_cols + opt_cols if c in df.columns]

    out = df[out_cols].copy()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Metrics (with local BG correction) saved to: {output_path}")
    return out


def match_annotation_to_seg_file(annotation_filename, segmentation_folder):
    base = os.path.basename(annotation_filename).replace("a_", "").replace(".tif", "")
    seg_file = f"{base}_segmentation.npy"
    seg_path = os.path.join(segmentation_folder, seg_file)
    return seg_path if os.path.exists(seg_path) else None


def read_cluster_segmentation(npy_path, min_area=0):
    """
    Accept both binary masks and int-labeled arrays.
    Returns an int32 labeled map (0=bg, 1..K).
    """
    arr = np.load(npy_path)
    # If binary (bool or only {0,1}), label it
    uniq = np.unique(arr)
    if arr.dtype == bool or (uniq.size <= 2 and set(uniq.tolist()).issubset({0,1})):
        labeled_seg, _ = ndi_label(arr.astype(bool))
    else:
        labeled_seg = arr.astype(np.int32, copy=False)

    if min_area > 0:
        areas = np.bincount(labeled_seg.ravel())
        keep = np.zeros_like(labeled_seg)
        for lbl, a in enumerate(areas):
            if lbl != 0 and a >= min_area:
                keep[labeled_seg == lbl] = lbl
        return keep
    return labeled_seg



def plot_kept_and_removed_clusters(
    raw_image,
    binary_dna,
    labeled_clusters,
    kept_ids,
    removed_ids,
    dilation_radius,
    output_path=None
):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(raw_image, cmap='gray')
    # label DNA once
    dna_labels, _ = ndi_label(binary_dna)

    # DNA contours (green)
    for cid in kept_ids:
        mask = (labeled_clusters == cid)
        mask_d = dilation(mask, disk(dilation_radius))
        for lab in np.unique(dna_labels[mask_d]):
            if lab == 0: continue
            dna_mask = (dna_labels == lab)
            for contour in find_contours(dna_mask.astype(float), 0.5):
                ax.plot(contour[:,1], contour[:,0], 'g-', lw=1, alpha=0.8)

    # kept cluster outlines (red) + IDs (yellow)
    for cid in kept_ids:
        mask = (labeled_clusters == cid)
        mask_d = dilation(mask, disk(dilation_radius))
        for contour in find_contours(mask_d.astype(float), 0.5):
            ax.plot(contour[:,1], contour[:,0], 'r-', lw=1.5)
        ys, xs = np.nonzero(mask)
        if ys.size:
            ax.text(xs.mean(), ys.mean(), str(cid), color='yellow', ha='center')

    # removed IDs annotations
    for cid, reason in removed_ids:
        mask = (labeled_clusters == cid)
        ys, xs = np.nonzero(mask)
        if ys.size:
            if reason in ('edge','multimer','dna_edge'):
                color = 'cyan'
            elif reason == 'no_dna':
                color = 'red'
            else:
                color = 'magenta'
            ax.text(xs.mean(), ys.mean(), str(cid), color=color, ha='center')

    ax.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def process_and_save_discarded(
        annotation_folder,
        segmentation_folder,
        pixelsize_csv,
        output_csv_path,
        dilation_radius=5,
        min_area_nm2=0,
        min_intensity_per_nm2=0,
        image_output_folder=None,
        segmentation_output_folder=None
):
    """
    Filters clusters by area/intensity, discards for no DNA, edge, multimer, dna_edge.
    Saves discarded CSV with extended metrics and overview plots.
    """
    if image_output_folder:
        os.makedirs(image_output_folder, exist_ok=True)
    if segmentation_output_folder:
        os.makedirs(segmentation_output_folder, exist_ok=True)

    px_df = pd.read_csv(pixelsize_csv)
    discarded = []

    for ann in sorted(os.listdir(annotation_folder)):
        if not ann.startswith('a_') or not ann.endswith('.tif'):
            continue
        base = ann.replace('a_', '').replace('.tif', '')
        seg_path = match_annotation_to_seg_file(ann, segmentation_folder)
        if not seg_path:
            print(f"No segmentation for {ann}")
            continue

        # load image and DNA mask
        raw_image, binary_dna = load_annotated_mask(
            os.path.join(annotation_folder, ann),
            dilation_radius=2, do_skeletonize=False)
        labeled_clusters = read_cluster_segmentation(seg_path, min_area=0)
        h, w = raw_image.shape
        dna_labels, _ = ndi_label(binary_dna)

        # pixel size & area
        # Normalize base name
        base = ann.replace("a_", "").replace(".tif", "").strip().lower()

        # Normalize filenames in the DataFrame
        px_df['filename_normalized'] = px_df['filename'].astype(str).str.replace(".tif", "",
                                                                                 case=False).str.strip().str.lower()

        # Match by normalized name
        px_vals = px_df.loc[px_df['filename_normalized'] == base, 'pixelsize'].values

        if len(px_vals) == 0:
            print(f"No pixel size found for {base} - using default 4.0 nm")
            pixelsize_nm = 4.0
        else:
            pixelsize_nm = float(px_vals[0])

        px_area = pixelsize_nm ** 2

        cluster_ids = [c for c in np.unique(labeled_clusters) if c]
        kept_ids = []
        metrics = {}

        # 1) Filter by area & intensity
        for cid in cluster_ids:
            mask = (labeled_clusters == cid)
            if mask.shape != raw_image.shape:
                print(f"Shape mismatch for cluster {cid} in {ann}, skipping.")
                continue
            area_px = int(mask.sum())
            area_nm2 = area_px * px_area
            summed_intensity = float(raw_image[mask].sum())
            summed_intensity_per_nm2 = summed_intensity / area_nm2 if area_nm2 else 0

            if area_nm2 < min_area_nm2:
                reason = 'low_area'
            elif summed_intensity_per_nm2 < min_intensity_per_nm2:
                reason = 'low_intensity'
            else:
                kept_ids.append(cid)
                metrics[cid] = (area_px, area_nm2,
                                summed_intensity, summed_intensity_per_nm2)
                continue

            discarded.append({
                'filename': ann,
                'cluster_id': cid,
                'reason': reason,
                'cluster_area': area_px,
                'cluster_area_nm2': area_nm2,
                'summed_intensity': summed_intensity,
                'summed_intensity_per_nm2': summed_intensity_per_nm2,
                'pixel_size': pixelsize_nm
            })

        # 2) Map DNA for kept
        dna_map = {}
        for cid in kept_ids:
            md = dilation(labeled_clusters == cid, disk(dilation_radius))
            for lab in np.unique(dna_labels[md]):
                if lab:
                    dna_map.setdefault(lab, []).append(cid)

        # 3) Discard no_dna
        for cid in kept_ids[:]:
            md = dilation(labeled_clusters == cid, disk(dilation_radius))
            labs = [lab for lab in np.unique(dna_labels[md]) if lab]
            if not labs:
                area_px, area_nm2, summ, ipn = metrics[cid]
                discarded.append({
                    'filename': ann,
                    'cluster_id': cid,
                    'reason': 'no_dna',
                    'cluster_area': area_px,
                    'cluster_area_nm2': area_nm2,
                    'summed_intensity': summ,
                    'summed_intensity_per_nm2': ipn,
                    'pixel_size': pixelsize_nm
                })
                kept_ids.remove(cid)

        # 4) Discard edge/multimer/dna_edge
        for cid in kept_ids[:]:
            area_px, area_nm2, summ, ipn = metrics[cid]
            mask = (labeled_clusters == cid)
            ys, xs = np.nonzero(mask)
            reason, no_clusters = None, None
            # edge
            if ys.min() == 0 or ys.max() == h - 1 or xs.min() == 0 or xs.max() == w - 1:
                reason = 'edge'
            # multimer
            if reason is None:
                assoc = []
                md = dilation(mask, disk(dilation_radius))
                for lab in np.unique(dna_labels[md]):
                    if lab and len(dna_map.get(lab, [])) > 1:
                        assoc.extend(dna_map[lab])
                if assoc:
                    reason = 'multimer';
                    no_clusters = len(set(assoc))
            # dna_edge
            if reason is None:
                md = dilation(mask, disk(dilation_radius))
                for lab in np.unique(dna_labels[md]):
                    if lab:
                        ys2, xs2 = np.nonzero(dna_labels == lab)
                        if (ys2.min() == 0 or ys2.max() == h - 1 or xs2.min() == 0 or xs2.max() == w - 1):
                            reason = 'dna_edge';
                            break
            if reason:
                discarded.append({
                    'filename': ann,
                    'cluster_id': cid,
                    'reason': reason,
                    'cluster_area': area_px,
                    'cluster_area_nm2': area_nm2,
                    'summed_intensity': summ,
                    'summed_intensity_per_nm2': ipn,
                    'pixel_size': pixelsize_nm
                })
                kept_ids.remove(cid)

        # 5) Save kept segs
        if segmentation_output_folder and kept_ids:
            km = np.isin(labeled_clusters, kept_ids)
            out_seg = np.where(km, labeled_clusters, 0)
            np.save(os.path.join(segmentation_output_folder, f"{base}_kept_seg.npy"), out_seg)

        # 6) Overview
        if image_output_folder:
            rem_list = [(d['cluster_id'], d['reason']) for d in discarded if d['filename'] == ann]
            plot_kept_and_removed_clusters(
                raw_image, binary_dna, labeled_clusters,
                kept_ids, rem_list, dilation_radius,
                output_path=os.path.join(image_output_folder, f"{base}_overview.png")
            )

    # Write discarded CSV
    df = pd.DataFrame(discarded)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved discarded clusters to {output_csv_path}")


def count_clusters_per_image(
    annotation_folder,
    segmentation_folder,
    pixelsize_csv,
    output_csv_path,
    min_area_nm2=0,
    min_intensity_per_nm2=0
):
    """
    Counts clusters per image after removing those below area/intensity thresholds or touching the edge.
    Skips any mask shape mismatches or indexing errors.
    Saves a CSV with columns ['filename', 'no_clusters', 'pixel_size'].
    """
    import os
    import numpy as np
    import pandas as pd

    px_df = pd.read_csv(pixelsize_csv)
    results = []

    for ann in sorted(os.listdir(annotation_folder)):
        if not ann.startswith('a_') or not ann.endswith('.tif'):
            continue
        base = ann.replace('a_','').replace('.tif','')
        ann_path = os.path.join(annotation_folder, ann)

        seg_path = match_annotation_to_seg_file(ann, segmentation_folder)
        if not seg_path:
            print(f"No segmentation for {ann}, skipping.")
            continue

        # look up pixel size for this image
        pix_vals = px_df.loc[px_df.filename == base, 'pixelsize'].values
        pixelsize_nm = float(pix_vals[0]) if len(pix_vals) else 4.0
        px_area = pixelsize_nm ** 2

        raw_image, binary_dna = load_annotated_mask(
            ann_path, dilation_radius=2, do_skeletonize=False)
        labeled_clusters = read_cluster_segmentation(seg_path, min_area=0)
        h, w = raw_image.shape

        valid_count = 0
        for cid in np.unique(labeled_clusters):
            if cid == 0:
                continue
            mask = (labeled_clusters == cid)
            # skip mismatched shapes
            if mask.shape != raw_image.shape:
                print(f"Shape mismatch for cluster {cid} in {ann}, skipping.")
                continue
            try:
                area_px = int(mask.sum())
                area_nm2 = area_px * px_area
                summed_intensity = float(raw_image[mask].sum())
            except Exception as e:
                print(f"Skipping cluster {cid} in {ann} due to error: {e}")
                continue
            intensity_per_nm2 = summed_intensity / area_nm2 if area_nm2 else 0

            # apply thresholds
            if area_nm2 < min_area_nm2 or intensity_per_nm2 < min_intensity_per_nm2:
                continue
            # skip edge-touching
            ys, xs = np.nonzero(mask)
            if np.any(ys == 0) or np.any(ys == h-1) or np.any(xs == 0) or np.any(xs == w-1):
                continue

            valid_count += 1

        results.append({
            'filename': ann,
            'no_clusters': valid_count,
            'pixel_size': pixelsize_nm
        })

    df = pd.DataFrame(results, columns=['filename', 'no_clusters', 'pixel_size'])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved cluster counts to {output_csv_path}")


def link_clusters_to_dna(
    annotation_folder: str,
    cluster_seg_folder: str,
    segmentation_results_csv: str,
    output_csv: str,
    dilation_px: int = 3,
):
    """
    Match clusters to DNA:
      - For each image, dilate each cluster by `dilation_px` pixels (disk).
      - Any DNA that intersects that dilation OR is connected to any DNA
        that intersects it (same connected DNA component) is linked.
    Writes a CSV with columns:
        file, cluster_id (GLOBAL), dna_ids (semicolon-separated), n_dna, touches_edge_dna
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    from scipy.ndimage import label as ndi_label, center_of_mass
    from skimage.morphology import dilation, disk
    from skimage.measure import label as sk_label
    from dnasight.shared import load_annotated_mask

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # --- 1) Load cluster - GLOBAL mapping from segmentation_results.csv
    if not os.path.isfile(segmentation_results_csv):
        raise FileNotFoundError(f"Missing segmentation_results.csv: {segmentation_results_csv}")

    seg_df = pd.read_csv(segmentation_results_csv)
    required_cols = {"global_cluster_id", "file", "local_id"}
    missing = required_cols - set(seg_df.columns)
    if missing:
        raise ValueError(f"{segmentation_results_csv} is missing columns: {missing}")

    # group per file for quick lookups
    file_to_rows = {
        str(f): sub[["local_id", "global_cluster_id", "centroid_x", "centroid_y"]].copy()
        for f, sub in seg_df.groupby("file")
    }
    def _norm(s: str) -> str:
        return os.path.basename(s).strip().lower()

    # --- 2) Index cluster segmentation .npy files
    npy_files = [f for f in os.listdir(cluster_seg_folder) if f.endswith(".npy")]
    if not npy_files:
        raise RuntimeError(f"No *.npy cluster segmentations in: {cluster_seg_folder}")

    stem_to_npy = {}
    for f in npy_files:
        base = os.path.splitext(f)[0]
        stem = re.sub(r"(_segmentation|_kept_seg)$", "", base, flags=re.I)
        stem_to_npy[stem] = os.path.join(cluster_seg_folder, f)

    # --- 3) Walk images by stem, load DNA + clusters, perform linking
    rows_out = []

    for stem, npy_path in sorted(stem_to_npy.items()):
        csv_file_key = f"{stem}.tif"
        if csv_file_key not in file_to_rows:
            alt = f"{stem}.tiff"
            if alt in file_to_rows:
                csv_file_key = alt
        if csv_file_key not in file_to_rows:
            keys_norm = {_norm(k): k for k in file_to_rows.keys()}
            csv_file_key = keys_norm.get(_norm(csv_file_key), None)
            if csv_file_key is None:
                print(f"No rows in segmentation_results.csv for stem '{stem}'. Skipping.")
                continue

        rows_this_file = file_to_rows[csv_file_key]
        rows_this_file["local_id"] = pd.to_numeric(rows_this_file["local_id"], errors="coerce")

        seg = np.load(npy_path)
        if seg.ndim != 2:
            raise ValueError(f"{os.path.basename(npy_path)} is not a 2D array.")
        if np.issubdtype(seg.dtype, np.integer) and seg.max() > 1:
            labeled_clusters = seg.astype(np.int32, copy=False)
        else:
            labeled_clusters, _ = ndi_label(seg > 0)

        # Load DNA annotation
        ann_path_candidates = [
            os.path.join(annotation_folder, f"a_{stem}.tif"),
            os.path.join(annotation_folder, f"{stem}.tif"),
            os.path.join(annotation_folder, f"a_{stem}.tiff"),
            os.path.join(annotation_folder, f"{stem}.tiff"),
        ]
        ann_path = next((p for p in ann_path_candidates if os.path.isfile(p)), None)
        if ann_path is None:
            print(f"No DNA annotation found for stem '{stem}'.")
            continue

        try:
            raw, ann = load_annotated_mask(ann_path, dilation_radius=0, do_skeletonize=False)
        except Exception as e:
            print(f"load_annotated_mask failed for {os.path.basename(ann_path)}: {e}")
            continue

        if ann.shape != labeled_clusters.shape:
            print(f"Shape mismatch for '{stem}': DNA {ann.shape} vs clusters {labeled_clusters.shape}. Skipping.")
            continue

        if ann.dtype == bool or (np.unique(ann).size <= 2 and set(np.unique(ann).tolist()).issubset({0, 1})):
            ann = sk_label(ann.astype(bool), connectivity=2).astype(np.int32)

        dna_bin = ann > 0
        dna_cc, _ = ndi_label(dna_bin)

        se = disk(int(max(1, dilation_px)))
        centroids_cache = {}

        # --- NEW: compute which DNA IDs touch the image edge
        edge_dna_ids = set()
        if ann.any():
            # any labeled DNA pixel on border?
            top    = np.unique(ann[0, :]);    bot  = np.unique(ann[-1, :])
            left   = np.unique(ann[:, 0]);    right= np.unique(ann[:, -1])
            for arr in (top, bot, left, right):
                edge_dna_ids.update([int(x) for x in arr if x != 0])

        # Process clusters
        for cid in [c for c in np.unique(labeled_clusters) if c != 0]:
            cluster_mask = (labeled_clusters == cid)
            if not cluster_mask.any():
                continue

            g_rows = rows_this_file[rows_this_file["local_id"] == cid]
            if len(g_rows) == 1:
                global_id = int(g_rows.iloc[0]["global_cluster_id"])
            else:
                if csv_file_key not in centroids_cache:
                    centroids_cache[csv_file_key] = rows_this_file.dropna(subset=["centroid_x", "centroid_y"])
                cand = centroids_cache[csv_file_key]
                if len(cand):
                    cy, cx = center_of_mass(cluster_mask)
                    d2 = (cand["centroid_x"] - cx) ** 2 + (cand["centroid_y"] - cy) ** 2
                    j = int(d2.idxmin())
                    global_id = int(cand.loc[j, "global_cluster_id"])
                else:
                    continue

            dil = dilation(cluster_mask, se)

            touched_cc = np.unique(dna_cc[dil & dna_bin])
            touched_cc = [int(k) for k in touched_cc if k != 0]

            dna_ids = set()
            for ccid in touched_cc:
                dna_ids.update(np.unique(ann[dna_cc == ccid]).tolist())

            dna_ids.discard(0)
            dna_ids_sorted = sorted(int(g) for g in dna_ids)

            # --- NEW: flag if any linked DNA touches edge
            touches_edge_dna = any(d in edge_dna_ids for d in dna_ids_sorted)

            rows_out.append({
                "file": csv_file_key,
                "cluster_id": global_id,
                "dna_ids": ";".join(map(str, dna_ids_sorted)),
                "n_dna": len(dna_ids_sorted),
                "touches_edge_dna": touches_edge_dna,   # NEW
            })

    out_df = pd.DataFrame(
        rows_out, columns=["file", "cluster_id", "dna_ids", "n_dna", "touches_edge_dna"]
    ).sort_values(["file", "cluster_id"])
    out_df.to_csv(output_csv, index=False)
    print(f"Linked clusters - DNA saved to: {output_csv}  (rows={len(out_df)})")
    return out_df



import os
import ast
import pandas as pd
import numpy as np


def _parse_id_list(cell):
    """
    Robustly parse a cell that may contain:
      - a JSON/Python-like list string: "[1, 2, 3]"
      - a delimited string: "1,2,3" or "1; 2; 3" or "1 2 3"
      - a scalar int/float
      - empty/NaN
    Returns a list[int].
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, set)):
        return [int(x) for x in cell if pd.notna(x)]
    if isinstance(cell, (int, np.integer)):
        return [int(cell)]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        # Try safe-eval like "[1,2,3]"
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [int(x) for x in v if pd.notna(x)]
        except Exception:
            pass
        # Fallback split on common separators
        seps = [",", ";", "|", " "]
        for sep in seps:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                out = []
                for p in parts:
                    try:
                        out.append(int(float(p)))
                    except Exception:
                        pass
                return out
        # Single number in a string
        try:
            return [int(float(s))]
        except Exception:
            return []
    # last resort
    return []


def build_cluster_centered_summary(
    links_csv: str,
    cluster_quant_csv: str,
    dna_quant_csv: str | None,
    out_csv: str,
) -> pd.DataFrame:
    """
    Produce a cluster-centered table:
      - One row per global cluster_id.
      - Includes cluster metrics (area/intensity + bg-corrected if present).
      - Includes DNA linkage info (list of comp_ids) from links_csv.
      - Sums DNA lengths across all linked comp_ids:
           sum_length_px, sum_length_nm, sum_length_bp
        (missing units stay NaN; sums are 0.0 when no linked DNA).
      - NEW: touches_edge_dna = True if ANY linked DNA has touches_edge_dna==True in dna_quant_csv.
    """
    if not os.path.isfile(links_csv):
        raise FileNotFoundError(f"Missing links CSV: {links_csv}")
    if not os.path.isfile(cluster_quant_csv):
        raise FileNotFoundError(f"Missing cluster quant CSV: {cluster_quant_csv}")

    # --- Read sources ---
    links = pd.read_csv(links_csv)
    if "cluster_id" not in links.columns:
        if "global_cluster_id" in links.columns:
            links = links.rename(columns={"global_cluster_id": "cluster_id"})
        else:
            raise KeyError("links_csv must contain 'cluster_id' (or 'global_cluster_id').")
    if "dna_ids" not in links.columns:
        cand = [c for c in links.columns if c.lower() in {"dna_ids", "dna_id", "dna", "comp_ids", "comp_id"}]
        if cand:
            links = links.rename(columns={cand[0]: "dna_ids"})
        else:
            links["dna_ids"] = ""

    clusters = pd.read_csv(cluster_quant_csv)
    if "cluster_id" not in clusters.columns:
        if "global_cluster_id" in clusters.columns:
            clusters = clusters.rename(columns={"global_cluster_id": "cluster_id"})
        else:
            raise KeyError("cluster_quant_csv must contain 'cluster_id' (or 'global_cluster_id').")

    # Normalize types
    links["cluster_id"] = pd.to_numeric(links["cluster_id"], errors="coerce").astype("Int64")
    clusters["cluster_id"] = pd.to_numeric(clusters["cluster_id"], errors="coerce").astype("Int64")

    # Parse dna_ids to list
    def _parse_id_list(s):
        if pd.isna(s) or str(s).strip() == "":
            return []
        if isinstance(s, (list, tuple, np.ndarray)):
            return [int(x) for x in s if pd.notna(x)]
        # split on ; or , and coerce to ints
        parts = [p for tok in str(s).replace(",", ";").split(";") for p in [tok.strip()] if p]
        out = []
        for p in parts:
            try:
                out.append(int(float(p)))
            except Exception:
                pass
        return out

    links = links.copy()
    links["dna_id_list"] = links["dna_ids"].apply(_parse_id_list)
    links["n_dna_linked"] = links["dna_id_list"].apply(len)

    # Explode to one row per (cluster_id, comp_id)
    exploded = links.explode("dna_id_list", ignore_index=True).rename(columns={"dna_id_list": "comp_id"})
    exploded["comp_id"] = pd.to_numeric(exploded["comp_id"], errors="coerce")

    # --- Optionally bring in DNA lengths + touches flag ---
    dna = None
    touches_col = None
    if dna_quant_csv and os.path.isfile(dna_quant_csv):
        dna = pd.read_csv(dna_quant_csv)

        # ID column
        if "comp_id" not in dna.columns:
            if "gid" in dna.columns:
                dna = dna.rename(columns={"gid": "comp_id"})
            else:
                raise KeyError("dna_quant_csv must contain 'comp_id' (or 'gid').")

        # Detect a touches flag: prefer 'touches_edge_dna', fallback to 'touches_edge'
        if "touches_edge_dna" in dna.columns:
            touches_col = "touches_edge_dna"
        elif "touches_edge" in dna.columns:
            touches_col = "touches_edge"

        keep_cols = [c for c in ["comp_id", "length_px", "length_nm", "length_bp", "pixel_size_nm", touches_col] if c and c in dna.columns]
        dna = dna[keep_cols].copy()
        dna = dna.drop_duplicates(subset=["comp_id"])

        # Ensure boolean dtype for touches column
        if touches_col:
            dna[touches_col] = dna[touches_col].astype(bool)

    # Join exploded links with dna info (if provided)
    if dna is not None and not exploded.empty:
        ex = exploded.merge(dna, on="comp_id", how="left", suffixes=("", "_dna"))
    else:
        ex = exploded.copy()
        for c in ["length_px", "length_nm", "length_bp"]:
            if c not in ex.columns:
                ex[c] = np.nan
        # If we have no DNA table, we cannot compute touches_edge_dna from DNA - leave it absent/False later
        if "touches_edge_dna" not in ex.columns:
            ex["touches_edge_dna"] = np.nan

    # --- Aggregates per cluster ---
    def _sum_or_default(series, default_nan=False):
        if series is None:
            return np.nan if default_nan else 0.0
        return float(np.nansum(series))

    def _any_true(series):
        if series is None or series.size == 0:
            return False
        # treat NaN as False
        return bool(pd.Series(series).fillna(False).astype(bool).any())

    agg_dict = dict(
        sum_length_px=("length_px", lambda s: _sum_or_default(s, default_nan=False)),
        sum_length_nm=("length_nm", lambda s: _sum_or_default(s, default_nan=True)),
        sum_length_bp=("length_bp", lambda s: _sum_or_default(s, default_nan=True)),
        n_dna_linked=("comp_id", lambda s: int(np.sum(pd.notna(s)))),
        dna_ids_list=("comp_id", lambda s: ";".join(str(int(x)) for x in s.dropna().astype(int)) if s.notna().any() else ""),
    )
    # NEW: cluster-level touches_edge_dna = any linked DNA True
    if "touches_edge_dna" in ex.columns:
        agg_dict["touches_edge_dna"] = ("touches_edge_dna", _any_true)

    agg = ex.groupby("cluster_id", dropna=False).agg(**agg_dict).reset_index()

    # Ensure rows for clusters that had no links at all
    base = clusters[["cluster_id", "file"]].copy() if "file" in clusters.columns else clusters[["cluster_id"]].copy()
    out = base.merge(agg, on="cluster_id", how="left")

    # Fill defaults when no links:
    out["n_dna_linked"] = out["n_dna_linked"].fillna(0).astype(int)
    out["dna_ids_list"] = out["dna_ids_list"].fillna("")
    out["sum_length_px"] = out["sum_length_px"].fillna(0.0)
    # If touches isn't present from DNA table, create it as False
    if "touches_edge_dna" not in out.columns:
        out["touches_edge_dna"] = False
    else:
        out["touches_edge_dna"] = out["touches_edge_dna"].fillna(False).astype(bool)

    # Bring all cluster metrics onto the same row
    summary = clusters.merge(
        out.drop(columns=["file"], errors="ignore"),
        on="cluster_id",
        how="left",
        suffixes=("", "_x"),
    )

    # Write
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Cluster-centered summary saved to: {out_csv} (rows={len(summary)})")
    return summary

# trackpy-based segmentation that outputs the same "segmentation_results.csv"
# ---------------------------------------------------------------------------
# Requirements: trackpy, numpy, pandas, matplotlib, tifffile, scikit-image, opencv-python

import os, numpy as np, pandas as pd, numpy.ma as ma
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

import tifffile
import trackpy as tp

from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
from skimage.draw import disk as rr_disk

# ------------------------
# Helpers
# ------------------------
def _read_gray_cv(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read {path}")
    return img.astype(np.float32)

def _safe_crop(img, cx, cy, r):
    H, W = img.shape
    x0 = int(max(0, np.floor(cx - r)))
    x1 = int(min(W, np.ceil(cx + r + 1)))
    y0 = int(max(0, np.floor(cy - r)))
    y1 = int(min(H, np.ceil(cy + r + 1)))
    patch = img[y0:y1, x0:x1]
    return patch, (slice(y0, y1), slice(x0, x1)), (cx - x0, cy - y0)

def detect_with_trackpy(
    img,
    *,
    diameter=11,
    minmass=300,
    separation=None,
    percentile=64,
    threshold=None,
    invert=False,
    smoothing_size=None,
    locate_kwargs=None
):
    if separation is None:
        separation = diameter
    kws = dict(
        diameter=diameter, minmass=minmass, separation=separation,
        percentile=percentile, threshold=threshold, invert=invert,
        smoothing_size=smoothing_size, preprocess=True
    )
    if locate_kwargs:
        kws.update(locate_kwargs)
    f = tp.locate(img, **kws)
    if f is None or len(f) == 0:
        return pd.DataFrame(columns=["x","y","mass","size","ecc","signal","raw_mass"])
    return f.reset_index(drop=True)

def segment_one_per_detection(
    img, feats_df,
    *,
    window_radius=8,
    invert=False,           # True if spots are darker than background
    method="otsu",          # "otsu" or "percentile"
    percentile=90,
    min_area=5,
    fallback_radius=5
):
    """
    Returns:
      masks_stack  : (N, H, W) bool stack (overlapping allowed)
      centers_xy   : (N, 2) float array of centers (x,y)
      areas        : (N,) int area of each mask (bool sum)
      intensities  : (N,) float sum of pixel values under each mask
    """
    H, W = img.shape
    N = len(feats_df)
    masks = np.zeros((N, H, W), dtype=bool)
    centers = np.zeros((N, 2), dtype=float)
    areas = np.zeros(N, dtype=int)
    intensities = np.zeros(N, dtype=float)

    for i, row in enumerate(feats_df.itertuples(index=False)):
        cx, cy = float(row.x), float(row.y)
        centers[i] = (cx, cy)

        # Crop local patch
        patch, slc, rel = _safe_crop(img, cx, cy, window_radius)
        rx, ry = rel  # (x,y) in patch coords

        if patch.size == 0:
            # Edge fallback: draw disk on full canvas
            rr, cc = rr_disk((int(round(cy)), int(round(cx))), fallback_radius, shape=(H, W))
            m = np.zeros((H, W), dtype=bool); m[rr, cc] = True
            masks[i] = m
        else:
            # Local threshold
            thr = threshold_otsu(patch) if method == "otsu" else np.percentile(patch, percentile)
            bin_local = (patch < thr) if invert else (patch > thr)
            if min_area and min_area > 1:
                bin_local = remove_small_objects(bin_local, min_size=min_area)

            lab = sk_label(bin_local, connectivity=2)
            if lab.max() == 0:
                rr, cc = rr_disk((int(round(cy)), int(round(cx))), fallback_radius, shape=(H, W))
                m = np.zeros((H, W), dtype=bool); m[rr, cc] = True
                masks[i] = m
            else:
                c_lab = lab[int(round(ry)), int(round(rx))]
                if c_lab == 0:
                    # pick closest component to (rx,ry)
                    best_l, best_d2 = None, np.inf
                    for rp in regionprops(lab):
                        py, px = rp.centroid
                        d2 = (px - rx)**2 + (py - ry)**2
                        if d2 < best_d2:
                            best_d2, best_l = d2, rp.label
                    c_lab = best_l if best_l is not None else 0

                if c_lab == 0:
                    rr, cc = rr_disk((int(round(cy)), int(round(cx))), fallback_radius, shape=(H, W))
                    m = np.zeros((H, W), dtype=bool); m[rr, cc] = True
                    masks[i] = m
                else:
                    spot_local = (lab == c_lab)
                    m = np.zeros((H, W), dtype=bool)
                    m[slc] = spot_local
                    masks[i] = m

        # per-spot stats on (possibly overlapping) mask
        areas[i] = int(masks[i].sum())
        intensities[i] = float(img[masks[i]].sum()) if areas[i] > 0 else 0.0

    return masks, centers, areas, intensities

def make_local_label_map_exclusive(masks_bool, centers_xy):
    """
    Convert overlapping mask stack into an INT-labeled map with LOCAL IDs (1..N).
    For pixels covered by multiple masks, assign to the nearest center.
    """
    N, H, W = masks_bool.shape
    label_map = np.zeros((H, W), dtype=np.int32)
    ys, xs = np.nonzero(masks_bool.any(axis=0))
    if len(xs) == 0:
        return label_map

    # Build list of candidate indices per pixel
    # For speed, process chunked
    coords = np.stack([xs, ys], axis=1).astype(np.float32)

    # Precompute centers
    centers = centers_xy.astype(np.float32)
    cx = centers[:, 0][:, None]
    cy = centers[:, 1][:, None]

    # For each pixel, find candidate masks covering it
    # (vectorized approach using boolean indexing)
    flat_idx = ys * W + xs
    masks_flat = masks_bool.reshape(N, -1)
    cand = masks_flat[:, flat_idx]  # shape (N, P)
    # For each pixel p, indices of masks that cover it:
    for p in range(cand.shape[1]):
        cover = np.nonzero(cand[:, p])[0]
        if cover.size == 0:
            continue
        x, y = coords[p]
        # nearest center among 'cover'
        dx = centers_xy[cover, 0] - x
        dy = centers_xy[cover, 1] - y
        k = cover[np.argmin(dx*dx + dy*dy)]
        label_map[ys[p], xs[p]] = int(k + 1)  # LOCAL ID = index+1

    return label_map

def save_overlay_with_globals(img, kept_mask_local, rows_out, out_png, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    masked = ma.masked_where(kept_mask_local == 0, kept_mask_local)
    ax.imshow(masked, cmap='jet', alpha=0.4)
    ax.set_title(title)
    ax.axis('off')

    for row in rows_out:
        ax.text(
            row["centroid_x"] + 10, row["centroid_y"] + 10,
            str(row["global_cluster_id"]),
            color='yellow', fontsize=6, ha='center', va='center',
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none')
        )
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------------
# Per-image processing (trackpy - per-spot segmentation - CSV rows)
# ------------------------
def segment_trackpy_with_globals(
    image_path,
    output_folder,
    next_global_id,
    *,
    # detection
    diameter=11, minmass=300, separation=None, percentile=64, threshold=None, invert=False, smoothing_size=None,
    # segmentation
    window_radius=8, seg_invert=False, seg_method="otsu", seg_percentile=90, min_area=5, fallback_radius=5,
    # filters to match your previous logic (apply AFTER segmentation)
    filter_min_area=10, filter_max_area=2050,
    filter_min_circularity=0.70, filter_max_eccentricity=0.95, filter_min_solidity=0.20
):
    """
    Returns:
      rows_out (list[dict]), next_global_id (int)
    Saves:
      <stem>_segmentation.npy (INT-labeled kept mask using LOCAL IDs)
      <stem>_segmented.png    (overlay with GLOBAL IDs)
    """
    img = _read_gray_cv(image_path)
    feats = detect_with_trackpy(
        img,
        diameter=diameter, minmass=minmass, separation=separation,
        percentile=percentile, threshold=threshold, invert=invert, smoothing_size=smoothing_size
    )

    # quick exit if no detections
    if len(feats) == 0:
        return [], next_global_id

    # build one mask per detection (may overlap), then exclusive INT label map
    masks, centers, areas, intensities = segment_one_per_detection(
        img, feats,
        window_radius=window_radius, invert=seg_invert,
        method=seg_method, percentile=seg_percentile,
        min_area=min_area, fallback_radius=fallback_radius
    )
    kept_local = make_local_label_map_exclusive(masks, centers)

    # regionprops on exclusive map for geometric filters
    props = regionprops(kept_local, intensity_image=img)
    keep_labels = set()
    rows_tmp = {}

    for rp in props:
        area = float(rp.area)
        if area < filter_min_area or area > filter_max_area:
            continue
        perim = float(rp.perimeter) if rp.perimeter > 0 else np.nan
        circ = (4.0 * np.pi * area / (perim ** 2)) if perim and np.isfinite(perim) and perim > 0 else 0.0
        ecc  = float(getattr(rp, "eccentricity", np.nan))
        sol  = float(getattr(rp, "solidity", np.nan))

        if circ >= filter_min_circularity and (np.isnan(ecc) or ecc <= filter_max_eccentricity) and (np.isnan(sol) or sol >= filter_min_solidity):
            keep_labels.add(rp.label)
            cy, cx = rp.centroid
            intensity_sum = float(rp.intensity_image[rp.image].sum())
            rows_tmp[int(rp.label)] = dict(
                local_id=int(rp.label),
                centroid_x=float(cx),
                centroid_y=float(cy),
                area=area,
                perimeter=perim if np.isfinite(perim) else np.nan,
                circularity=circ,
                eccentricity=ecc,
                solidity=sol,
                equivalent_diameter=float(rp.equivalent_diameter),
                intensity=intensity_sum
            )

    # keep only those LOCAL labels; zero the others
    kept_mask_local = np.where(np.isin(kept_local, list(keep_labels)), kept_local, 0).astype(np.int32, copy=False)

    # Build rows_out with GLOBAL IDs, using the kept labels in ascending order for stability
    rows_out = []
    base = os.path.splitext(os.path.basename(image_path))[0]
    for lid in sorted(keep_labels):
        info = rows_tmp[int(lid)]
        g = int(next_global_id)
        rows_out.append({
            "global_cluster_id": g,
            "file": os.path.basename(image_path),
            "local_id": int(lid),
            "centroid_x": info["centroid_x"],
            "centroid_y": info["centroid_y"],
            "intensity": info["intensity"],
            "area": info["area"]
        })
        next_global_id += 1

    # Save INT-LABELED kept mask & overlay
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f"{base}_segmentation.npy"), kept_mask_local)

    out_png = os.path.join(output_folder, f"{base}_segmented.png")
    title = f"Trackpy circular small: {os.path.basename(image_path)}"
    save_overlay_with_globals(img, kept_mask_local, rows_out, out_png, title)

    # (Optional) compute an overlap flag per detection:
    #   overlap_any = (masks.sum(axis=0) > 1)
    #   ... map back to local IDs if you want a separate diagnostics file.
    return rows_out, next_global_id

# ------------------------
# Folder processor (writes identical CSV schema)
# ------------------------
def process_folder_circular_small_trackpy(
    input_folder,
    output_folder,
    *,
    # detection
    diameter=11, minmass=300, separation=None, percentile=64, threshold=None, invert=False, smoothing_size=None,
    # segmentation
    window_radius=8, seg_invert=False, seg_method="otsu", seg_percentile=90, min_area=5, fallback_radius=5,
    # geometric filters (match your previous defaults)
    min_area_filter=10, max_area_filter=2050,
    min_circularity=0.70, max_eccentricity=0.95, min_solidity=0.20
):
    """
    Produces:
      - <stem>_segmentation.npy (INT-labeled kept mask using LOCAL IDs)
      - <stem>_segmented.png    (overlay with GLOBAL IDs)
      - segmentation_results.csv with columns:
            global_cluster_id, file, local_id, centroid_x, centroid_y, intensity, area
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    files.sort()

    all_rows = []
    next_global_id = 1

    for fname in tqdm(files, desc="Trackpy circular small", unit="file"):
        fpath = os.path.join(input_folder, fname)
        try:
            rows, next_global_id = segment_trackpy_with_globals(
                image_path=fpath,
                output_folder=output_folder,
                next_global_id=next_global_id,
                # detection
                diameter=diameter, minmass=minmass, separation=separation,
                percentile=percentile, threshold=threshold, invert=invert, smoothing_size=smoothing_size,
                # segmentation
                window_radius=window_radius, seg_invert=seg_invert, seg_method=seg_method,
                seg_percentile=seg_percentile, min_area=min_area, fallback_radius=fallback_radius,
                # filters
                filter_min_area=min_area_filter, filter_max_area=max_area_filter,
                filter_min_circularity=min_circularity, filter_max_eccentricity=max_eccentricity,
                filter_min_solidity=min_solidity
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"{fname}: {e}")
            continue

    df = pd.DataFrame(all_rows, columns=[
        "global_cluster_id","file","local_id","centroid_x","centroid_y","intensity","area"
    ])
    csv_path = os.path.join(output_folder, "segmentation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone! {len(df)} clusters kept. Results: {csv_path}")

    return df, csv_path

# --- Add near your other imports ---
def _ensure_trackpy():
    try:
        import trackpy as _tp  # noqa: F401
        return True
    except Exception:
        return False

# --- Existing imports assumed: cv2, numpy, pandas, tqdm, etc. ---
# You already defined:
#   - process_folder(...)                   # the Random-Walker pipeline
#   - process_folder_circular_small_trackpy # the Trackpy pipeline (you pasted)

# Put near your other folder processors
def process_folder_clusters_dispatch(
    *,
    model: str,
    input_folder: str,
    output_folder: str,
    **cfg
):
    """
    Unified entry: chooses RW vs Trackpy and passes parameters.
    Accepts either prefixed keys (rw_sigma, tp_diameter) or bare keys.
    Bare keys are auto-prefixed to the selected model.
    """
    model = (model or "rw").lower().strip()

    # Auto-prefix bare keys to the selected model
    def _auto_prefix(d, pref):
        out = {}
        for k, v in d.items():
            if k in ("model",):
                continue
            if k.startswith("rw_") or k.startswith("tp_"):
                out[k] = v
            else:
                out[f"{pref}{k}"] = v
        return out

    if model in ("rw",):
        cfg = _auto_prefix(cfg, "rw_")
        return process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            sigma=cfg.get("rw_sigma", 1),
            threshold_factor=cfg.get("rw_threshold_factor", 1.5),
            dilation_foreground=cfg.get("rw_dilation_foreground", 5),
            dilation_background=cfg.get("rw_dilation_background", 10),
            beta=cfg.get("rw_beta", 90),
            min_area=cfg.get("rw_min_area", 200),
        )

    elif model in ("trackpy", "tp", "small"):
        try:
            import trackpy  # noqa: F401
        except Exception:
            raise RuntimeError("Trackpy requested but not installed. Try: pip install trackpy")

        cfg = _auto_prefix(cfg, "tp_")
        return process_folder_circular_small_trackpy(
            input_folder=input_folder,
            output_folder=output_folder,
            # detection
            diameter=cfg.get("tp_diameter", 11),
            minmass=cfg.get("tp_minmass", 300),
            separation=cfg.get("tp_separation", None),
            percentile=cfg.get("tp_percentile", 64),
            threshold=cfg.get("tp_threshold", None),
            invert=cfg.get("tp_invert", False),
            smoothing_size=cfg.get("tp_smoothing_size", None),
            # segmentation
            window_radius=cfg.get("tp_window_radius", 8),
            seg_invert=cfg.get("tp_seg_invert", False),
            seg_method=cfg.get("tp_seg_method", "otsu"),
            seg_percentile=cfg.get("tp_seg_percentile", 90),
            min_area=cfg.get("tp_min_area", 5),
            fallback_radius=cfg.get("tp_fallback_radius", 5),
            # geometric filters
            min_area_filter=cfg.get("tp_min_area_filter", 10),
            max_area_filter=cfg.get("tp_max_area_filter", 2050),
            min_circularity=cfg.get("tp_min_circularity", 0.70),
            max_eccentricity=cfg.get("tp_max_eccentricity", 0.95),
            min_solidity=cfg.get("tp_min_solidity", 0.20),
        )

    else:
        raise ValueError(f"Unknown cluster segmentation model: {model!r} (use 'rw' or 'trackpy')")

    
def _normalize_filename_key(s: str) -> str:
    base = os.path.basename(str(s))
    base = re.sub(r'\.(tif|tiff|npy)$', '', base, flags=re.IGNORECASE)
    if base.startswith('a_'):
        base = base[2:]
    base = re.sub(r'(CH1)_+$', r'\1', base)
    return base

def _stem_variants(base_key: str):
    v = {base_key, f"a_{base_key}", re.sub(r'(CH1)$', r'\1_', base_key)}
    tmp = re.sub(r'(CH1)$', r'\1_', base_key)
    v.add(f"a_{tmp}")
    return list(v)

def _find_ml_tiff(dna_annot_folder, stem_or_key, debug=False):
    key = _normalize_filename_key(stem_or_key)
    candidates = []
    for base in {stem_or_key, key}:
        for cand in _stem_variants(_normalize_filename_key(base)):
            for ext in ('.tif', '.tiff'):
                candidates.append(os.path.join(dna_annot_folder, cand + ext))
    for p in candidates:
        if os.path.exists(p):
            if debug: print(f"Using ML TIFF: {os.path.basename(p)}")
            return p
    if debug:
        print(f"No ML TIFF found for stem={stem_or_key}. Tried: {[os.path.basename(x) for x in candidates]}")
    raise FileNotFoundError(f"No ML TIFF found for stem={stem_or_key}")

def _load_dna_ids_with_loader(dna_annot_folder, stem, dilation_radius=0, do_skeletonize=False, debug=False):
    tiff_path = _find_ml_tiff(dna_annot_folder, stem, debug=debug)
    raw, ann = load_annotated_mask(tiff_path, dilation_radius=dilation_radius, do_skeletonize=do_skeletonize)
    if not np.issubdtype(ann.dtype, np.integer):
        ann = (ann > 0).astype(np.uint8)
    uniq = np.unique(ann)
    if (np.array_equal(uniq, [0, 1]) or np.array_equal(uniq, [0]) or
        np.array_equal(uniq, [1, 0]) or np.array_equal(uniq, [0, 255]) or
        np.array_equal(uniq, [255])):
        ann = sk_label((ann > 0), connectivity=2)
    return raw, ann.astype(np.int32, copy=False)

def _align_to_shape(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    return resize(arr, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(arr.dtype)

def _label_adjacency_4n(expanded_labels):
    lab = expanded_labels
    edges = set()
    A, B = lab[:, :-1], lab[:, 1:]
    m = (A > 0) & (B > 0) & (A != B)
    if m.any():
        a = A[m].astype(np.int32); b = B[m].astype(np.int32)
        for x, y in zip(a, b): edges.add((min(x,y), max(x,y)))
    A, B = lab[:-1, :], lab[1:, :]
    m = (A > 0) & (B > 0) & (A != B)
    if m.any():
        a = A[m].astype(np.int32); b = B[m].astype(np.int32)
        for x, y in zip(a, b): edges.add((min(x,y), max(x,y)))
    return edges

class UnionFind:
    def __init__(self, elems):
        self.p = {e: e for e in elems}
        self.r = {e: 0 for e in elems}
    def find(self, x):
        px = self.p.get(x, x)
        if px != x:
            self.p[x] = self.find(px)
        return self.p.get(x, x)
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def _load_lengths_table(lengths_csv_path, debug=False):
    df = pd.read_csv(lengths_csv_path)
    lc = {c.lower(): c for c in df.columns}
    fn_col = next((lc[k] for k in ('filename','file','stem','name') if k in lc), None)
    if fn_col is None:
        raise ValueError("Lengths CSV missing filename column (filename/file/stem/name).")
    id_col = next((lc[k] for k in ('comp_id','dna_global_id','global_id','id','dna_id') if k in lc), None)
    if id_col is None:
        raise ValueError("Lengths CSV missing DNA id column (comp_id/dna_global_id/global_id/id/dna_id).")
    out = pd.DataFrame({
        'filename_key': df[fn_col].astype(str).map(_normalize_filename_key),
        'dna_global_id': pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
    })
    px_col = lc.get('length_px'); nm_col = lc.get('length_nm'); bp_col = lc.get('length_bp')
    out['length_px'] = pd.to_numeric(df[px_col], errors='coerce') if px_col else np.nan
    out['length_nm'] = pd.to_numeric(df[nm_col], errors='coerce') if nm_col else np.nan
    out['length_bp'] = pd.to_numeric(df[bp_col], errors='coerce') if bp_col else np.nan
    out = out.dropna(subset=['dna_global_id']).copy()
    out['dna_global_id'] = out['dna_global_id'].astype(int)
    if debug:
        print(f"Lengths mapping loaded: {len(out)} rows, "
              f"{out['filename_key'].nunique()} files, "
              f"{out['dna_global_id'].nunique()} unique DNA ids")
    return out
# ---------------------------------------------------------------

def summarize_and_make_overlays(
    dna_annot_folder,
    cluster_seg_folder,
    lengths_csv_path,
    output_csv_path,
    output_overlay_folder,
    *,
    dilation_radius_px=15,
    min_overlap_px=1,
    min_dna_component_area_px=1,
    cluster_fill_alpha=0.25,
    cluster_edge_alpha=0.85,
    dna_fill_alpha=0.40,
    dna_edge_color="yellow",
    dna_edge_lw=1.8,
    text_size=8,
    debug=False
):
    """
    Exactly the behavior from your standalone script:
      - writes group_summary.csv
      - writes per-image overlays into output_overlay_folder
    """
    os.makedirs(output_overlay_folder, exist_ok=True)
    seg_csv = os.path.join(cluster_seg_folder, "segmentation_results.csv")
    if not os.path.exists(seg_csv):
        raise FileNotFoundError(f"Missing segmentation_results.csv in {cluster_seg_folder}")
    clusters_df = pd.read_csv(seg_csv)
    len_df = _load_lengths_table(lengths_csv_path, debug=debug)
    rows_out = []

    def _autocontrast(img, p_lo=2, p_hi=98):
        a = img.astype(np.float32, copy=False)
        lo, hi = np.percentile(a, [p_lo, p_hi])
        if hi <= lo:
            lo, hi = a.min(), a.max()
            if hi <= lo:
                return np.zeros_like(a)
        a = (a - lo) / (hi - lo)
        return np.clip(a, 0, 1)

    for fname, sub in tqdm(clusters_df.groupby("file"), desc="Process images", unit="file"):
        stem = os.path.splitext(fname)[0]
        filename_key = _normalize_filename_key(stem)

        cl_path = os.path.join(cluster_seg_folder, f"{stem}_segmentation.npy")
        if not os.path.exists(cl_path):
            if debug: print(f"Missing cluster segmentation: {cl_path}")
            continue
        cluster_labels = np.load(cl_path).astype(np.int32, copy=False)

        try:
            raw, id_map = _load_dna_ids_with_loader(dna_annot_folder, stem, dilation_radius=0, do_skeletonize=False, debug=debug)
        except Exception as e:
            if debug: print(f"{stem}: {e}")
            continue

        if id_map.shape != cluster_labels.shape:
            if debug:
                print(f"{filename_key}: resizing DNA IDs {id_map.shape} -> {cluster_labels.shape}")
            id_map = _align_to_shape(id_map, cluster_labels.shape)

        if raw.ndim == 2 and raw.shape != cluster_labels.shape:
            raw = _align_to_shape(raw, cluster_labels.shape)
        elif raw.ndim == 3 and raw.shape[-2:] != cluster_labels.shape:
            raw = np.stack([_align_to_shape(raw[c], cluster_labels.shape) for c in range(raw.shape[0])], axis=0)

        if min_dna_component_area_px > 1:
            ids = id_map[id_map > 0]
            if ids.size:
                areas = np.bincount(ids)
                small = np.nonzero(areas < min_dna_component_area_px)[0]
                if small.size:
                    id_map[np.isin(id_map, small)] = 0

        expanded = expand_labels(cluster_labels, distance=int(dilation_radius_px)) if dilation_radius_px > 0 else cluster_labels

        kept_local_ids = set(map(int, sub['local_id'].unique()))
        if not kept_local_ids:
            continue
        local_to_global = dict(zip(sub['local_id'].astype(int), sub['global_cluster_id'].astype(int)))

        per_label_dna = {}
        for lid in kept_local_ids:
            m = (expanded == lid)
            if not m.any():
                per_label_dna[lid] = set(); continue
            dids = id_map[m]; dids = dids[dids > 0]
            if dids.size == 0:
                per_label_dna[lid] = set()
            else:
                counts = np.bincount(dids)
                hits = np.nonzero(counts >= min_overlap_px)[0]
                per_label_dna[lid] = set(map(int, hits))

        edges = set()
        for a, b in _label_adjacency_4n(expanded):
            if a in kept_local_ids and b in kept_local_ids:
                edges.add((min(a,b), max(a,b)))
        dna_to_lids = {}
        for lid in kept_local_ids:
            for d in per_label_dna.get(lid, ()):
                dna_to_lids.setdefault(d, []).append(lid)
        for d, lids_list in dna_to_lids.items():
            if len(lids_list) >= 2:
                arr = sorted(set(lids_list))
                for i in range(len(arr)-1):
                    for j in range(i+1, len(arr)):
                        edges.add((arr[i], arr[j]))

        uf = UnionFind(kept_local_ids)
        for a, b in edges:
            uf.union(a, b)
        groups = {}
        for lid in kept_local_ids:
            groups.setdefault(uf.find(lid), set()).add(lid)

        lengths_sub = len_df[len_df['filename_key'] == filename_key].copy()
        lengths_by_id = {int(r['dna_global_id']): r for _, r in lengths_sub.iterrows()}

        for _, lids_set in groups.items():
            cluster_ids_global = sorted([int(local_to_global[lid]) for lid in lids_set])
            dna_ids_group = set()
            for lid in lids_set:
                dna_ids_group.update(per_label_dna.get(lid, set()))
            dna_ids_global = sorted(int(d) for d in dna_ids_group)

            by_id = {int(r['dna_global_id']): r for _, r in lengths_sub.iterrows()}
            lens_px_list, lens_nm_list, lens_bp_list = [], [], []
            for d in dna_ids_global:
                rec = by_id.get(d)
                lens_px_list.append(None if rec is None or pd.isna(rec.get('length_px')) else float(rec['length_px']))
                lens_nm_list.append(None if rec is None or pd.isna(rec.get('length_nm')) else float(rec['length_nm']))
                lens_bp_list.append(None if rec is None or pd.isna(rec.get('length_bp')) else float(rec['length_bp']))

            def _sum_safe(vals):
                arr = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
                return float(np.sum(arr)) if arr else None

            rows_out.append({
                "filename": filename_key,
                "n_clusters_in_group": len(cluster_ids_global),
                "cluster_ids": json.dumps(cluster_ids_global),
                "dna_ids": json.dumps(dna_ids_global),
                "total_length_px": _sum_safe(lens_px_list),
                "total_length_nm": _sum_safe(lens_nm_list),
                "total_length_bp": _sum_safe(lens_bp_list),
                "lengths_px_list": json.dumps(lens_px_list),
                "lengths_nm_list": json.dumps(lens_nm_list),
                "lengths_bp_list": json.dumps(lens_bp_list),
            })

        # overlay
        import matplotlib.pyplot as plt
        H, W = cluster_labels.shape
        cl_props = {rp.label: rp.centroid for rp in regionprops(cluster_labels)}
        dna_props = {rp.label: rp.centroid for rp in regionprops(id_map)}
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        raw_disp = _autocontrast(raw if raw.ndim == 2 else raw[0])
        ax.imshow(raw_disp, cmap='gray', zorder=1)
        ax.set_axis_off()
        ax.set_title(f"{filename_key} - groups/lengths")
        cl_rgb = label2rgb(np.where(cluster_labels > 0, cluster_labels, 0), bg_label=0, alpha=None, image=None, kind='overlay')
        ax.imshow(cl_rgb, alpha=cluster_fill_alpha, zorder=2)
        cl_bound = find_boundaries(cluster_labels, mode='inner')
        ax.imshow(np.where(cl_bound, 1.0, np.nan), cmap='Blues', alpha=cluster_edge_alpha, zorder=3)
        dna_rgb = label2rgb(np.where(id_map > 0, id_map, 0), bg_label=0, alpha=None, image=None, kind='overlay')
        ax.imshow(dna_rgb, alpha=dna_fill_alpha, zorder=4)
        contours = measure.find_contours(id_map.astype(float), level=0.5)
        for cnt in contours:
            ax.plot(cnt[:, 1], cnt[:, 0], color=dna_edge_color, linewidth=dna_edge_lw, zorder=5)
        group_index = 1
        for _, lids_set in sorted(groups.items(), key=lambda kv: min(kv[1])):
            pts = [cl_props.get(int(lid), None) for lid in lids_set if int(lid) in cl_props]
            if len(pts) == 0:
                gy, gx = H/2, W/2
            else:
                gy = float(np.mean([p[0] for p in pts])); gx = float(np.mean([p[1] for p in pts]))
            ax.text(gx, gy, f"G{group_index}  (n={len(lids_set)})",
                    color='w', fontsize=text_size, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.65, pad=2, edgecolor='none'),
                    zorder=6)
            dna_ids_group = set()
            for lid in lids_set:
                dna_ids_group.update(per_label_dna.get(lid, set()))
            for did in sorted(dna_ids_group):
                cyx = dna_props.get(int(did), None)
                if cyx is None: continue
                dy, dx = float(cyx[0]), float(cyx[1])
                rec = lengths_by_id.get(int(did), {})
                bp = rec.get('length_bp', np.nan); nm = rec.get('length_nm', np.nan); px = rec.get('length_px', np.nan)
                if pd.notna(bp):   txt = f"{int(round(bp))} bp"
                elif pd.notna(nm): txt = f"{nm:.1f} nm"
                elif pd.notna(px): txt = f"{int(round(px))} px"
                else:              txt = "len NA"
                ax.text(dx + 4, dy + 4, txt,
                        color='k', fontsize=text_size, ha='center', va='center',
                        bbox=dict(facecolor='yellow', alpha=0.80, pad=1.6, edgecolor='none'),
                        zorder=7)
            group_index += 1
        out_png = os.path.join(output_overlay_folder, f"{filename_key}_group_overlay.pdf")
        fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05, format='pdf')
        plt.close(fig)

    # write CSV
    out_df = pd.DataFrame(rows_out, columns=[
        "filename","n_clusters_in_group","cluster_ids","dna_ids",
        "total_length_px","total_length_nm","total_length_bp",
        "lengths_px_list","lengths_nm_list","lengths_bp_list"
    ])
    out_df.to_csv(output_csv_path, index=False)
    return out_df
