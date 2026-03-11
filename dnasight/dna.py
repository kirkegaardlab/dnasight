from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os, glob, shutil
import torch
import numpy as np
import tifffile
from tifffile import imread, TiffFileError
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import skeletonize, label, dilation, disk
from scipy.stats import norm
from skimage.measure import find_contours
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
from skimage.filters import difference_of_gaussians
import pandas as pd
from math import sqrt, pi
from skimage.measure import label as sk_label, regionprops
from matplotlib.collections import LineCollection
from collections import defaultdict
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from skimage.draw import line
from scipy.ndimage import grey_dilation
import re
from skimage.filters import gaussian
from skimage.segmentation import clear_border
from scipy.ndimage import binary_dilation
import mahotas as mh
from numba import njit
import sys
import matplotlib.patheffects as pe


from dnasight.shared import load_annotated_mask



# ---------- MODEL INFERENCE ON UNANNOTATED FOLDER ----------
class UnannotatedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(".tif")
        ]
        if not self.image_files:
            raise ValueError("No TIFF images found in the provided folder.")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = tifffile.imread(img_path)
        except Exception as e:
            print(f"Skipping file {img_path} due to error: {e}")
            # Return a marker (None) so the collate_fn can filter this out.
            return None, img_path

        # Optionally apply a transformation.
        if self.transform:
            image = self.transform(image)
        # Normalize image and add channel dimension: shape (1, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        return image, img_path


# Custom collate function to filter out None samples.
def none_filter_collate(batch):
    filtered_batch = [item for item in batch if item[0] is not None]
    if len(filtered_batch) == 0:
        return None
    images, paths = zip(*filtered_batch)
    images = torch.stack(images, dim=0)
    return images, paths


def segment_from_distance_map(distance_map, threshold=0.75, min_distance=5):
    """
    Convert a predicted distance map into a segmentation mask using watershed.
    """
    # Use only high-confidence regions.
    binary_mask = distance_map > threshold

    # Find coordinates of local maxima (seeds).
    coordinates = peak_local_max(distance_map, threshold_abs=threshold, min_distance=min_distance)

    # Create markers: assign a unique integer label to each seed.
    markers = np.zeros_like(distance_map, dtype=np.int32)
    for idx, (row, col) in enumerate(coordinates, start=1):
        markers[row, col] = idx

    # Use watershed on the negative distance map so that seeds flow toward the maxima.
    segmentation = watershed(-distance_map, markers, mask=binary_mask)
    return segmentation


def run_model_on_unannotated(
    model,
    folder_path,
    output_folder,
    device,
    batch_size=1,
    threshold=0.83,
    min_distance=5,
    dog_distance=5.0,
    peak_threshold=0.025,
    min_area=20  # px; after watershed, before skeletonization
):
    """
    Runs UNet on raw TIFFs, segments via watershed, filters by area, then
    builds a skeleton using DoG + threshold, and writes a 2-channel stack:
      [0] raw image
      [1] global-ID map on the skeleton (unique across the entire folder)

    - Global IDs are consecutive and never repeat between images.
    - The saved stack is ImageJ-compatible (uint16 if possible, else float32).
    - Also writes a 4-panel diagnostic figure per image, with global ID overlays.

    If the output folder already exists, segmentation is skipped.
    """

    # Output dirs
    annotated_folder = os.path.join(output_folder, "ML_annotated")
    plot_dir = os.path.join(annotated_folder, "segmentation_plots")

    # if os.path.exists(annotated_folder) and os.listdir(annotated_folder):
    #     print(f" Skipping segmentation: folder already exists and is not empty: {annotated_folder}")
    #     return

    os.makedirs(annotated_folder, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Dataset / dataloader
    print('Loading folder', folder_path)
    dataset = UnannotatedDataset(folder_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=none_filter_collate
    )

    # Global running ID
    next_global_id = 1
    save_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            images, paths = batch
            images = images.to(device)
            outputs = model(images)           # (B,1,H,W)
            outputs_np = outputs.cpu().numpy()
            images_np  = images.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                pred_distance = outputs_np[i, 0]  # float [0,1]

                # 1) watershed segmentation
                segmentation = segment_from_distance_map(
                    pred_distance, threshold=threshold, min_distance=min_distance
                )
                pred_binary  = segmentation > 0

                # 2) area filter on connected components
                labeled_mask = label(pred_binary, connectivity=2)
                kept_mask = np.zeros_like(pred_binary, bool)
                for r in regionprops(labeled_mask):
                    if r.area >= min_area:
                        kept_mask[labeled_mask == r.label] = True

                # 3) DoG + threshold skeletonization
                dog = difference_of_gaussians(pred_distance, 1.0, dog_distance)
                binary_skel_seed = np.logical_and(dog > peak_threshold, pred_distance > threshold)
                skeleton_binary = skeletonize(binary_skel_seed)

                # Restrict skeleton to filtered comps
                skeleton_binary &= kept_mask

                # 4) Assign global IDs (+ collect centroids for overlay)
                H, W = skeleton_binary.shape
                id_map_global = np.zeros((H, W), dtype=np.uint32)
                labeled_skel = label(skeleton_binary, connectivity=2)

                comp_labels = [lab for lab in range(1, labeled_skel.max() + 1)]
                gid_centroids = {}  # NEW: global_id -> (cy, cx)

                for lab_id in comp_labels:
                    comp = (labeled_skel == lab_id)
                    if comp.sum() == 0:
                        continue
                    gid = next_global_id
                    id_map_global[comp] = gid
                    next_global_id += 1

                    # centroid (row,col) -> (y,x); use simple mean of pixel coords
                    ys, xs = np.nonzero(comp)
                    cy = float(ys.mean())
                    cx = float(xs.mean())
                    gid_centroids[gid] = (cy, cx)

                # 5) Save 2-channel stack
                base = os.path.splitext(os.path.basename(paths[i]))[0]
                out_name = f"a_{base}.tif"
                save_path = os.path.join(annotated_folder, out_name)

                max_id = int(id_map_global.max())
                if max_id <= 65535:
                    # uint16
                    id_plane = id_map_global.astype(np.uint16)
                    raw_plane = (images_np[i, 0] * 255).astype(np.uint8).astype(np.uint16)
                    stack = np.stack([raw_plane, id_plane], axis=0)
                else:
                    # float32
                    id_plane = id_map_global.astype(np.float32)
                    raw_plane = images_np[i, 0].astype(np.float32)
                    stack = np.stack([raw_plane, id_plane], axis=0)

                tifffile.imwrite(
                    save_path,
                    stack,
                    imagej=True,
                    photometric='minisblack',
                    metadata={'axes': 'CYX'}
                )
                print(f"Saved 2-channel (raw + GLOBAL skeleton IDs): {save_path} | components kept: {len(comp_labels)}")

                # 6) Diagnostics (with global ID overlay on the final panel)
                fig, axes = plt.subplots(1, 4, figsize=(24, 6))

                axes[0].imshow(images_np[i, 0], cmap='gray')
                axes[0].set_title("Raw Image"); axes[0].axis('off')

                im1 = axes[1].imshow(pred_distance, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title("Normalized Proximity Map\nto DNA Backbone"); axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                axes[2].imshow(images_np[i, 0], cmap='gray')
                axes[2].imshow(kept_mask, cmap='Reds', alpha=0.5)
                axes[2].set_title("Segmentation Overlay (kept)"); axes[2].axis('off')

                axes[3].imshow(images_np[i, 0], cmap='gray')
                axes[3].imshow(id_map_global, cmap='nipy_spectral', alpha=0.7)
                axes[3].set_title("Skeleton with Global IDs"); axes[3].axis('off')

                # NEW: overlay numeric global IDs at their centroids on the final panel
                for gid, (cy, cx) in gid_centroids.items():
                    axes[3].text(
                        cx, cy, str(gid),
                        color='yellow', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.55, pad=1.5)
                    )

                plt.tight_layout()
                pdf_name = os.path.splitext(out_name)[0] + ".pdf"   # e.g., a_<basename>.png
                plt.savefig(os.path.join(plot_dir, pdf_name), dpi=150, bbox_inches="tight")
                plt.close(fig)


# ---------------- DNA CALIBRATION & LENGTH QUANTIFICATION ----------------
_OFFS8 = [(-1,-1), (-1,0), (-1,1),
          ( 0,-1),         ( 0, 1),
          ( 1,-1), ( 1,0), ( 1, 1)]

def _step_weight(a, b):
    (y0, x0), (y1, x1) = a, b
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    return 1.0 if (dy == 0 or dx == 0) else sqrt(2.0)

def _is_corner_cut(a, b, coord_set):
    (y0, x0), (y1, x1) = a, b
    dy, dx = y1 - y0, x1 - x0
    if dy == 0 or dx == 0:
        return False
    mid1 = (y0, x1)
    mid2 = (y1, x0)
    return (mid1 in coord_set) or (mid2 in coord_set)

def _build_adjacency(coord_set):
    adj = {p: [] for p in coord_set}
    for p in coord_set:
        y, x = p
        for dy, dx in _OFFS8:
            q = (y+dy, x+dx)
            if q in coord_set and not _is_corner_cut(p, q, coord_set):
                adj[p].append(q)
    return adj

def _graph_segments(adj):
    """
    ...
    Returns:
      segments: list[list[(y,x)]], ordered. Loops do NOT repeat the first point at the end.
      deg:      dict[node] -> degree
    """
    deg = {v: len(ns) for v, ns in adj.items()}
    nodes = {v for v, d in deg.items() if d != 2}
    visited_edge = set()
    segments = []

    def mark(a, b):
        visited_edge.add((a, b)); visited_edge.add((b, a))

    def walk(prev, cur):
        path = [prev, cur]; mark(prev, cur)
        while True:
            nxts = [n for n in adj[cur] if n != prev]
            if len(nxts) != 1 or deg[cur] != 2:
                break
            nxt = nxts[0]
            if (cur, nxt) in visited_edge:
                break
            path.append(nxt); mark(cur, nxt)
            prev, cur = cur, nxt
        return path

    # Start from junctions/endpoints
    for v in nodes:
        for n in adj[v]:
            if (v, n) in visited_edge:
                continue
            segments.append(walk(v, n))

    # Handle cycles (all deg==2)
    for v in adj:
        for n in adj[v]:
            if (v, n) in visited_edge:
                continue
            path = [v, n]; mark(v, n)
            prev, cur = v, n
            while True:
                nxts = [t for t in adj[cur] if t != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if (cur, nxt) in visited_edge:
                    break
                path.append(nxt); mark(cur, nxt)
                prev, cur = cur, nxt
                if cur == v:
                    break
            segments.append(path)

    # Deduplicate
    cleaned, seen = [], set()
    for path in segments:
        if len(path) < 2:
            continue
        unique_edge_found = any((path[i], path[i+1]) not in seen for i in range(len(path)-1))
        if not unique_edge_found:
            continue
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            seen.add((a, b)); seen.add((b, a))
        cleaned.append(path)

    return cleaned, deg   # <-- return both


def _segment_length(path, deg):
    n = len(path)
    if n <= 1:
        return 0.0
    L = sum(_step_weight(path[i], path[i+1]) for i in range(n-1))
    if deg.get(path[0], 0) == 1:
        L += 0.5 * _step_weight(path[0], path[1])
    if deg.get(path[-1], 0) == 1:
        L += 0.5 * _step_weight(path[-2], path[-1])
    return L



def compute_lengths_euclid(
    folder_path: str,
    max_pixel_length: int = 5000,
    dilation_radius: int = 0,
    do_skeletonize: bool = False,
):
    """
    Measure per-component Euclidean lengths (px) from annotated masks.
    Uses load_annotated_mask so it works for BOTH:
      - ML 2-channel TIFFs (raw, mask)
      - ImageJ TIFFs with ROI overlays
    Returns list of rows: {file, comp_id, length_px, touches_edge_dna, H, W}
    """
    rows = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue

        fpath = os.path.join(folder_path, fname)

        # Always go through the robust loader (handles ML + ROI)
        try:
            raw, ann = load_annotated_mask(
                fpath,
                dilation_radius=dilation_radius,
                do_skeletonize=False  # keep full mask; we'll skeletonize per component
            )
        except Exception as e:
            print(f"{fname}: could not load annotated mask ({e}); skipping")
            continue

        lab = sk_label(ann.astype(bool), connectivity=2)

        H, W = ann.shape
        for cid in range(1, lab.max() + 1):
            comp = (lab == cid)
            ys, xs = np.nonzero(comp)
            if ys.size == 0:
                continue

            # edge-touch check
            touches_edge_dna = (
                xs.min() == 0 or xs.max() == W - 1 or
                ys.min() == 0 or ys.max() == H - 1
            )

            # skeletonize the component to a 1-px wide graph
            sk = skeletonize(comp)
            ys, xs = np.nonzero(sk)
            if ys.size == 0:
                continue

            coord_set = set(zip(ys, xs))
            adj = _build_adjacency(coord_set)
            segs, deg = _graph_segments(adj)
            #deg  = {v: len(ns) for v, ns in adj.items()}     # build degrees here
            Lpx = sum(_segment_length(seg, deg) for seg in segs)

            if 0 < Lpx <= max_pixel_length:
                rows.append({
                    'file': fname,
                    'comp_id': cid,
                    'length_px': float(Lpx),
                    'touches_edge_dna': bool(touches_edge_dna),
                    'H': int(H),
                    'W': int(W),
                })

    return rows

def _dna_lengths_by_gid_from_idmap(ann: np.ndarray,
                                   min_area_px: int = 0,
                                   exclude_edge_touching: bool = False,
                                   px_nm: float | None = None,
                                   nm_per_bp_mean: float | None = None):
    """
    Return dict:
      { gid: {
          'dna_length_px': ...,
          'dna_length_nm': ... (if px_nm),
          'dna_length_bp': ... (if nm_per_bp_mean),
          'touches_edge_dna': True/False  # NEW
        }
      }
    """
    out = {}
    rows = _component_lengths_px_from_idmap(
        id_map=ann,
        min_area_px=min_area_px,
        exclude_edge_touching=exclude_edge_touching
    )
    for gid, area_px, touches_edge, length_px in rows:
        rec = {
            'dna_length_px': float(length_px),
            'touches_edge_dna': bool(touches_edge)  # NEW
        }
        if px_nm is not None and np.isfinite(px_nm) and px_nm > 0:
            length_nm = float(length_px) * float(px_nm)
            rec['dna_length_nm'] = length_nm
            if nm_per_bp_mean is not None and np.isfinite(nm_per_bp_mean) and nm_per_bp_mean > 0:
                rec['dna_length_bp'] = length_nm / float(nm_per_bp_mean)
        out[int(gid)] = rec
    return out



# --------------------------------------------------
# Percentile filter (after removing edge-touchers)
# --------------------------------------------------
def filter_by_percentiles(df, perc_low=25.0, perc_high=75.0):
    df = df.copy()
    base = df[~df['touches_edge_dna']]            # remove edge-touchers first
    if len(base) == 0:
        raise RuntimeError("All components touch image edges; adjust dataset or acquisition.")
    q_low, q_high = np.percentile(base['length_px'].to_numpy(), [perc_low, perc_high])
    kept_mask = (~df['touches_edge_dna']) & (df['length_px'] >= q_low) & (df['length_px'] <= q_high)
    df['kept'] = kept_mask
    return df[df['kept']].reset_index(drop=True), df, float(q_low), float(q_high)

# --------------------------------------------------
# Plot helpers
# --------------------------------------------------
def _draw_outlines(ax, lab, mask, color, lw=0.8, alpha=0.8):
    """Draw component outlines given a boolean mask."""
    if mask.dtype != bool: mask = mask.astype(bool)
    # find contours on the mask
    for cnt in find_contours(mask.astype(float), 0.5):
        ax.plot(cnt[:,1], cnt[:,0], color=color, lw=lw, alpha=alpha)

def _edge_touch_ids_from_ann(ann: np.ndarray) -> set[int]:
    """
    Return the set of integer label IDs whose mask touches the image border.
    'ann' can be a binary or labeled array; we (re)label it.
    """
    lab = sk_label(ann.astype(bool), connectivity=2)
    H, W = lab.shape
    edge_ids = set()

    # Any labeled pixel on the outer rim -> edge toucher
    border_mask = np.zeros_like(lab, dtype=bool)
    if H > 0 and W > 0:
        border_mask[0, :]  = True
        border_mask[-1, :] = True
        border_mask[:, 0]  = True
        border_mask[:, -1] = True

    edge_labels = np.unique(lab[border_mask])
    for cid in edge_labels:
        if cid != 0:
            edge_ids.add(int(cid))
    return edge_ids


def _build_edge_touch_index(folder_path: str, files: list[str]) -> dict[str, set[int]]:
    """
    For each filename, load its annotated mask and collect labels that touch the border.
    Returns: {file -> set of comp_id ints that touch border}
    """
    index = {}
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            _, ann = load_annotated_mask(
                fpath,
                dilation_radius=0,
                do_skeletonize=False
            )
            edge_ids = _edge_touch_ids_from_ann(ann)
        except Exception:
            edge_ids = set()
        index[fname] = edge_ids
    return index


def save_compare_panel(
    folder_path: str,
    fname: str,
    df_all: pd.DataFrame,
    out_path: str,
    title_left: str,
    title_right: str,
    color_kept: str    = "#8964CE",  # kept
    color_small: str   = "#FFFFFF",  # too small
    color_big: str     = "#4590D2",  # too big
    color_removed: str = "#BBBBBB",  # fallback if only kept/removed
    color_edge: str    = "#000000"   # edge-touch removals (orange)
):
    """
    Make side-by-side panel (All vs Filtered components) for a single image.
    Expects df_all to include ['file','comp_id'] and either:
      - 'flag' ∈ {'kept','too_small','too_big','edge_touch'}, or
      - boolean 'kept' column (fallback).
    """
    fpath = os.path.join(folder_path, fname)
    try:
        raw, ann = load_annotated_mask(
            fpath,
            dilation_radius=0,
            do_skeletonize=False
        )
    except Exception as e:
        print(f"{fname}: could not load for panel ({e}); skipping panel")
        return

    lab = sk_label(ann.astype(bool), connectivity=2)

    kept_mask   = np.zeros_like(ann, dtype=bool)
    small_mask  = np.zeros_like(ann, dtype=bool)
    big_mask    = np.zeros_like(ann, dtype=bool)
    edge_mask   = np.zeros_like(ann, dtype=bool)
    rem_mask    = np.zeros_like(ann, dtype=bool)  # generic removed fallback

    # Rebuild masks from rows for this file
    rows = df_all[df_all['file'] == fname]
    has_flag = 'flag' in rows.columns
    for _, r in rows.iterrows():
        cid = r.get('comp_id')
        try:
            cid_int = int(cid)
        except Exception:
            continue
        comp = (lab == cid_int)

        if has_flag:
            f = r['flag']
            if f == 'kept':
                kept_mask |= comp
            elif f == 'too_small':
                small_mask |= comp
            elif f == 'too_big':
                big_mask |= comp
            elif f == 'edge_touch':
                edge_mask |= comp
            else:
                rem_mask |= comp
        else:
            if bool(r.get('kept', False)):
                kept_mask |= comp
            else:
                rem_mask  |= comp

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)

    # Left: all components (white outlines)
    axs[0].imshow(raw, cmap='gray'); axs[0].axis('off')
    _draw_outlines(axs[0], lab, ann, color='white', lw=1, alpha=0.9)
    axs[0].text(7, 20, title_left, color='white', fontsize=13, weight='bold')

    # Right: colored by flag
    axs[1].imshow(raw, cmap='gray'); axs[1].axis('off')
    if has_flag:
        if np.any(edge_mask):  _draw_outlines(axs[1], lab, edge_mask,  color=color_edge,   lw=1,   alpha=0.95)
        if np.any(small_mask): _draw_outlines(axs[1], lab, small_mask, color=color_small,  lw=1,   alpha=0.95)
        if np.any(big_mask):   _draw_outlines(axs[1], lab, big_mask,   color=color_big,    lw=1,   alpha=0.95)
        if np.any(kept_mask):  _draw_outlines(axs[1], lab, kept_mask,  color=color_kept,   lw=1.2, alpha=1.0)
    else:
        if np.any(rem_mask):   _draw_outlines(axs[1], lab, rem_mask,   color=color_removed, lw=1, alpha=0.9)
        if np.any(kept_mask):  _draw_outlines(axs[1], lab, kept_mask,  color=color_kept,    lw=1.2, alpha=1.0)

    axs[1].text(7, 20, title_right, color='white', fontsize=13, weight='bold')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def save_hist_all_vs_filtered(df_all, df_kept, dna_bp, px_nm, out_path, bins=30):
    """
    Draw histogram in bp units. Bars are split by 'flag':
      - kept:       "#8964CE"
      - too_small:  "#9B9595"
      - too_big:    "#4590D2"
      - edge_touch: "#E67E22"
    Falls back to 'kept' boolean if 'flag' not present.
    Returns (nm_per_bp_mean, nm_per_bp_sd)
    """
    mean_px = df_kept['length_px'].mean()
    sd_px   = df_kept['length_px'].std(ddof=1) if len(df_kept) > 1 else 0.0
    nm_per_bp_mean = (mean_px * px_nm) / float(dna_bp)
    nm_per_bp_sd   = (sd_px   * px_nm) / float(dna_bp)

    raw_bp  = df_all['length_px'].to_numpy()  * px_nm / nm_per_bp_mean if len(df_all) else np.array([])
    kept_bp = df_kept['length_px'].to_numpy() * px_nm / nm_per_bp_mean if len(df_kept) else np.array([])

    # Binning
    if raw_bp.size == 0:
        bin_edges = np.linspace(0, 1, bins+1)
    else:
        lo, hi = float(np.min(raw_bp)), float(np.max(raw_bp))
        if lo == hi:
            lo -= 0.5; hi += 0.5
        bin_edges = np.linspace(lo, hi, bins+1)

    # Prepare subsets by flag (preferred) or kept/fallback
    has_flag = 'flag' in df_all.columns
    if has_flag:
        colors = {'kept': "#8964CE", 'too_small': "#FFFFFF", 'too_big': "#4590D2", 'edge_touch': "#000000"}
        data_by_flag = {}
        for key in ['edge_touch', 'too_small', 'too_big', 'kept']:
            sel = df_all[df_all['flag'] == key]['length_px'].to_numpy() * px_nm / nm_per_bp_mean
            data_by_flag[key] = sel
        order = ['edge_touch', 'too_small', 'too_big', 'kept']
    else:
        colors = {'kept': 'mediumpurple', 'removed': 'lightgray'}
        kept_set = set(df_kept['comp_id'].astype(str))
        all_bp = df_all.copy()
        all_bp['is_kept'] = df_all['comp_id'].astype(str).isin(kept_set)
        data_by_flag = {
            'removed': all_bp.loc[~all_bp['is_kept'], 'length_px'].to_numpy() * px_nm / nm_per_bp_mean,
            'kept':    all_bp.loc[ all_bp['is_kept'], 'length_px'].to_numpy() * px_nm / nm_per_bp_mean,
        }
        order = ['removed', 'kept']

    # Plot
    plt.figure(figsize=(3.6, 3.2))

    # Draw removed subsets first, then kept on top
    labels = []
    for key in order:
        if key not in data_by_flag: 
            continue
        arr = data_by_flag[key]
        lab = f"{key.replace('_',' ')} (N={len(arr)})"
        labels.append(lab)
        plt.hist(arr, bins=bin_edges, alpha=0.75 if key=='kept' else 0.6,
                 color=colors[key], edgecolor='k', label=lab)

    # Overlay Gaussian fit to kept only (if >1 sample)
    if kept_bp.size > 0:
        mu = float(np.mean(kept_bp))
        sigma = float(np.std(kept_bp, ddof=1)) if kept_bp.size > 1 else 0.0
        x = np.linspace(bin_edges[0], bin_edges[-1], 400)
        if sigma > 0:
            pdf = norm.pdf(x, loc=mu, scale=sigma)
            binw = bin_edges[1] - bin_edges[0]
            plt.plot(x, pdf * kept_bp.size * binw, linestyle='--', linewidth=2,
                     color='black', label=f'μ={mu:.1f} bp, σ={sigma:.1f} bp')

    plt.xlabel('Length (bp)')
    plt.ylabel('Count')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    return nm_per_bp_mean, nm_per_bp_sd


def calibrate_folder_percentiles(folder_path, dna_bp, pixel_size_nm, output_folder,
                                 perc_low=25.0, perc_high=75.0,
                                 bins=25, example_images_to_save=None,
                                 max_pixel_length=5000):
    os.makedirs(output_folder, exist_ok=True)

    rows = compute_lengths_euclid(folder_path, max_pixel_length=max_pixel_length)
    if not rows:
        raise RuntimeError(f"No valid components in {folder_path}")
    df = pd.DataFrame(rows)

    # --- NEW: mark edge-touchers (by file + comp_id) ---
    files = sorted(df['file'].astype(str).unique().tolist())
    edge_index = _build_edge_touch_index(folder_path, files)

    def _touches_edge(row) -> bool:
        # comp_id is expected to match skimage label IDs (int)
        cid = row.get('comp_id')
        try:
            cid_int = int(cid)
        except Exception:
            return False
        return cid_int in edge_index.get(str(row['file']), set())

    df = df.copy()
    df['edge_touch'] = df.apply(_touches_edge, axis=1)

    # Keep a copy of edge-touchers to add back (for coloring/CSV), but
    # EXCLUDE them from the percentile-based length filtering:
    df_edge   = df[df['edge_touch']].copy()
    df_clean  = df[~df['edge_touch']].copy()

    # percentile filtering ONLY on the clean (non-edge) set
    df_kept, df_all_clean, ql, qh = filter_by_percentiles(
        df_clean, perc_low=perc_low, perc_high=perc_high
    )

    # --- flags for coloring/overlays/hist ---
    df_all = df_all_clean.copy()
    cond_small = df_all['length_px'] < ql
    cond_big   = df_all['length_px'] > qh
    df_all['flag'] = 'kept'
    df_all.loc[cond_small, 'flag'] = 'too_small'
    df_all.loc[cond_big,   'flag'] = 'too_big'
    df_all['kept'] = (df_all['flag'] == 'kept')
    df_all['edge_touch'] = False  # all here are non-edge by construction

    # Add edge-touchers back as a dedicated removal class
    if len(df_edge):
        df_edge = df_edge.copy()
        df_edge['flag'] = 'edge_touch'
        df_edge['kept'] = False
        # percentiles were computed without these; store q for transparency
        df_edge['q_low_px']  = ql
        df_edge['q_high_px'] = qh
        df_all = pd.concat([df_all, df_edge], ignore_index=True)

    # summary numbers (nm/bp from kept set, i.e., non-edge & in-range)
    mean_px = df_kept['length_px'].mean()
    sd_px   = df_kept['length_px'].std(ddof=1) if len(df_kept) > 1 else 0.0
    nm_per_bp_mean = (mean_px * pixel_size_nm) / float(dna_bp)
    nm_per_bp_sd   = (sd_px   * pixel_size_nm) / float(dna_bp)

    # folder-level histogram (PDF)
    hist_path = os.path.join(output_folder, 'hist_all_vs_filtered.pdf')
    save_hist_all_vs_filtered(df_all, df_kept, dna_bp, pixel_size_nm,
                              hist_path, bins=bins)

    # per-image comparison panels - save ALL by default (as PDF)
    saved = 0
    files = sorted(df_all['file'].unique())
    for fname in files:
        if isinstance(example_images_to_save, int) and saved >= example_images_to_save:
            break
        outp = os.path.join(output_folder, f'compare_{os.path.splitext(fname)[0]}.pdf')
        ttlL = f'All segmentations\n{dna_bp} bp plasmid'
        ttlR = f'Filtered segmentations\n{dna_bp} bp plasmid'
        save_compare_panel(folder_path, fname, df_all, outp, ttlL, ttlR)
        saved += 1

    # CSV (store percentiles + folder nm/bp used for bp-axis calibration)
    df_all['q_low_px']  = ql
    df_all['q_high_px'] = qh
    df_all['nm_per_bp_folder_mean'] = nm_per_bp_mean
    csv_path = os.path.join(output_folder, 'per_component_lengths_and_flags.csv')
    df_all.to_csv(csv_path, index=False)

    print(f"[{os.path.basename(folder_path)}] nm/bp = {nm_per_bp_mean:.4f} ± {nm_per_bp_sd:.4f} (SD) | "
          f"kept {len(df_kept)}/{len(df_all)} | q=[{ql:.1f},{qh:.1f}] px | "
          f"panels saved: {saved}")

    return {
        'folder': folder_path,
        'dna_bp': float(dna_bp),
        'pixel_size_nm': float(pixel_size_nm),
        'perc_low': float(perc_low),
        'perc_high': float(perc_high),
        'nm_per_bp_mean': float(nm_per_bp_mean),
        'nm_per_bp_sd': float(nm_per_bp_sd),
        'N_all': int(len(df_all)),
        'N_kept': int(len(df_kept)),
        'histogram_pdf': hist_path,
        'csv': csv_path
    }, df_all, df_kept



def calibrate_multiple_folders(folders, output_root,
                               default_perc_low=25.0, default_perc_high=75.0,
                               bins=25, example_images_to_save=None, pooled=True):
    os.makedirs(output_root, exist_ok=True)
    per_folder = []
    all_components = []

    for info in folders:
        p    = info['path']
        bp   = info['dna_bp']
        pxnm = info['pixel_size_nm']
        pl   = info.get('perc_low',  default_perc_low)
        ph   = info.get('perc_high', default_perc_high)

        base   = os.path.basename(p.rstrip('/'))
        parent = os.path.basename(os.path.dirname(p.rstrip('/')))
        out    = os.path.join(output_root, f"{parent}__{base}")  # avoid collisions

        summary, df_all, df_kept = calibrate_folder_percentiles(
            folder_path=p, dna_bp=bp, pixel_size_nm=pxnm,
            output_folder=out, perc_low=pl, perc_high=ph,
            bins=bins, example_images_to_save=example_images_to_save
        )

        # Per-structure nm/bp for KEPT items
        nm_per_bp_each = (df_kept['length_px'].to_numpy(dtype=float) * float(pxnm)) / float(bp)
        all_components.append(pd.DataFrame({
            'folder': p,
            'dna_bp': bp,
            'pixel_size_nm': pxnm,
            'nm_per_bp': nm_per_bp_each
        }))

        # Robust per-folder stats (mean, SD, SEM, N)
        N = nm_per_bp_each.size
        mean = float(np.mean(nm_per_bp_each)) if N > 0 else np.nan
        std  = float(np.std(nm_per_bp_each, ddof=1)) if N > 1 else (0.0 if N == 1 else np.nan)
        sem  = float(std / np.sqrt(N)) if N > 1 else (np.nan if N == 0 else 0.0)

        # Merge with summary from calibrate_folder_percentiles (overwrite with robust values)
        summary = dict(summary)
        summary.update({
            'folder': p,
            'dna_bp': bp,
            'pixel_size_nm': pxnm,
            'perc_low': pl,
            'perc_high': ph,
            'nm_per_bp_mean': mean,
            'nm_per_bp_std': std,
            'nm_per_bp_sem': sem,
            'N_kept': int(N),
        })
        per_folder.append(summary)

    # Build DataFrames
    pf = pd.DataFrame(per_folder).sort_values('folder')
    all_struct_df = pd.concat(all_components, ignore_index=True)

    # Aggregates across all structures / folders
    if len(all_struct_df):
        pooled_mean = float(all_struct_df['nm_per_bp'].mean())
        pooled_sem  = float(all_struct_df['nm_per_bp'].std(ddof=1) / np.sqrt(len(all_struct_df))) if len(all_struct_df) > 1 else (0.0 if len(all_struct_df) == 1 else np.nan)
    else:
        pooled_mean, pooled_sem = np.nan, np.nan

    if len(pf):
        folder_mean = float(pf['nm_per_bp_mean'].mean())
        folder_sd   = float(pf['nm_per_bp_mean'].std(ddof=1)) if len(pf) > 1 else (0.0 if len(pf) == 1 else np.nan)
    else:
        folder_mean, folder_sd = np.nan, np.nan

    # Save CSVs
    pf_path = os.path.join(output_root, 'per_folder_summary.csv')
    pf.to_csv(pf_path, index=False)

    all_path = os.path.join(output_root, 'per_structure_nm_per_bp.csv')
    all_struct_df.to_csv(all_path, index=False)

    print("\n=== Aggregated nm/bp (filtered) ===")
    print(f"Pooled over structures: {pooled_mean:.4f} ± {pooled_sem:.4f} (SEM), N={len(all_struct_df)}")
    print(f"Per-folder means:       {folder_mean:.4f} ± {folder_sd:.4f} (SD),  n={len(pf)}")
    print(f"Saved: {pf_path}")
    print(f"Saved: {all_path}")

    return (pooled_mean, pooled_sem) if pooled else (folder_mean, folder_sd)


def _lookup_px_nm_for_file(fname: str, px_map: dict[str, float]) -> float:
    """
    Best-effort pixel-size lookup using multiple filename variants.
    Returns finite nm/px or np.nan.
    """
    root, _ = os.path.splitext(fname)

    def _strip_a(s: str) -> str:
        return s[2:] if s.startswith('a_') else s

    fname0 = _strip_a(fname)
    root0  = _strip_a(root)

    # Try all combinations of (with/without a_) x (with/without extension)
    tries = [
        fname, root,
        fname0, root0,
        'a_' + fname0, 'a_' + root0,
    ]

    for key in tries:
        val = px_map.get(key, np.nan)
        if val is not None and np.isfinite(val) and val > 0:
            return float(val)
    return np.nan



def _component_lengths_px_from_idmap(id_map: np.ndarray,
                                     min_area_px: int,
                                     exclude_edge_touching: bool):
    """
    Measure skeleton length (px) per labeled component in an ID map.
    id_map: 2D array with 0=background, positive ints = global DNA IDs.
    Returns list of tuples: (global_id, area_px, touches_edge_dna, length_px).
    """
    rows = []
    H, W = id_map.shape
    # unique positive IDs
    gids = np.unique(id_map)
    gids = gids[gids > 0]

    for gid in gids:
        comp = (id_map == gid)
        area = int(comp.sum())
        if area < min_area_px:
            continue

        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue

        touches_edge_dna = (xs.min() == 0) or (xs.max() == W - 1) or (ys.min() == 0) or (ys.max() == H - 1)
        if exclude_edge_touching and touches_edge_dna:
            continue

        sk = skeletonize(comp)
        ys, xs = np.nonzero(sk)
        if ys.size == 0:
            continue
        coord_set = set(zip(ys, xs))
        adj = _build_adjacency(coord_set)
        segs, deg = _graph_segments(adj)
        length_px = float(sum(_segment_length(seg, deg) for seg in segs))

        if length_px <= 0:
            continue

        rows.append((int(gid), area, bool(touches_edge_dna), length_px))
    return rows

def _overlay_text(ax, x, y, txt):
    ax.text(x, y, txt, color='yellow', fontsize=7,
            ha='left', va='top', bbox=dict(facecolor='black', alpha=0.55, pad=2))

def quantify_dna_lengths_bp(
    segmented_folder: str,
    pixel_size_csv: str | None = None,
    nm_per_bp_mean: float | None = None,
    nm_per_bp_sem:  float | None = None,
    output_folder: str | None = None,
    min_component_area_px: int = 50,
    exclude_edge_touching: bool = True,
    overlay: bool = True,
    debug: bool = False,
    loader_kwargs: dict | None = None,
    # NEW: keep parity with batch_curvature_for_folder
    pixel_filename_col: str = "filename",
    pixel_size_col: str = "pixel_size_nm",
):
    """
    Length quantification from annotated masks that ALREADY contain global DNA IDs.

    - Always outputs pixel lengths.
    - If per-image pixel size is available (>0), outputs nm.
    - If nm_per_bp_mean & nm_per_bp_sem are provided, also outputs bp + bp_sem.
    - If a file's pixel size is missing/invalid, we use 1.0 nm/px internally
      BUT do NOT report nm/bp for that file (only px) and flag it.
    """
    if loader_kwargs is None:
        loader_kwargs = {"dilation_radius": 0, "do_skeletonize": False}  # keep IDs exactly as stored

    if output_folder is None:
        output_folder = os.path.join(segmented_folder, "lengths_out")
    os.makedirs(output_folder, exist_ok=True)
    overlay_dir = os.path.join(output_folder, "overlays")
    if overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    # Load pixel-size map (multi-key per filename variant)
    px_map = {}
    if pixel_size_csv:
        try:
            if not os.path.isfile(pixel_size_csv):
                if debug:
                    print(f"[pixel_size_csv] not found: {pixel_size_csv} - will output PX only (nm/bp blank).")
                px_map = {}
            else:
                px_map = _auto_load_pixel_sizes(
                    pixel_csv=pixel_size_csv,
                    pixel_filename_col=pixel_filename_col,
                    pixel_size_col=pixel_size_col,
                )
        except Exception as e:
            if debug:
                print(f"[pixel_size_csv] could not load '{pixel_size_csv}' ({e}) - will output PX only (nm/bp blank).")
            px_map = {}


    allow_bp_globally = (nm_per_bp_mean is not None) and (nm_per_bp_sem is not None)

    tiff_files = [f for f in sorted(os.listdir(segmented_folder)) if f.lower().endswith(".tif")]
    print(f"[segmented_folder] found {len(tiff_files)} TIFFs in {segmented_folder}")

    rows_all = []
    kept_any = False

    for fname in tiff_files:
        fpath = os.path.join(segmented_folder, fname)

        try:
            raw, ann = load_annotated_mask(fpath, **loader_kwargs)
            # ann should be an ID map. If it's binary, label ONCE (per-image IDs).
            if ann.dtype == bool or (np.unique(ann).size <= 2 and set(np.unique(ann).tolist()).issubset({0, 1})):
                if debug:
                    print(f"{fname}: binary mask detected; relabeling 1..N for this image.")
                ann = sk_label(ann, connectivity=2).astype(np.uint32)
        except Exception as e:
            if debug:
                print(f"{fname}: could not load annotated mask ({e}); skipping")
            continue

        # Get per-file nm/px (using many variants)
        px_nm_val = _lookup_px_nm_for_file(fname, px_map)
        has_valid_px = np.isfinite(px_nm_val) and (px_nm_val > 0)

        if not has_valid_px:
            if debug:
                print(f"{fname}: pixel size missing/invalid in CSV; "
                      f"defaulting to 1.0 nm/px internally; will report PX only for this file.")
            px_nm_internal = 1.0
            allow_nm_for_file = False
            allow_bp_for_file = False
        else:
            px_nm_internal = float(px_nm_val)
            allow_nm_for_file = True
            allow_bp_for_file = bool(allow_bp_globally)

        comps = _component_lengths_px_from_idmap(ann, min_component_area_px, exclude_edge_touching)
        if debug:
            print(f"• {fname}: {len(comps)} components kept | "
                  f"px_nm={'NA' if not has_valid_px else px_nm_internal} | "
                  f"allow_nm={allow_nm_for_file}, allow_bp={allow_bp_for_file}")
        if not comps:
            continue
        kept_any = True

        # Prepare overlay
        if overlay:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(raw, cmap='gray')
            ax.axis('off')

        # Compute centroids once
        centroids = {}
        for gid in np.unique(ann):
            if gid == 0:
                continue
            ys, xs = np.nonzero(ann == gid)
            if ys.size > 0:
                centroids[int(gid)] = (float(ys.mean()), float(xs.mean()))

        for gid, area_px, touches_edge_dna, length_px in comps:
            length_nm = (length_px * px_nm_internal) if allow_nm_for_file else np.nan
            if allow_bp_for_file and np.isfinite(length_nm):
                length_bp = length_nm / nm_per_bp_mean
                length_bp_sem = length_bp * (nm_per_bp_sem / nm_per_bp_mean)
            else:
                length_bp = np.nan
                length_bp_sem = np.nan

            # compute a simple 'stem' like before (no lowercasing/normalization beyond a_ + extension)
            stem = os.path.splitext(fname[2:] if fname.startswith('a_') else fname)[0]

            rows_all.append({
                "filename": fname,
                #"stem": stem,
                "comp_id": int(gid),             # global ID preserved
                "area_px": int(area_px),
                "touches_edge_dna": bool(touches_edge_dna),
                "length_px": float(length_px),
                "pixel_size_nm": float(px_nm_val) if has_valid_px else np.nan,
                "length_nm": float(length_nm) if np.isfinite(length_nm) else np.nan,
                "length_bp": float(length_bp) if np.isfinite(length_bp) else np.nan,
                "length_bp_sem": float(length_bp_sem) if np.isfinite(length_bp_sem) else np.nan,
                "had_valid_pixel_size": bool(has_valid_px),
                "bp_calibration_used": bool(allow_bp_for_file),
            })

            if overlay:
                cy, cx = centroids.get(int(gid), (None, None))
                if cy is not None:
                    if np.isfinite(length_bp):
                        label_txt = f"{int(round(length_bp))} bp"
                    elif np.isfinite(length_nm):
                        label_txt = f"{int(round(length_nm))} nm"
                    else:
                        label_txt = f"{int(round(length_px))} px"
                    _overlay_text(ax, x=cx, y=cy, txt=label_txt)

        if overlay:
            out_png = os.path.join(overlay_dir, f"{os.path.splitext(fname)[0]}_overlay.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

    if not kept_any:
        raise RuntimeError("No lengths were measured. Check pixel-size matching, masks, and filters.")

    df = pd.DataFrame(rows_all)
    out_csv = os.path.join(output_folder, "lengths_per_component.csv")
    df.to_csv(out_csv, index=False)

    n_px_only = int((~df["had_valid_pixel_size"]).sum())
    n_nm = int(df["length_nm"].notna().sum())
    n_bp = int(df["length_bp"].notna().sum())
    print(f"[summary] rows={len(df)} | px-only rows={n_px_only} | nm rows={n_nm} | bp rows={n_bp}")
    print(f"Saved: {out_csv}")
    if overlay:
        print(f"Overlays: {overlay_dir}")

    return df, out_csv


# ------------------------------------------------------------
# Radius of gyration in pixel space (no calibration).
# ------------------------------------------------------------
def compute_radius_of_gyration(coords_px: np.ndarray) -> float:
    """
    coords_px: (N, 2) array of (y, x) pixel coordinates where mask==True.
    Returns Rg in pixel units.
    """
    if coords_px.size == 0:
        return np.nan
    center = np.mean(coords_px, axis=0)
    rg_sq = np.mean(np.sum((coords_px - center) ** 2, axis=1))
    return float(np.sqrt(rg_sq))


def compute_normalized_rg_px(region_mask: np.ndarray, total_length_px: float) -> tuple[float, float]:
    """
    Return (rg_px, normalized_rg_px). Normalized by graph length in px.
    If either is invalid, returns (nan, nan).
    """
    ys, xs = np.nonzero(region_mask)
    coords_px = np.column_stack((ys, xs))
    if coords_px.size == 0 or total_length_px <= 0:
        return np.nan, np.nan
    rg_px = compute_radius_of_gyration(coords_px)
    return rg_px, (1-((rg_px*np.sqrt(3)*2) / total_length_px))  # normalized


def _blocked_diagonals(coord_set):
    """Return list of (a,b) diagonals that are disallowed by _is_corner_cut."""
    blocked = []
    for (y0,x0) in coord_set:
        for dy,dx in [(-1,-1),(-1,1),(1,-1),(1,1)]:  # diagonals only
            a = (y0,x0); b = (y0+dy, x0+dx)
            if b in coord_set and _is_corner_cut(a, b, coord_set):
                if a < b:  # store each undirected pair once
                    blocked.append((a,b))
    return blocked

def _segments_to_lines(segments):
    """Convert list of pixel paths to arrays of (x,y) for LineCollection."""
    lines = []
    for path in segments:
        if len(path) < 2: 
            continue
        xy = np.array([(x+0.5, y+0.5) for (y,x) in path], dtype=float)  # pixel centers
        lines.append(xy)
    return lines

def _scatter_deg(ax, deg_map, s=10):
    """Scatter skeleton pixels by degree class (legend included)."""
    pts_ep, pts_int, pts_junc, pts_iso = [], [], [], []
    for (y,x), d in deg_map.items():
        if   d == 1: pts_ep.append((x+0.5, y+0.5))
        elif d == 2: pts_int.append((x+0.5, y+0.5))
        elif d >= 3: pts_junc.append((x+0.5, y+0.5))
        else:        pts_iso.append((x+0.5, y+0.5))
    if pts_int:  ax.scatter(*np.array(pts_int).T,  s=s, label="interior (deg=2)")
    if pts_ep:   ax.scatter(*np.array(pts_ep).T,   s=s, label="endpoint (deg=1)")
    if pts_junc: ax.scatter(*np.array(pts_junc).T, s=s, label="junction (deg≥3)")
    if pts_iso:  ax.scatter(*np.array(pts_iso).T,  s=s, label="isolated (deg=0)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

def debug_plot_region(
    raw, submask, adj, segments,
    branch_centers=None,
    title="debug", savepath=None,
    show_raw=True, show_mask_outline=True, show_degrees=True,
    show_segments=True, show_corner_cuts=True,
    skel=None,  # <-- NEW: precomputed skeleton in crop coords
):
    """Multi-layer debug plot for a single connected component crop."""
    H, W = submask.shape
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # match image y-down display

    if show_raw:
        ax.imshow(raw, cmap='gray', interpolation='nearest')

    if show_mask_outline:
        ax.contour(submask.astype(float), levels=[0.5], linewidths=0.8)

    # degree classes
    if show_degrees:
        deg = _degree_map(adj)
        _scatter_deg(ax, deg, s=8)

    # segments
    if show_segments and segments:
        lines = _segments_to_lines(segments)
        if lines:
            lc = LineCollection(lines, linewidths=1.5)
            lc.set_array(np.arange(len(lines)))  # color by segment index
            ax.add_collection(lc)

    # blocked diagonals
    if show_corner_cuts:
        # use provided skeleton if available; otherwise (rare) compute safely
        if skel is None:
            skel_local = skeletonize(_as_bool_c(submask))
        else:
            skel_local = _as_bool_c(skel)
        ys, xs = np.nonzero(skel_local)
        coord_set = set(zip(ys, xs))
        blocked = _blocked_diagonals(coord_set)
        if blocked:
            blines = []
            for (a,b) in blocked:
                (y0,x0),(y1,x1) = a,b
                blines.append(np.array([(x0+0.5,y0+0.5),(x1+0.5,y1+0.5)], dtype=float))
            lc2 = LineCollection(blines, linewidths=2.0, alpha=0.8)
            ax.add_collection(lc2)
            ax.plot([], [], lw=2.0, label="blocked diag")
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    # branch centers
    if branch_centers:
        xs = [x+0.5 for (y,x) in branch_centers]
        ys = [y+0.5 for (y,x) in branch_centers]
        ax.scatter(xs, ys, s=40, facecolors='none', edgecolors='yellow', linewidths=1.2, label="branch ctrs")
        ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def _degree_map(adj):
    return {v: len(ns) for v, ns in adj.items()}


def count_branchpoint_clusters_by_conv(region_mask: np.ndarray,
                                       cluster_eps: float = 2.0,
                                       skel_pre: np.ndarray | None = None):
    """
    If skel_pre is provided, use it; otherwise skeletonize region_mask (safely).
    """
    if skel_pre is None:
        skel = skeletonize(_as_bool_c(region_mask))
    else:
        skel = _as_bool_c(skel_pre)

    if not np.any(skel):
        return 0, []

    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=int)
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    branch_mask = skel & (neighbor_count > 2)

    ys, xs = np.nonzero(branch_mask)
    if ys.size == 0:
        return 0, []
    coords = np.column_stack((ys, xs))
    clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(coords)
    centers = []
    for lbl in np.unique(clustering.labels_):
        pts = coords[clustering.labels_ == lbl]
        cy, cx = pts.mean(axis=0).astype(int)
        centers.append((int(cy), int(cx)))
    return len(centers), centers


def tortuosity_simple_from_skeleton(skel: np.ndarray, total_length_px: float) -> float:
    """
    Tortuosity = total_length_px / chord, where chord is the maximum Euclidean
    distance between any two endpoints (deg==1) on the skeleton.

    - Returns NaN if there are fewer than 2 endpoints (e.g., pure loop),
      or if inputs are invalid.
    """
    if not np.isfinite(total_length_px) or total_length_px <= 0:
        return np.nan
    if skel is None:
        return np.nan

    sk = _as_bool_c(skel)
    if not sk.any():
        return np.nan

    H, W = sk.shape
    ys, xs = np.nonzero(sk)

    # collect endpoints (8-connected degree == 1)
    endpoints = []
    for y, x in zip(ys, xs):
        deg = 0
        for yy, xx in _neighbors8(y, x, H, W):
            if sk[yy, xx]:
                deg += 1
        if deg == 1:
            endpoints.append((y, x))

    # need at least two ends; otherwise (loop) tortuosity is undefined
    if len(endpoints) < 2:
        return np.nan

    # farthest-apart endpoints
    max_d2 = 0.0
    for i in range(len(endpoints) - 1):
        y0, x0 = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            y1, x1 = endpoints[j]
            d2 = (y1 - y0) ** 2 + (x1 - x0) ** 2
            if d2 > max_d2:
                max_d2 = d2

    chord = float(np.sqrt(max_d2))
    if chord <= 0.0:
        return np.nan

    return float(total_length_px) / chord


def _as_bool_c(arr):
    """Bool array, C-contiguous (what skimage.skeletonize expects)."""
    return np.ascontiguousarray(arr.astype(bool, copy=False))


def _looks_skeletonized(mask_bool: np.ndarray) -> bool:
    """
    Heuristic: return True if there is no fully-filled 2 times 2 block.
    If any 2 times 2 block is all ones, it's probably not 1-px wide yet.
    """
    k = np.array([[1,1],[1,1]], dtype=np.uint8)
    s = convolve(mask_bool.astype(np.uint8, copy=False), k, mode='constant', cval=0)
    return not np.any(s == 4)

def _in_bounds(y, x, H, W):
    return (0 <= y < H) and (0 <= x < W)

def _neighbors8(y, x, H, W):
    for dy, dx in _OFFS8:
        yy, xx = y+dy, x+dx
        if _in_bounds(yy, xx, H, W):
            yield yy, xx

def _degree(y, x, skel):
    H, W = skel.shape
    return sum(1 for (yy, xx) in _neighbors8(y, x, H, W) if skel[yy, xx])

def _k_dir(vecs, k=3):
    """Direction from last k steps of a (y,x) sequence."""
    if len(vecs) < 2:
        return np.array([0.0, 0.0])
    k = min(k, len(vecs)-1)
    y0, x0 = vecs[-k-1]
    y1, x1 = vecs[-1]
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _cluster_overlaps(junc_mask, radius=3):
    """
    Merge junction pixels (deg>=3) within 'radius' using dilation + CC.
    Returns labeled hubs (1..K) only where original junction pixels are true.
    """
    if not junc_mask.any():
        return np.zeros_like(junc_mask, dtype=np.int32), 0
    dil = dilation(junc_mask, disk(int(radius)))
    lab = sk_label(dil, connectivity=2).astype(np.int32)
    out = np.zeros_like(lab, dtype=np.int32)
    out[junc_mask] = lab[junc_mask]
    # reindex to 1..K
    u = np.unique(out[out > 0])
    remap = {old:i+1 for i, old in enumerate(u)}
    for old, new in remap.items():
        out[out == old] = new
    return out, len(u)

def _build_node_maps(skel, overlap_radius=3):
    """
    Identify endpoints (deg==1) and merged hubs (deg>=3 within radius).
    Returns node_id_map (0 bg, 1..M nodes) and node list.
    """
    H, W = skel.shape
    ys, xs = np.where(skel)
    deg = np.zeros_like(skel, dtype=np.uint8)
    for y, x in zip(ys, xs):
        deg[y, x] = _degree(y, x, skel)

    end_mask  = (skel & (deg == 1))
    junc_mask = (skel & (deg >= 3))

    hub_lab, K = _cluster_overlaps(junc_mask, radius=overlap_radius)

    node_id_map = np.zeros_like(skel, dtype=np.int32)
    nodes = []

    # hubs 1..K
    for k in range(1, K+1):
        pix = np.column_stack(np.where(hub_lab == k))
        for y, x in pix:
            node_id_map[y, x] = k
        nodes.append({'type':'hub', 'pixels':[tuple(p) for p in pix]})

    # endpoints K+1...
    eys, exs = np.where(end_mask)
    for i, (y, x) in enumerate(zip(eys, exs), start=K+1):
        node_id_map[y, x] = i
        nodes.append({'type':'end', 'pixels':[(int(y), int(x))]})

    return node_id_map, nodes


def _choose_straightest(next_edges, prev_dir, lookahead=3):
    best, best_score = None, -np.inf
    for e in next_edges:
        cand_dir = _k_dir(e['pix'][:lookahead], k=min(lookahead, len(e['pix']))) if e['pix'] else np.array([0.0,0.0])
        score = float(len(e['pix'])) if np.linalg.norm(prev_dir) < 1e-8 else float(np.dot(prev_dir, cand_dir))
        if score > best_score:
            best, best_score = e, score
    return best



def _trace_edge_from(node_id, start_pix, skel, node_id_map):
    H, W = skel.shape
    curr = start_pix
    node_nbrs = [(yy, xx) for (yy, xx) in _neighbors8(curr[0], curr[1], H, W) if node_id_map[yy, xx] > 0]
    prev = node_nbrs[0] if node_nbrs else curr
    path = [curr]
    while True:
        if node_id_map[curr] > 0:
            end_node = node_id_map[curr]
            return end_node, path[:-1]
        nbrs = [(yy, xx) for (yy, xx) in _neighbors8(curr[0], curr[1], H, W) if skel[yy, xx]]
        nbrs = [p for p in nbrs if p != prev]
        if len(nbrs) == 0:
            return -1, path
        if len(nbrs) > 1:
            node_next = [p for p in nbrs if node_id_map[p] > 0]
            if node_next:
                end_node = node_id_map[node_next[0]]
                return end_node, path
            return -1, path
        nxt = nbrs[0]
        path.append(nxt)
        prev, curr = curr, nxt


def _build_graph_with_edge_ids(skel, node_id_map):
    """
    Like _build_graph(), but each undirected edge gets a unique edge_id
    shared by the forward and reverse entries.
    """
    H, W = skel.shape
    edges = defaultdict(list)
    visited_starts = set()
    next_edge_id = 0

    node_ids = np.unique(node_id_map[node_id_map > 0])
    for nid in node_ids:
        node_pixels = np.column_stack(np.where(node_id_map == nid))
        boundary_steps = set()
        for y, x in node_pixels:
            for yy, xx in _neighbors8(y, x, H, W):
                if skel[yy, xx] and node_id_map[yy, xx] == 0:
                    boundary_steps.add((yy, xx))

        for step in boundary_steps:
            key = (nid, step)
            if key in visited_starts:
                continue
            end_node, pix = _trace_edge_from(nid, step, skel, node_id_map)
            if len(pix) == 0 or end_node <= 0:
                continue
            # assign an undirected edge_id; mark both directions with same id
            eid = next_edge_id
            next_edge_id += 1
            edges[nid].append({'to': end_node, 'pix': pix, 'edge_id': eid})
            edges[end_node].append({'to': nid, 'pix': pix[::-1], 'edge_id': eid})
            visited_starts.add((nid, step))
            visited_starts.add((end_node, pix[-1]))
    return edges


def _emit_all_segments(edges, node_id_map, lookahead=3):
    """
    Cover all edges by emitting maximal straightest paths; returns list of (y,x) sequences.
    """
    used_edge_ids = set()
    remaining = {nid: list(adj) for nid, adj in edges.items()}

    def pop_next_edge(nid, prev_dir):
        cand = [e for e in remaining.get(nid, []) if e['edge_id'] not in used_edge_ids]
        if not cand:
            return None
        return _choose_straightest(cand, prev_dir, lookahead=lookahead)

    # classify node degree in the graph
    node_types = {}
    ids = np.unique(node_id_map[node_id_map > 0])
    for nid in ids:
        deg = sum(1 for _ in edges.get(nid, []))
        node_types[nid] = 'end' if deg == 1 else 'hub'

    def start_nodes():
        ends = [nid for nid,t in node_types.items() if t=='end' and any(e['edge_id'] not in used_edge_ids for e in edges.get(nid, []))]
        hubs = [nid for nid,t in node_types.items() if t=='hub' and any(e['edge_id'] not in used_edge_ids for e in edges.get(nid, []))]
        return ends + hubs

    segments = []
    while True:
        starts = start_nodes()
        if not starts:
            break
        curr = starts[0]
        prev_dir = np.array([0.0, 0.0])
        seg = []

        e = pop_next_edge(curr, prev_dir)
        if e is None:
            remaining[curr] = []
            continue

        while e is not None:
            seg.extend(e['pix'])
            used_edge_ids.add(e['edge_id'])
            prev_dir = _k_dir(e['pix'][-lookahead:], k=min(lookahead, len(e['pix'])))
            curr = e['to']
            e = pop_next_edge(curr, prev_dir)

        if len(seg) > 0:
            segments.append(seg)

    return segments

def extract_all_segments_from_annotation(annotation_mask,
                                         overlap_radius=3,
                                         lookahead=3,
                                         do_skeletonize=True):
    """
    Returns ALL path segments that cover every edge of every labeled DNA, even if
    a label breaks into multiple disconnected bits or has >2 ends.
    -------
    Output:
      segments_by_gid : dict[gid] -> list[list[(y,x), ...]]  # multiple segments per gid
      debug : dict[gid] -> {'skel': ..., 'node_id_map': ..., 'edges': ...}
    """
    segments_by_gid = {}
    debug = {}
    gids = [g for g in np.unique(annotation_mask) if g != 0]

    for gid in gids:
        mask = (annotation_mask == gid)
        if not mask.any():
            segments_by_gid[gid] = []
            continue

        skel_full = skeletonize(mask) if do_skeletonize else mask.astype(bool)
        # Process each connected component of the skeleton separately
        comp_lab = sk_label(skel_full, connectivity=2)
        segs = []
        all_edges = {}
        all_node_maps = []

        for cid in np.unique(comp_lab[comp_lab > 0]):
            skel = (comp_lab == cid)
            if not skel.any():
                continue

            # Build nodes; if none exist, this is a pure loop (deg==2 everywhere)
            node_id_map, _nodes = _build_node_maps(skel, overlap_radius=overlap_radius)
            has_nodes = np.any(node_id_map > 0)

            if not has_nodes:
                # --- Pure loop fallback: walk the cycle directly on the skeleton ---
                H, W = skel.shape
                ys, xs = np.where(skel)
                if ys.size >= 2:
                    start = (int(ys[0]), int(xs[0]))
                    # first neighbor
                    nbrs0 = [(yy, xx) for (yy, xx) in _neighbors8(start[0], start[1], H, W) if skel[yy, xx]]
                    if nbrs0:
                        prev = start
                        curr = nbrs0[0]
                        cyc = [start, curr]
                        steps = 0
                        max_steps = H * W + 10  # safety
                        while steps < max_steps:
                            # forward neighbor (exclude where we came from)
                            nbrs = [(yy, xx) for (yy, xx) in _neighbors8(curr[0], curr[1], H, W)
                                    if skel[yy, xx] and (yy, xx) != prev]
                            if not nbrs:
                                break
                            nxt = nbrs[0]  # deg==2 in a loop, so one forward neighbor
                            cyc.append(nxt)
                            if nxt == start:
                                break
                            prev, curr = curr, nxt
                            steps += 1
                        if len(cyc) >= 2:
                            segs.append(cyc)

                # Debug placeholders for this component
                all_node_maps.append(np.zeros_like(skel, dtype=np.int32))
                all_edges[cid] = {}
                continue

            # --- Normal branch: graph-based segmentation covering all edges ---
            edges = _build_graph_with_edge_ids(skel, node_id_map)
            segments = _emit_all_segments(edges, node_id_map, lookahead=lookahead)
            segs.extend(segments)

            # stash debug per component (optional)
            all_node_maps.append(node_id_map)
            all_edges[cid] = edges


        segments_by_gid[gid] = segs
        debug[gid] = {'skel': skel_full, 'node_id_maps': all_node_maps, 'edges': all_edges}

    return segments_by_gid, debug


def _polyline_arclength(xs, ys):
    dx = np.diff(xs); dy = np.diff(ys)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s, ds


def curvature_of_path(seq,
                      spacing=1.0,
                      pre_smooth_window=7,
                      pre_smooth_poly=2,
                      trim_frac=0.02):
    """
    Compute curvature along one ordered pixel path [(y,x), ...] using:
    1) optional Savitzky-Golay pre-smoothing of x,y
    2) arclength parameter s
    3) CubicSpline fits x(s), y(s)
    4) curvature kappa(s) via derivatives
    5) NEW: tangent angle theta(s)=atan2(y'(s), x'(s)) and persistence length estimate

    Returns:
    mean_kappa        : float (mean |kappa| after trimming)
    ...
    lp_px             : float, persistence length in pixels (<ds> / <dtheta^2>)
    theta_trim        : (M-2t,) array, unwrapped tangent angles on s_trim (radians)
    Notes:
    - dtheta are adjacent differences on the trimmed grid, theta is unwrapped before differencing.
    """

    if seq is None or len(seq) < 4:
        return np.nan, np.array([]), np.array([]), np.array([]), np.nan, np.array([])

    arr = np.asarray(seq, dtype=float)
    ys = arr[:, 0]
    xs = arr[:, 1]

    # remove exact duplicates that break arclength
    keep = np.ones(len(xs), dtype=bool)
    keep[1:] = (np.diff(xs) != 0) | (np.diff(ys) != 0)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 4:
        return np.nan, np.array([]), np.array([]), np.array([]), np.nan, np.array([])

    # light pre-smoothing (optional)
    if pre_smooth_window >= 3 and pre_smooth_window % 2 == 1 and len(xs) >= pre_smooth_window:
        xs_s = savgol_filter(xs, pre_smooth_window, pre_smooth_poly, mode='interp')
        ys_s = savgol_filter(ys, pre_smooth_window, pre_smooth_poly, mode='interp')
    else:
        raise AssertionError("pre_smooth_window must be odd and >=3")

    # resample by arclength to reduce pixel anisotropy
    xr, yr, s = _resample_by_arclength(xs_s, ys_s, spacing=spacing)
    if len(xr) < 4:
        return np.nan, np.array([]), np.array([]), np.array([]), np.nan, np.array([])

    # fit interpolating cubic splines (end bc: 'natural' is robust)
    csx = CubicSpline(s, xr, bc_type='natural')
    csy = CubicSpline(s, yr, bc_type='natural')

    # curvature on full s, then trim internally in helper
    kappa_full = _curvature_from_splines(s, csx, csy, trim_frac=trim_frac)
    mean_kappa = float(np.nanmean(kappa_full)) if kappa_full.size else np.nan

    # compute the same trim indices used inside _curvature_from_splines
    m = len(s)
    t = int(max(1, np.floor(m * trim_frac)))
    if m > 2 * t:
        s_trim = s[t: m - t]
    else:
        s_trim = s

    # Evaluate coordinates on s_trim (for plotting)
    xr_t = csx(s_trim)
    yr_t = csy(s_trim)
    coords_trim_yx = np.column_stack((yr_t, xr_t))

    # Tangent angle θ(s) = atan2(y'(s), x'(s)) on trimmed grid
    x1_t = csx(s_trim, 1)
    y1_t = csy(s_trim, 1)
    theta_trim = np.arctan2(y1_t, x1_t)

    # Unwrap θ to avoid π-jumps, then compute adjacent Δθ and Δs
    theta_u = np.unwrap(theta_trim)
    if theta_u.size >= 2 and s_trim.size >= 2:
        dtheta = np.diff(theta_u)
        ds     = np.diff(s_trim)
        # filter finite and strictly positive ds
        ok = np.isfinite(dtheta) & np.isfinite(ds) & (ds > 0)
        dtheta2 = dtheta[ok] ** 2
        ds_ok   = ds[ok]
        if dtheta2.size > 0 and np.nanmean(dtheta2) > 0:
            lp_px = float(np.nanmean(ds_ok) / np.nanmean(dtheta2))
        else:
            lp_px = np.nan
    else:
        lp_px = np.nan

    # Return kappa trimmed (already trimmed inside helper)
    kappa_trim = kappa_full
    return mean_kappa, kappa_trim, s_trim, coords_trim_yx, lp_px, theta_trim

#!!!

def _resample_by_arclength(xs, ys, spacing=1.0):
    """Linear resample to ~uniform arclength spacing in pixels."""
    s, _ = _polyline_arclength(xs, ys)
    if s[-1] <= 0:
        return xs.copy(), ys.copy(), s.copy()
    n = max(3, int(np.floor(s[-1] / max(spacing, 1e-6))) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    x_new = np.interp(s_new, s, xs)
    y_new = np.interp(s_new, s, ys)
    return x_new, y_new, s_new


def _curvature_from_splines(s_eval, csx, csy, trim_frac=0.02):
    """kappa(s) = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2); trim small ends to reduce boundary artifacts."""
    x1 = csx(s_eval, 1); x2 = csx(s_eval, 2)
    y1 = csy(s_eval, 1); y2 = csy(s_eval, 2)
    num = np.abs(x1 * y2 - y1 * x2)
    den = np.power(x1*x1 + y1*y1 + 1e-12, 1.5)
    kappa = num / den
    m = len(kappa)
    t = int(max(1, np.floor(m * trim_frac)))
    return kappa[t: m - t] if m > 2*t else kappa



def _auto_load_pixel_sizes(pixel_csv, pixel_filename_col='filename', pixel_size_col='pixel_size_nm'):
    """
    Load per-image pixel size (nm/px) with robust header detection and filename
    normalization. Returns dict mapping multiple filename variants -> nm_per_pixel.
    Variants include: original basename, with/without leading 'a_', with/without extension.
    """
    px_map = {}
    if pixel_csv is None:
        return px_map

    df_px = pd.read_csv(pixel_csv)

    def _norm(name):
        return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())

    cols_norm = {_norm(c): c for c in df_px.columns}

    def _pick_col(user_name, kind):
        # 1) honor explicit user arg if present
        if user_name in df_px.columns:
            return user_name
        if _norm(user_name) in cols_norm:
            return cols_norm[_norm(user_name)]

        # 2) auto-detect
        if kind == 'filename':
            aliases = ['filename','file','fname','image','img','path','basename']
            for k, orig in cols_norm.items():
                if any(tok in k for tok in aliases):
                    return orig
        elif kind == 'pixelsize':
            # common pixel-size header patterns (nm per pixel)
            pixel_aliases = [
                'pixelsize', 'pixel_size', 'pixel_size_nm', 'nmperpixel', 'nm_per_pixel',
                'nmpx', 'nm_px', 'nmperpx', 'pixelwidthnm', 'pixelheightnm', 'pixelsiz (nm)'
            ]
            for alias in pixel_aliases:
                if alias in cols_norm:
                    return cols_norm[alias]
            # generic heuristic: any col containing 'nm' AND ('pixel'|'px'|'pix')
            for k, orig in cols_norm.items():
                if 'nm' in k and ('pixel' in k or 'px' in k or 'pix' in k):
                    return orig
            # last resort: a single numeric-ish column besides filename
            numeric_like = []
            for col in df_px.columns:
                if col == pixel_filename_col:
                    continue
                try:
                    pd.to_numeric(df_px[col].astype(str).str.replace(',', '.'), errors='coerce')
                    numeric_like.append(col)
                except Exception:
                    pass
            if len(numeric_like) == 1:
                return numeric_like[0]

        raise KeyError(f"Could not auto-detect '{kind}' column. Found: {list(df_px.columns)}")

    fname_col = _pick_col(pixel_filename_col, 'filename')
    psize_col = _pick_col(pixel_size_col,   'pixelsize')

    def _variants(basename):
        """Return reasonable key variants for mapping (including with/without leading 'a_')."""
        base = os.path.basename(basename)
        root, ext = os.path.splitext(base)

        def _strip_a(s: str) -> str:
            return s[2:] if s.startswith('a_') else s

        # canonical stripped forms
        base0 = _strip_a(base)
        root0 = _strip_a(root)

        alts = set()

        # always include original + root
        alts.add(base); alts.add(root)

        # always include stripped versions
        alts.add(base0); alts.add(root0)

        # always include prefixed versions (of the stripped forms)
        alts.add('a_' + base0)
        alts.add('a_' + root0)

        return alts


    for _, row in df_px.iterrows():
        raw_fname = str(row[fname_col]).strip()
        raw_val = str(row[psize_col]).strip().replace(',', '.')  # accept "4,0"
        try:
            nm_per_px = float(raw_val)
        except Exception:
            nm_per_px = np.nan
        for key in _variants(raw_fname):
            px_map[key] = nm_per_px

    return px_map

# --- helpers ---------------------------------------------------------------

def _sg_window_odd_ge3(win: int) -> int:
    """Force odd and >=3."""
    w = int(max(3, win))
    if w % 2 == 0:
        w += 1  # bump to odd
    return w

def _sg_poly_lt_window(poly: int, window: int) -> int:
    """Ensure 0 <= poly < window."""
    p = int(max(0, poly))
    if p >= window:
        p = max(0, window - 1)
    return p

def _curvature_of_path_safe(
    seq,
    spacing,
    pre_smooth_window,
    pre_smooth_poly,
    trim_frac
):
    """
    Call curvature_of_path with robust SavGol params.
    Retries with decreasing odd windows if needed for short segments.

    NEW: bubbles through lp_px and theta as additional return values.
    """
    # Start from sanitized global suggestion
    w0 = _sg_window_odd_ge3(pre_smooth_window)
    p0 = _sg_poly_lt_window(pre_smooth_poly, w0)

    # Try a small ladder of windows: w0, w0-2, w0-4, ..., down to 3
    last_err = None
    w = w0
    while w >= 3:
        if w % 2 == 0:
            w -= 1
            continue
        p = _sg_poly_lt_window(p0, w)
        try:
            return curvature_of_path(
                seq,
                spacing=spacing,
                pre_smooth_window=w,
                pre_smooth_poly=p,
                trim_frac=trim_frac
            )
        except Exception as e:
            last_err = e
            w -= 2  # next smaller odd window

    # Give one last try with the minimal valid window=3, poly=2 or 1
    try:
        w = 3
        p = min(2, _sg_poly_lt_window(p0, w))
        return curvature_of_path(
            seq,
            spacing=spacing,
            pre_smooth_window=w,
            pre_smooth_poly=p,
            trim_frac=trim_frac
        )
    except Exception as e2:
        raise RuntimeError(f"curvature_of_path failed after retries (last={last_err}) and final={e2}")


def _count_strong_bends_from_kappa(
    s_eval: np.ndarray,
    kappa: np.ndarray,
    angle_threshold_deg: float = 10.0,
    min_span_px: float = 3.0,
    gap_tolerance_px: float = 1.0,
    keep_frac: float = 0.9
) -> int:
    if kappa is None or s_eval is None or len(kappa) < 2 or len(s_eval) < 2:
        return 0

    ds = np.diff(s_eval)
    L = min(len(kappa), len(ds))
    if L <= 0:
        return 0

    kap = np.abs(kappa[:L])
    ds  = ds[:L]

    kappa_thr = np.deg2rad(float(angle_threshold_deg)) / float(min_span_px)
    keep_thr = keep_frac * kappa_thr
    allow_gap = float(gap_tolerance_px)

    count = 0
    i = 0
    while i < L:
        if kap[i] < kappa_thr:
            i += 1
            continue

        span = 0.0
        gap  = 0.0
        j = i
        while j < L:
            kj = kap[j]
            if kj >= keep_thr:
                span += ds[j]
                gap = 0.0
            else:
                gap += ds[j]
                if gap > allow_gap:
                    break
            j += 1

        if span >= float(min_span_px):
            count += 1
        i = j

    return count

def batch_curvature_for_folder(
    folder,
    pixel_csv=None,
    pixel_filename_col='filename',
    pixel_size_col='pixel_size_nm',
    output_csv='curvature_stats.csv',
    # --- path extraction params ---
    dilation_radius=0,
    overlap_radius=3,
    lookahead=3,
    do_skeletonize=True,
    # --- curvature params ---
    spacing=1.0,
    pre_smooth_window=15,
    pre_smooth_poly=3,
    trim_frac=0.02,
    # --- strong-bend params (angle+span API) ---
    bend_angle_deg=60.0,
    bend_min_span_px=5.0,      # pixel-space threshold (kept as-is)
    bend_span_nm_ref=10.0,     # NEW: physical span for nm-based bend count
    # misc
    recursive=False,
    image_exts=('.tif', '.tiff', '.TIF', '.TIFF'),
):
    """
    For each image in `folder`, compute curvature stats, persistence length (lp),
    and strong-bend counts (both pixel- and nm-based). Returns a DataFrame and
    optionally saves to CSV if output_csv is set.

    New columns (vs. original):
      - no_strong_bends_px : count using the raw pixel-space span
      - no_strong_bends_nm : count using a fixed physical span (10 nm by default)
      - lp_px, lp_nm, n_theta_increments (persistence length)
    """
    # sanitize SavGol
    pre_smooth_window = _sg_window_odd_ge3(pre_smooth_window)
    pre_smooth_poly   = _sg_poly_lt_window(pre_smooth_poly, pre_smooth_window)

    # pixel size map (nm/px)
    px_map = _auto_load_pixel_sizes(pixel_csv, pixel_filename_col, pixel_size_col) if pixel_csv else {}

    # gather files
    files = []
    if recursive:
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                if f.endswith(image_exts):
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.endswith(image_exts):
                files.append(os.path.join(folder, f))
    files.sort()

    rows = []
    missing_px = set()

    for path in files:
        fname = os.path.basename(path)

        # pixel size lookup
        nm_per_px = px_map.get(fname, np.nan)
        if not np.isfinite(nm_per_px):
            root_noext, _ = os.path.splitext(fname)
            nm_per_px = px_map.get(root_noext, nm_per_px)
            if not np.isfinite(nm_per_px) and fname.startswith('a_'):
                nm_per_px = px_map.get(fname[2:], nm_per_px)
                nm_per_px = px_map.get(os.path.splitext(fname[2:])[0], nm_per_px)
            if not np.isfinite(nm_per_px) and not fname.startswith('a_'):
                nm_per_px = px_map.get('a_' + fname, nm_per_px)
                nm_per_px = px_map.get('a_' + root_noext, nm_per_px)
            if not np.isfinite(nm_per_px):
                missing_px.add(fname)

        # load annotation
        try:
            _, ann = load_annotated_mask(path, dilation_radius=dilation_radius, do_skeletonize=False)
        except Exception as e:
            rows.append({
                'filename': fname,
                'gid': None,
                'pixel_size_nm': nm_per_px,
                'mean_kappa_px_inv': np.nan,
                'std_kappa_px_inv': np.nan,
                'min_kappa_px_inv': np.nan,
                'max_kappa_px_inv': np.nan,
                'mean_kappa_nm_inv': np.nan,
                'std_kappa_nm_inv': np.nan,
                'min_kappa_nm_inv': np.nan,
                'max_kappa_nm_inv': np.nan,
                'n_eval': 0,
                'no_strong_bends_px': 0,
                'no_strong_bends_nm': np.nan,
                'lp_px': np.nan,
                'lp_nm': np.nan,
                'n_theta_increments': 0,
                'error': f'load_annotated_mask failed: {e}',
            })
            continue

        # relabel if binary
        try:
            u = np.unique(ann)
            if ann.dtype == bool or (u.size <= 2 and set(u.tolist()).issubset({0, 1})):
                from skimage.measure import label as _sk_label
                ann = _sk_label(ann.astype(bool), connectivity=2).astype(np.uint32)
        except Exception:
            pass

        # gids
        try:
            gids_in_ann = [int(g) for g in np.unique(ann) if int(g) != 0]
        except Exception:
            gids_in_ann = []

        # segments for all gids
        try:
            segments_by_gid, _dbg = extract_all_segments_from_annotation(
                ann,
                overlap_radius=overlap_radius,
                lookahead=lookahead,
                do_skeletonize=do_skeletonize
            )
        except Exception as e:
            for gid in gids_in_ann:
                rows.append({
                    'filename': fname,
                    'gid': int(gid),
                    'pixel_size_nm': nm_per_px,
                    'mean_kappa_px_inv': np.nan,
                    'std_kappa_px_inv': np.nan,
                    'min_kappa_px_inv': np.nan,
                    'max_kappa_px_inv': np.nan,
                    'mean_kappa_nm_inv': np.nan,
                    'std_kappa_nm_inv': np.nan,
                    'min_kappa_nm_inv': np.nan,
                    'max_kappa_nm_inv': np.nan,
                    'n_eval': 0,
                    'no_strong_bends_px': 0,
                    'no_strong_bends_nm': np.nan,
                    'lp_px': np.nan,
                    'lp_nm': np.nan,
                    'n_theta_increments': 0,
                    'error': f'extract_all_segments_from_annotation failed: {e}',
                })
            continue

        # per-gid stats
        for gid in gids_in_ann:
            segs = segments_by_gid.get(gid, [])
            kappa_px_all = []
            seg_errors = []

            # persistence length accumulators
            all_dtheta2 = []
            all_ds = []
            n_theta_increments = 0

            bend_count_px = 0
            bend_count_nm = np.nan

            if not segs:
                rows.append({
                    'filename': fname,
                    'gid': int(gid),
                    'pixel_size_nm': nm_per_px,
                    'mean_kappa_px_inv': np.nan,
                    'std_kappa_px_inv': np.nan,
                    'min_kappa_px_inv': np.nan,
                    'max_kappa_px_inv': np.nan,
                    'mean_kappa_nm_inv': np.nan,
                    'std_kappa_nm_inv': np.nan,
                    'min_kappa_nm_inv': np.nan,
                    'max_kappa_nm_inv': np.nan,
                    'n_eval': 0,
                    'no_strong_bends_px': 0,
                    'no_strong_bends_nm': np.nan,
                    'lp_px': np.nan,
                    'lp_nm': np.nan,
                    'n_theta_increments': 0,
                    'error': 'no segments extracted for gid',
                })
                continue

            for seq in segs:
                if len(seq) < 4:
                    continue
                try:
                    mean_kappa_seg, kappa, s_eval, _, lp_px_seg, theta_trim = _curvature_of_path_safe(
                        seq=seq,
                        spacing=spacing,
                        pre_smooth_window=pre_smooth_window,
                        pre_smooth_poly=pre_smooth_poly,
                        trim_frac=trim_frac
                    )
                except Exception as e:
                    seg_errors.append(f'curvature_of_path failed: {e}')
                    continue

                if isinstance(kappa, np.ndarray) and kappa.size:
                    kappa_px_all.append(kappa)

                # persistence-length accumulators
                if isinstance(theta_trim, np.ndarray) and theta_trim.size >= 2 and isinstance(s_eval, np.ndarray) and s_eval.size >= 2:
                    th_u = np.unwrap(theta_trim)
                    dtheta = np.diff(th_u)
                    ds = np.diff(s_eval)
                    ok = np.isfinite(dtheta) & np.isfinite(ds) & (ds > 0)
                    if ok.any():
                        all_dtheta2.append((dtheta[ok] ** 2))
                        all_ds.append(ds[ok])
                        n_theta_increments += int(np.sum(ok))

                # strong-bends (px)
                if isinstance(kappa, np.ndarray) and kappa.size and s_eval is not None:
                    try:
                        bend_count_px += _count_strong_bends_from_kappa(
                            s_eval=s_eval,
                            kappa=kappa,
                            angle_threshold_deg=bend_angle_deg,
                            min_span_px=bend_min_span_px,
                            gap_tolerance_px=1.0,
                            keep_frac=0.9
                        )
                    except Exception as e:
                        seg_errors.append(f'bend_count_px failed: {e}')

                # strong-bends (nm span -> px span)
                if np.isfinite(nm_per_px) and nm_per_px > 0 and isinstance(kappa, np.ndarray) and kappa.size and s_eval is not None:
                    try:
                        span_px_for_10nm = float(bend_span_nm_ref) / float(nm_per_px)
                        bend_count_nm_i = _count_strong_bends_from_kappa(
                            s_eval=s_eval,
                            kappa=kappa,
                            angle_threshold_deg=bend_angle_deg,
                            min_span_px=span_px_for_10nm,
                            gap_tolerance_px=1.0,
                            keep_frac=0.9
                        )
                        bend_count_nm = bend_count_nm_i if np.isnan(bend_count_nm) else bend_count_nm + bend_count_nm_i
                    except Exception as e:
                        seg_errors.append(f'bend_count_nm failed: {e}')

            # summarize curvature
            if not kappa_px_all:
                rows.append({
                    'filename': fname,
                    'gid': int(gid),
                    'pixel_size_nm': nm_per_px,
                    'mean_kappa_px_inv': np.nan,
                    'std_kappa_px_inv': np.nan,
                    'min_kappa_px_inv': np.nan,
                    'max_kappa_px_inv': np.nan,
                    'mean_kappa_nm_inv': np.nan,
                    'std_kappa_nm_inv': np.nan,
                    'min_kappa_nm_inv': np.nan,
                    'max_kappa_nm_inv': np.nan,
                    'n_eval': 0,
                    'no_strong_bends_px': int(bend_count_px),
                    'no_strong_bends_nm': bend_count_nm,
                    'lp_px': np.nan,
                    'lp_nm': np.nan,
                    'n_theta_increments': int(n_theta_increments),
                    'error': '; '.join(list(dict.fromkeys(seg_errors))[:3]),
                })
                continue

            kcat = np.concatenate(kappa_px_all).astype(float)
            mean_px = float(np.nanmean(kcat))
            std_px  = float(np.nanstd(kcat))
            min_px  = float(np.nanmin(kcat))
            max_px  = float(np.nanmax(kcat))

            if np.isfinite(nm_per_px) and nm_per_px > 0:
                scale = 1.0 / nm_per_px
                mean_nm = mean_px * scale
                std_nm  = std_px  * scale
                min_nm  = min_px  * scale
                max_nm  = max_px  * scale
            else:
                mean_nm = std_nm = min_nm = max_nm = np.nan

            # persistence length (ensemble)
            if all_dtheta2 and all_ds:
                dtheta2_cat = np.concatenate(all_dtheta2)
                ds_cat      = np.concatenate(all_ds)
                ok = np.isfinite(dtheta2_cat) & np.isfinite(ds_cat) & (ds_cat > 0)
                if ok.any() and np.nanmean(dtheta2_cat[ok]) > 0:
                    lp_px = float(np.nanmean(ds_cat[ok]) / np.nanmean(dtheta2_cat[ok]))
                else:
                    lp_px = np.nan
            else:
                lp_px = np.nan
            lp_nm = (lp_px * nm_per_px) if (np.isfinite(lp_px) and np.isfinite(nm_per_px) and nm_per_px > 0) else np.nan

            rows.append({
                'filename': fname,
                'gid': int(gid),
                'pixel_size_nm': nm_per_px,
                'mean_kappa_px_inv': mean_px,
                'std_kappa_px_inv': std_px,
                'min_kappa_px_inv': min_px,
                'max_kappa_px_inv': max_px,
                'mean_kappa_nm_inv': mean_nm,
                'std_kappa_nm_inv': std_nm,
                'min_kappa_nm_inv': min_nm,
                'max_kappa_nm_inv': max_nm,
                'n_eval': int(kcat.size),
                'no_strong_bends_px': int(bend_count_px),
                'no_strong_bends_nm': bend_count_nm,
                'lp_px': lp_px,
                'lp_nm': lp_nm,
                'n_theta_increments': int(n_theta_increments),
                'error': '; '.join(list(dict.fromkeys(seg_errors))[:3]),
            })

    out_df = pd.DataFrame(rows).sort_values(['filename','gid'], na_position='last').reset_index(drop=True)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    if missing_px:
        print(f" No pixel size found for {len(missing_px)} file(s), e.g.: {sorted(list(missing_px))[:5]}")

    return out_df


def analyze_rg_branch_shape(
    folder: str,
    output_folder: str,
    dilation_radius: int = 0,
    do_skeletonize: bool = False,
    loader_kwargs: dict | None = None,
    input_is_skeleton: bool = False,
    min_pixels: int = 10,
    cluster_eps: float = 2.0,
    exclude_edge_touching: bool = False,
    debug: bool = False,
    debug_max_per_file: int = 12,
    pixel_size_csv: str | None = None,
    pixel_filename_col: str = "filename",
    pixel_size_col: str = "pixel_size_nm",
    overlap_radius: int = 3,
    lookahead: int = 3,
    spacing: float = 1.0,
    pre_smooth_window: int = 15,
    pre_smooth_poly: int = 3,
    trim_frac: float = 0.02,
    bend_angle_deg: float = 60.0,
    #bend_kappa_min: float = 0.0,
    bend_min_span_px: float = 5.0,
    bend_span_nm_ref: float = 10.0,
):
    """
    Compute geometric, curvature, persistence-length, and bend-count features.

    Columns:
      - pixel_size_nm directly after filename
      - rg_px, rg_nm
      - normalized_rg (dimensionless) -> renamed to compaction in final CSV
      - total_length_px, total_length_nm
      - no_strong_bends_px / no_strong_bends_nm from curvature batch

    NOTE: The final CSV intentionally excludes: n_edges, n_theta_increments, n_eval.

    Pixel-size behavior:
      - Always outputs pixel-based quantities.
      - If pixel_size_csv is missing/unreadable OR a file has no valid pixel size entry,
        nm-converted columns are left blank (NaN in dataframe / empty in CSV).
      - Never crashes just because pixel_size_csv is not present.
    """
    if loader_kwargs is None:
        loader_kwargs = {}

    os.makedirs(output_folder, exist_ok=True)
    if debug:
        os.makedirs(os.path.join(output_folder, "DEBUG"), exist_ok=True)

    # ----------------------------
    # Robust pixel-size CSV loading
    # ----------------------------
    px_map = {}
    pixel_csv_ok = False
    if pixel_size_csv:
        try:
            if not os.path.isfile(pixel_size_csv):
                if debug:
                    print(f"[pixel_size_csv] not found: {pixel_size_csv} → PX-only output (nm columns blank).")
                px_map = {}
                pixel_csv_ok = False
            else:
                px_map = _auto_load_pixel_sizes(
                    pixel_csv=pixel_size_csv,
                    pixel_filename_col=pixel_filename_col,
                    pixel_size_col=pixel_size_col,
                )
                pixel_csv_ok = True
        except Exception as e:
            if debug:
                print(f"[pixel_size_csv] could not load '{pixel_size_csv}' ({e}) → PX-only output (nm columns blank).")
            px_map = {}
            pixel_csv_ok = False

    # Helper to lookup nm/px with the same filename-variant logic you had
    def _lookup_nm_per_px(fname: str) -> float:
        if not px_map:
            return np.nan
        nm_per_px = px_map.get(fname, np.nan)
        if not np.isfinite(nm_per_px):
            root_noext, _ = os.path.splitext(fname)
            nm_per_px = px_map.get(root_noext, nm_per_px)

            if not np.isfinite(nm_per_px) and fname.startswith("a_"):
                nm_per_px = px_map.get(fname[2:], nm_per_px)
                nm_per_px = px_map.get(os.path.splitext(fname[2:])[0], nm_per_px)

            if not np.isfinite(nm_per_px) and not fname.startswith("a_"):
                nm_per_px = px_map.get("a_" + fname, nm_per_px)
                nm_per_px = px_map.get("a_" + root_noext, nm_per_px)
        return nm_per_px

    summary = []
    tiffs = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".tif", ".tiff"))]

    for fname in tiffs:
        fpath = os.path.join(folder, fname)

        # Lookup pixel size (nm/px). If missing/invalid -> keep NaN (blank in CSV).
        nm_per_px = _lookup_nm_per_px(fname)

        # Load raw + annotation
        try:
            raw, ann = load_annotated_mask(
                fpath,
                dilation_radius=dilation_radius,
                do_skeletonize=do_skeletonize,
                **loader_kwargs
            )
        except Exception:
            continue

        # Relabel if binary
        try:
            u = np.unique(ann)
            if ann.dtype == bool or (u.size <= 2 and set(u.tolist()).issubset({0, 1})):
                ann = sk_label(ann.astype(bool), connectivity=2).astype(np.uint32)
        except Exception:
            continue

        try:
            comps = _component_lengths_px_from_idmap(ann, min_pixels, exclude_edge_touching)
        except Exception:
            continue
        if not comps:
            continue

        dbg_count_for_file = 0

        for gid, area_px, touches_edge_dna, total_len_px in comps:
            try:
                gid = int(gid)
                comp_mask = (ann == gid)
                ys, xs = np.nonzero(comp_mask)
                if ys.size == 0:
                    continue

                # Crop region
                miny, maxy = int(ys.min()), int(ys.max())
                minx, maxx = int(xs.min()), int(xs.max())
                raw_crop = raw[miny:maxy+1, minx:maxx+1]
                submask  = comp_mask[miny:maxy+1, minx:maxx+1]

                # Skeletonization check
                submask_c = _as_bool_c(submask)
                try:
                    is_skel = input_is_skeleton or _looks_skeletonized(submask_c)
                except Exception:
                    is_skel = False
                skel = submask_c if is_skel else skeletonize(submask_c)

                ys_s, xs_s = np.nonzero(skel)
                if ys_s.size == 0:
                    segments = []
                    adj = None
                else:
                    coord_set = set(zip(ys_s, xs_s))
                    adj = _build_adjacency(coord_set)
                    segments = _graph_segments(adj)

                # ---- Rg computation ----
                try:
                    rg_px, normalized_rg = compute_normalized_rg_px(submask, total_len_px)
                except Exception:
                    rg_px, normalized_rg = (np.nan, np.nan)

                # convert to nm ONLY if valid nm/px
                has_valid_px = np.isfinite(nm_per_px) and (nm_per_px > 0)
                if has_valid_px:
                    rg_nm = (rg_px * nm_per_px) if np.isfinite(rg_px) else np.nan
                    total_length_nm = (total_len_px * nm_per_px) if np.isfinite(total_len_px) else np.nan
                else:
                    rg_nm = np.nan
                    total_length_nm = np.nan

                # other features unchanged
                try:
                    n_br, br_centers = count_branchpoint_clusters_by_conv(
                        submask, cluster_eps=cluster_eps, skel_pre=skel
                    )
                except Exception:
                    n_br, br_centers = 0, None

                try:
                    tort = tortuosity_simple_from_skeleton(skel, total_len_px)
                except Exception:
                    tort = np.nan

                try:
                    lab_one = sk_label(submask.astype(np.uint8), connectivity=2)
                    rps = regionprops(lab_one)
                    if rps:
                        rp = rps[0]
                        maj  = float(rp.major_axis_length)
                        minr = float(max(rp.minor_axis_length, 1e-6))
                        elong = maj / minr
                    else:
                        elong = np.nan
                except Exception:
                    elong = np.nan

                # (optional debug plot unchanged)
                if debug and dbg_count_for_file < debug_max_per_file and ys_s.size > 0:
                    try:
                        dbg_path = os.path.join(
                            output_folder, "DEBUG",
                            f"{os.path.splitext(fname)[0]}_gid{gid}_DEBUG.png"
                        )
                        debug_plot_region(
                            raw=raw_crop,
                            submask=submask,
                            adj=adj,
                            segments=segments,
                            branch_centers=br_centers,
                            title=f"{fname} gid {gid}",
                            savepath=dbg_path,
                            skel=skel
                        )
                        dbg_count_for_file += 1
                    except Exception:
                        pass

                # Store results
                summary.append({
                    "filename": fname,
                    "pixel_size_nm": float(nm_per_px) if has_valid_px else np.nan,
                    "had_valid_pixel_size": bool(has_valid_px),
                    "comp_id": gid,
                    "touches_edge_dna": bool(touches_edge_dna),
                    "total_length_px": float(total_len_px),
                    "total_length_nm": float(total_length_nm) if np.isfinite(total_length_nm) else np.nan,
                    "rg_px": float(rg_px) if np.isfinite(rg_px) else np.nan,
                    "rg_nm": float(rg_nm) if np.isfinite(rg_nm) else np.nan,
                    "normalized_rg": float(normalized_rg) if np.isfinite(normalized_rg) else np.nan,
                    "n_branch_clusters": int(n_br),
                    "tortuosity": float(tort) if np.isfinite(tort) else np.nan,
                    "elongation": float(elong) if np.isfinite(elong) else np.nan,
                })
            except Exception:
                continue

    df_geom = pd.DataFrame(summary)

    # ----------------------------
    # Curvature / lp / bends batch
    # Ensure we don't pass a missing CSV path downstream
    # ----------------------------
    out_csv = os.path.join(output_folder, "geometric_features_summary.csv")
    pixel_csv_for_curv = pixel_size_csv if (pixel_size_csv and pixel_csv_ok) else None

    try:
        df_curv = batch_curvature_for_folder(
            folder=folder,
            pixel_csv=pixel_csv_for_curv,
            pixel_filename_col=pixel_filename_col,
            pixel_size_col=pixel_size_col,
            output_csv=None,
            dilation_radius=dilation_radius,
            overlap_radius=overlap_radius,
            lookahead=lookahead,
            do_skeletonize=do_skeletonize,
            spacing=spacing,
            pre_smooth_window=pre_smooth_window,
            pre_smooth_poly=pre_smooth_poly,
            trim_frac=trim_frac,
            bend_angle_deg=bend_angle_deg,
            #bend_kappa_min=bend_kappa_min,
            bend_min_span_px=bend_min_span_px,
            bend_span_nm_ref=bend_span_nm_ref,
            recursive=False,
        )
    except Exception as e:
        print(f"[analyze_rg_branch_shape] curvature batch failed: {e}")
        df_curv = pd.DataFrame()

    # merge geometric + curvature results
    if not df_curv.empty and not df_geom.empty:
        df_curv = df_curv.rename(columns={"gid": "comp_id"})
        merged = df_geom.merge(df_curv, on=["filename", "comp_id"], how="left", suffixes=("", "_curv"))
    else:
        merged = df_geom.copy()

    # --- remove redundant pixel_size_nm_curv if present ---
    if "pixel_size_nm_curv" in merged.columns:
        merged = merged.drop(columns=["pixel_size_nm_curv"])

    # --- hard-drop columns you asked to exclude from FINAL CSV ---
    drop_cols = [c for c in ["n_edges", "n_theta_increments", "n_eval"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # reorder columns for clarity
    preferred = [
        "filename", "pixel_size_nm", "had_valid_pixel_size", "comp_id",
        "touches_edge_dna",
        "total_length_px", "total_length_nm",
        "rg_px", "rg_nm", "normalized_rg",
        "n_branch_clusters", "tortuosity", "elongation",
        "mean_kappa_px_inv", "std_kappa_px_inv", "min_kappa_px_inv", "max_kappa_px_inv",
        "mean_kappa_nm_inv", "std_kappa_nm_inv", "min_kappa_nm_inv", "max_kappa_nm_inv",
        "no_strong_bends_px", "no_strong_bends_nm",
        "lp_px", "lp_nm",
        "error",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols]

    # rename normalized_rg -> compaction
    if "normalized_rg" in merged.columns:
        merged.rename(columns={"normalized_rg": "compaction"}, inplace=True)

    # save CSV (NaNs become blank cells)
    merged.to_csv(out_csv, index=False)

    # ----------------------------
    # SIMPLE OVERLAYS (optional)
    # ----------------------------
    make_overlays = True              # or add as a function arg
    overlay_dir = os.path.join(output_folder, "feature_overlays")
    overlay_fontsize = 3
    overlay_outline = False           # set True if you want outlines
    overlay_ext = "pdf"               # "png" recommended (PDF can break in PyInstaller)
    os.makedirs(overlay_dir, exist_ok=True)

    # tiny formatting inline (no helpers)
    def _fmt_val(x, nd=3):
        try:
            x = float(x)
        except Exception:
            return "NA"
        return "NA" if not np.isfinite(x) else f"{x:.{nd}g}"

    def _fmt_int(x):
        try:
            if pd.isna(x):
                return "NA"
            return str(int(float(x)))
        except Exception:
            return "NA"

    if make_overlays and (not merged.empty):
        for fname in sorted(set(merged["filename"].astype(str))):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue

            # reload raw + ann (simple + robust)
            try:
                raw, ann = load_annotated_mask(
                    fpath,
                    dilation_radius=dilation_radius,
                    do_skeletonize=do_skeletonize,
                    **(loader_kwargs or {})
                )
            except Exception:
                continue

            # ensure label map so comp_id matches
            try:
                u = np.unique(ann)
                if ann.dtype == bool or (u.size <= 2 and set(u.tolist()).issubset({0, 1})):
                    ann = sk_label(ann.astype(bool), connectivity=2).astype(np.uint32)
                else:
                    ann = ann.astype(np.uint32, copy=False)
            except Exception:
                continue

            sub = merged[merged["filename"].astype(str) == fname]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(raw, cmap="gray")
            ax.set_axis_off()

            text_effects = [pe.withStroke(linewidth=1.0, foreground="black")]

            for _, r in sub.iterrows():
                try:
                    gid = int(r["comp_id"])
                except Exception:
                    continue

                comp_mask = (ann == gid)
                ys, xs = np.nonzero(comp_mask)
                if ys.size == 0:
                    continue

                cy = float(np.mean(ys))
                cx = float(np.mean(xs))

                # optional outline
                if overlay_outline:
                    try:
                        contours = find_contours(comp_mask.astype(float), 0.5)
                        for ct in contours:
                            ax.plot(ct[:, 1], ct[:, 0], linewidth=0.7)
                    except Exception:
                        pass

                # pick nm if present, else px fallback
                kappa = r.get("mean_kappa_nm_inv", np.nan)
                kappa_unit = "nm⁻¹"
                if not np.isfinite(kappa):
                    kappa = r.get("mean_kappa_px_inv", np.nan)
                    kappa_unit = "px⁻¹"

                bends = r.get("no_strong_bends_nm", np.nan)
                bends_tag = "nm-span"
                if not np.isfinite(bends):
                    bends = r.get("no_strong_bends_px", np.nan)
                    bends_tag = "px-span"

                # compaction column is what you renamed normalized_rg to
                compaction = r.get("compaction", np.nan)

                label_txt = (
                    f"ID: {gid}\n"
                    f"compaction: {_fmt_val(compaction, 3)}\n"
                    f"#crosses: {_fmt_int(r.get('n_branch_clusters', np.nan))}\n"
                    f"elongation: {_fmt_val(r.get('elongation', np.nan), 3)}\n"
                    f"curvature ({kappa_unit}): {_fmt_val(kappa, 3)}\n"
                    f"#strong bends ({bends_tag}): {_fmt_int(bends)}"
                )

                ax.text(
                    cx, cy, label_txt,
                    ha="center", va="center",
                    fontsize=overlay_fontsize,
                    color="yellow",
                    path_effects=text_effects,
                )

            out_name = os.path.splitext(fname)[0] + f"_geom_overlay.{overlay_ext}"
            out_path = os.path.join(overlay_dir, out_name)
            fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    return merged, out_csv


conn4 = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]], dtype=bool)

# Mahotas hit-or-miss templates (1=fg, 0=bg, 2=don't-care)
skel_branchpoint1 = np.array([[2, 2, 1], [1, 1, 2], [2, 2, 1]])
skel_branchpoint2 = np.array([[1, 2, 2], [2, 1, 1], [1, 2, 2]])
skel_branchpoint3 = np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
skel_branchpoint4 = np.array([[2, 1, 2], [1, 1, 2], [2, 2, 1]])
skel_branchpoint5 = np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
skel_branchpoint6 = np.array([[2, 2, 1], [1, 1, 2], [2, 1, 2]])

def find_branch_endpoints(skel):
    ep = 0
    for tmpl in (
        skel_branchpoint1, skel_branchpoint2, skel_branchpoint3,
        skel_branchpoint4, skel_branchpoint5, skel_branchpoint6
    ):
        ep += mh.morph.hitmiss(skel, tmpl)
        ep += mh.morph.hitmiss(skel, tmpl[::-1, :])
        ep += mh.morph.hitmiss(skel, tmpl[:, ::-1])
        ep += mh.morph.hitmiss(skel, tmpl.T)
    return ep > 0.5

skel_endpoints1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
skel_endpoints2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
skel_endpoints3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])
skel_endpoints4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
skel_endpoints5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
skel_endpoints6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
skel_endpoints7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
skel_endpoints8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])

def find_skel_endpoints(skel):
    ep = 0
    for tmpl in (skel_endpoints1, skel_endpoints2, skel_endpoints3, skel_endpoints4,
                 skel_endpoints5, skel_endpoints6, skel_endpoints7, skel_endpoints8):
        ep += mh.morph.hitmiss(skel, tmpl)
    return ep > 0.5

def make_graph(skel):
    broken = skel & (~find_branch_endpoints(skel))
    sections = label(broken)
    if sections.max() == 0:
        return np.empty((0,2), dtype=int)
    vals, counts = np.unique(sections, return_counts=True)
    i = vals[1:][np.argmax(counts[1:])]
    skel2 = (sections == i) | find_branch_endpoints(skel)
    sections = label(skel2)
    if sections.max() == 0:
        return np.empty((0,2), dtype=int)
    vals, counts = np.unique(sections, return_counts=True)
    i = vals[1:][np.argmax(counts[1:])]
    skel2 = sections == i
    skel2 = skeletonize(skel2)
    return find_path(skel2)

def find_path(skel):
    if skel.sum() == 0:
        return np.empty((0,2), dtype=int)
    path = np.zeros((int(skel.sum()), 2), dtype=int)
    coords = np.argwhere(skel)
    midpoint = np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1])])
    r, c = coords[np.argmin(coords[:, 0] + (2 + len(coords)) * coords[:, 1])]
    path[0, :] = r, c
    directions = np.array([[-1, 0], [ 1, 0], [ 0,-1], [ 0, 1],
                           [-1,-1], [ 1, 1], [-1, 1], [ 1,-1]])
    count = 1
    for i in range(1, len(path)):
        filled = False
        new_coord = path[i - 1, :][None] + directions
        distance = np.sum((new_coord - midpoint[None])**2, axis=1)
        idxs = np.argsort(distance)[::-1]
        for dr, dc in directions[idxs]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]): 
                continue
            if not skel[nr, nc]:
                continue
            if np.any((path[:count, 0] == nr) & (path[:count, 1] == nc)):
                continue
            path[i, :] = (nr, nc)
            count += 1
            filled = True
            r, c = nr, nc
            break
        if not filled:
            break
    return path[:count]

def filter_paths(gs, max_overlap=5):
    if not gs: 
        return gs
    lengths = [len(g) for g in gs]
    idxs = np.argsort(lengths)
    gs = [gs[i] for i in idxs]
    kept = [gs[0]]
    for g in gs[1:]:
        keep = True
        for g2 in kept:
            overlap = np.sum(np.abs(g[:, None, :] - g2[None, :, :]), axis=-1) < 0.5
            if overlap.sum() > max_overlap:
                keep = False
                break
        if keep:
            kept.append(g)
    return kept

@njit
def _dijkstra(img, s0, s1, t0, t1):
    h, w = img.shape
    n = h * w
    inf = 1e30
    d  = np.full(n, inf)
    pr = np.full(n, -1, np.int32)
    pc = np.full(n, -1, np.int32)
    vis = np.zeros(n, np.uint8)
    if img[s0, s1] == 0 or img[t0, t1] == 0:
        return np.empty((0, 2), np.int32)
    si = s0 * w + s1
    ti = t0 * w + t1
    d[si] = 0.0
    dr = np.array([1, -1, 0, 0, 1, 1, -1, -1], np.int32)
    dc = np.array([0, 0, 1, -1, 1, -1, 1, -1], np.int32)
    rt2 = np.sqrt(2.0)
    cost = np.array([1.0, 1.0, 1.0, 1.0, rt2, rt2, rt2, rt2])
    for _ in range(n):
        u = -1
        best = inf
        for i in range(n):
            if vis[i] == 0 and d[i] < best:
                best = d[i]
                u = i
        if u == -1 or u == ti:
            break
        vis[u] = 1
        r = u // w
        c = u - r * w
        for k in range(len(dr)):
            nr = r + dr[k]
            nc = c + dc[k]
            if 0 <= nr < h and 0 <= nc < w and img[nr, nc] != 0:
                v = nr * w + nc
                nd = d[u] + cost[k]
                if nd < d[v]:
                    d[v] = nd
                    pr[v] = r
                    pc[v] = c
    if d[ti] >= inf:
        return np.empty((0, 2), np.int32)
    rr = np.empty(n, np.int32)
    cc = np.empty(n, np.int32)
    L = 0
    r, c = t0, t1
    while not (r == s0 and c == s1):
        rr[L], cc[L] = r, c
        L += 1
        i = r * w + c
        r2, c2 = pr[i], pc[i]
        if r2 == -1:
            break
        r, c = r2, c2
    rr[L], cc[L] = s0, s1
    L += 1
    out = np.empty((L, 2), np.int32)
    for i in range(L):
        out[i, 0], out[i, 1] = rr[L - 1 - i], cc[L - 1 - i]
    return out

def shortest_path(img, start, goal):
    img = np.asarray(img).astype(np.uint8)
    s0, s1 = start
    t0, t1 = goal
    return _dijkstra(img, s0, s1, t0, t1)

def _lookup_nm_per_px(px_map, filepath):
    base = os.path.basename(filepath)
    root, _ = os.path.splitext(base)
    for k in (base, root, 'a_'+base, 'a_'+root, base[2:] if base.startswith('a_') else None, root[2:] if root.startswith('a_') else None):
        if k and k in px_map and np.isfinite(px_map[k]):
            return float(px_map[k])
    return None


def _out_img_path(annotated_path):
    head, base = os.path.split(annotated_path)
    parent = os.path.dirname(head)  # .../tiff/output
    loops_dir = os.path.join(parent, 'loops')
    return os.path.join(loops_dir, os.path.splitext(base)[0] + '.png')


def _out_csv_path(base_folder):
    return os.path.join(base_folder, "loops_summary.csv")

def _quantify_loops_single(
    annotated,
    min_length=10,
    loops_on_path_dist=3,
    dilation_radius=1,
    do_skeletonize=False,
    px_map=None,
    nm_per_bp_mean=None,
    save_overlay=True,
    overlay_dir=None,
    same_pos_thr=2.0,  # loops closer than this (symmetric min-dist) are treated as same-position (excluded)
    # accepted for compatibility with caller; not used in this version
    dna_intensity_radius_px=None,
    anchor_intensity_radius_px=None,
):
    """
    Quantify loops in ONE annotated TIFF.

    Intensity sampling (image data are NOT modified):
      - DNA contour intensity: binary_dilation(skeleton, 3x3) ∩ DNA mask
      - Anchor intensity:      binary_dilation(anchor_pixel, 3x3) ∩ DNA mask

    Overlay:
      - Loops plotted as before.
      - Anchor labels whose mean_anchor_intensity > (mean_dna_intensity + std_dna_intensity)
        are circled on the plot.
      - Loop text labels are shown in bp (if nm_per_px and nm_per_bp_mean are available),
        otherwise fall back to px.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle  # <-- for anchor highlighting
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize, binary_dilation
    from skimage.segmentation import clear_border
    from skimage.measure import label
    from skimage.filters import gaussian

    # ---- tiny helpers to keep everything C-contiguous ----
    def _c_uint8(a):   return np.ascontiguousarray(a, dtype=np.uint8)
    def _c_int32(a):   return np.ascontiguousarray(a, dtype=np.int32)
    def _c_float64(a): return np.ascontiguousarray(a, dtype=np.float64)

    # ---- geometry helpers ----
    def _polyline_len(poly):
        if len(poly) < 2:
            return 0.0
        seg = poly[1:] - poly[:-1]
        return float(np.sum(np.sqrt(np.sum(seg**2, axis=1))))

    def _sym_min_distance(A, B):
        """Symmetric directed-min distance (max of directed mins)."""
        if len(A) == 0 or len(B) == 0:
            return np.inf
        dAB = np.sqrt(np.min(np.sum((A[:, None] - B[None, :])**2, axis=-1), axis=1)).max()
        dBA = np.sqrt(np.min(np.sum((B[:, None] - A[None, :])**2, axis=-1), axis=1)).max()
        return max(dAB, dBA)

    def _inclusive_length_for_path_no_self(path, loops, current_loop, loops_on_path_dist, same_pos_thr):
        """
        Return (raw_len, inclusive_len, n_added, n_samepos_excluded)
        for a given skeleton path. Adds FULL length of other loops that lie near the path,
        EXCLUDING the current loop and loops at ~same position.
        """
        if path is None or len(path) < 2:
            return np.inf, np.inf, 0, 0

        raw_len = _polyline_len(path)
        inclusive = raw_len
        n_added = 0
        n_samepos_excluded = 0

        for g2 in loops:
            g2 = _c_int32(g2)
            if _sym_min_distance(current_loop, g2) < same_pos_thr:
                n_samepos_excluded += 1
                continue
            d2 = np.sqrt(np.min(np.sum((g2[:, None] - path[None, :])**2, axis=-1), axis=1))
            if (d2 < loops_on_path_dist).any():
                inclusive += _polyline_len(g2)
                n_added += 1

        return raw_len, inclusive, n_added, n_samepos_excluded

    # ---- load annotated (IDs preserved) ----
    raw, ann = load_annotated_mask(
        annotated,
        dilation_radius=dilation_radius,
        do_skeletonize=do_skeletonize
    )

    # ---- per-file calibration data ----
    nm_per_px = _lookup_nm_per_px(px_map, annotated) if px_map else None

    # px -> bp conversion for overlays (bp per pixel)
    if (
        (nm_per_px is not None) and np.isfinite(nm_per_px) and (nm_per_px > 0) and
        (nm_per_bp_mean is not None) and np.isfinite(nm_per_bp_mean) and (nm_per_bp_mean > 0)
    ):
        px_to_bp = float(nm_per_px) / float(nm_per_bp_mean)
    else:
        px_to_bp = None

    dna_len_by_gid = _dna_lengths_by_gid_from_idmap(
        ann=ann,
        min_area_px=0,
        exclude_edge_touching=False,
        px_nm=nm_per_px,
        nm_per_bp_mean=nm_per_bp_mean
    )

    rows = []
    fig = None
    try:
        if save_overlay:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(0.1 * raw.astype(float) + (ann > 0).astype(float), cmap='gray')
            ax.axis('off')

        gids = np.unique(ann)
        gids = gids[gids != 0]
        if gids.size == 0:
            return rows

        color_cycle = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray']
        footprint3 = np.ones((3,3), dtype=bool)  # 1 px dilation (Chebyshev radius 1)

        # NOTE: conn4 / make_graph / filter_paths / shortest_path / _out_img_path
        # are assumed to exist in your codebase, as in your original function.
        for gi, gid in enumerate(gids):
            mask = (ann == gid)
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            # Local bbox (contiguous view)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            submask = _c_uint8(mask[y0:y1, x0:x1])
            raw_sub = raw[y0:y1, x0:x1]  # raw for intensities

            # Skeleton for endpoints & routing (keep contiguous)
            subskel = _c_uint8(skeletonize(submask.astype(bool)))
            sy, sx = np.nonzero(subskel)
            if sy.size < 2:
                continue

            # --- per-DNA contour intensity (1 px dilation, restricted to DNA area)
            dil_skel = binary_dilation(subskel.astype(bool), footprint=footprint3)
            dil_skel &= submask.astype(bool)  # keep only pixels on the DNA mask
            dna_vals = raw_sub[dil_skel]
            if dna_vals.size > 0:
                mean_dna_intensity = float(np.mean(dna_vals))
                std_dna_intensity  = float(np.std(dna_vals, ddof=0))
            else:
                mean_dna_intensity = np.nan
                std_dna_intensity  = np.nan

            # Build adjacency (true endpoints = degree-1 skeleton nodes)
            coord_set = set(zip(sy, sx))
            adj = _build_adjacency(coord_set)
            endpoints = [(y, x) for (y, x), ns in adj.items() if len(ns) == 1]

            # Background components (holes) that might define loops
            mask_label = label(~submask, connectivity=1)
            mask_label = clear_border(mask_label)

            # Build loop polylines touching DNA contour
            gs = []
            for r in range(1, mask_label.max() + 1):
                m = (mask_label == r)
                touching = submask & binary_dilation(m, footprint=conn4) & (~m)
                if np.sum(touching) > min_length:
                    g = make_graph(touching)
                    if len(g) < min_length:
                        g = None
                else:
                    g = None
                if g is not None:
                    gs.append(_c_int32(g))

            if len(gs) >= 2:
                gs = filter_paths(gs)

            # KDTree for snapping to skeleton (float64 C-contiguous)
            sk_coords = _c_float64(np.column_stack([sy, sx]))
            tree = cKDTree(sk_coords)

            # Per-loop metrics
            for j, g in enumerate(gs):
                g_meta = {'gid': int(gid)}

                # Attachment on loop (remove loop, blur, pick max)
                mask_no_g = submask.copy()
                mask_no_g[g[:, 0], g[:, 1]] = 0
                blurred = gaussian(mask_no_g.astype(float), sigma=5)
                attachment = g[np.argmax(blurred[g[:, 0], g[:, 1]])].astype(int)

                # Snap attachment to nearest skeleton pixel
                att_pt = _c_float64([attachment[0], attachment[1]])
                _, idx = tree.query(att_pt, k=1)
                att_on_sk = sk_coords[int(idx)]
                att_on_sk = (int(att_on_sk[0]), int(att_on_sk[1]))
                g_meta['attachment_local'] = np.array(att_on_sk, dtype=np.int32)

                # --- anchor intensity at attachment (1 px dilation, restricted to DNA)
                anchor_mask = np.zeros_like(submask, dtype=bool)
                ay, axx = att_on_sk
                if 0 <= ay < anchor_mask.shape[0] and 0 <= axx < anchor_mask.shape[1]:
                    anchor_mask[ay, axx] = True
                anchor_mask = binary_dilation(anchor_mask, footprint=footprint3)
                anchor_mask &= submask.astype(bool)  # keep only DNA pixels
                anchor_vals = raw_sub[anchor_mask]
                if anchor_vals.size > 0:
                    mean_anchor_intensity = float(np.mean(anchor_vals))
                    std_anchor_intensity  = float(np.std(anchor_vals, ddof=0))
                else:
                    mean_anchor_intensity = np.nan
                    std_anchor_intensity  = np.nan

                # Loop polyline length (px) for reference
                g_meta['loop_length_px'] = _polyline_len(g)

                # Evaluate endpoints on the *skeleton* image
                best_raw_len   = np.inf
                best_incl_len  = np.inf
                best_added     = 0
                best_samepos   = 0

                if len(endpoints) > 0:
                    for ep in endpoints:
                        path = shortest_path(
                            _c_uint8(subskel),
                            (int(ep[0]), int(ep[1])),
                            (int(att_on_sk[0]), int(att_on_sk[1]))
                        )
                        if len(path) == 0:
                            continue
                        path = _c_int32(path)

                        raw_len, incl_len, n_added, n_samepos = _inclusive_length_for_path_no_self(
                            path=path,
                            loops=[g2 for g2 in gs],
                            current_loop=g,
                            loops_on_path_dist=loops_on_path_dist,
                            same_pos_thr=same_pos_thr
                        )

                        if (incl_len < best_incl_len) or (np.isclose(incl_len, best_incl_len) and raw_len < best_raw_len):
                            best_incl_len = incl_len
                            best_raw_len  = raw_len
                            best_added    = n_added
                            best_samepos  = n_samepos

                raw_px  = best_raw_len  if np.isfinite(best_incl_len) else np.nan
                incl_px = best_incl_len if np.isfinite(best_incl_len) else np.nan

                # ---------------- Overlay (plot) ----------------
                if save_overlay:
                    color = color_cycle[(gi + j) % len(color_cycle)]

                    # Plot loop polyline
                    g_plot = g.copy()
                    g_plot[:, 0] += y0
                    g_plot[:, 1] += x0
                    ax.plot(g_plot[:, 1], g_plot[:, 0], alpha=0.85, lw=1.5, c=color)

                    # Label with loop stats (bp if possible, else px)
                    if px_to_bp is not None:
                        loop_len_bp = g_meta["loop_length_px"] * px_to_bp
                        incl_bp = (incl_px * px_to_bp) if np.isfinite(incl_px) else np.nan
                        label_txt = f"s={loop_len_bp:.0f} bp, l={incl_bp:.0f} bp"
                    else:
                        label_txt = f"s={g_meta['loop_length_px']:.1f} px, l={incl_px if np.isfinite(incl_px) else np.nan:.1f} px"

                    ax.text(
                        g_plot[:, 1].mean(), g_plot[:, 0].mean(),
                        label_txt,
                        color=color, fontsize=8
                    )

                    # Highlight anchor if above DNA mean+std
                    if (
                        np.isfinite(mean_dna_intensity) and
                        np.isfinite(std_dna_intensity) and
                        np.isfinite(mean_anchor_intensity) and
                        (mean_anchor_intensity > (mean_dna_intensity + std_dna_intensity))
                    ):
                        # anchor global coords on the figure
                        ay_plot = ay + y0
                        ax_plot = axx + x0
                        circ = Circle(
                            (ax_plot, ay_plot),
                            radius=5.0,            # display radius in px; tweak if you want
                            fill=False,
                            linewidth=1.8,
                            edgecolor='yellow',
                            alpha=0.95
                        )
                        ax.add_patch(circ)

                # --------------- Row for CSV ---------------
                lens = dna_len_by_gid.get(int(g_meta["gid"]), {})
                dna_len_px = lens.get("dna_length_px", np.nan)
                dna_len_nm = lens.get("dna_length_nm", np.nan)
                dna_len_bp = lens.get("dna_length_bp", np.nan)
                touches_edge = lens.get("touches_edge_dna", np.nan)

                attach_global_y = int(g_meta['attachment_local'][0] + y0)
                attach_global_x = int(g_meta['attachment_local'][1] + x0)

                rows.append({
                    "file": annotated,
                    "dna_id": int(g_meta["gid"]),
                    "loop_index": j,
                    "loop_length_px": round(float(g_meta["loop_length_px"]), 3),
                    "raw_dist_px": round(float(raw_px), 3) if np.isfinite(raw_px) else np.nan,
                    "dist_incl_loops_px": round(float(incl_px), 3) if np.isfinite(incl_px) else np.nan,
                    "n_loops_added_inclusive": int(best_added if np.isfinite(incl_px) else 0),
                    "n_loops_excluded_samepos": int(best_samepos if np.isfinite(incl_px) else 0),

                    # intensity columns (from raw image; image unchanged)
                    "mean_dna_intensity": mean_dna_intensity,
                    "std_dna_intensity":  std_dna_intensity,
                    "mean_anchor_intensity": mean_anchor_intensity,
                    "std_anchor_intensity":  std_anchor_intensity,

                    # DNA totals
                    "dna_length_px": round(float(dna_len_px), 3) if np.isfinite(dna_len_px) else np.nan,
                    "dna_length_nm": round(float(dna_len_nm), 3) if np.isfinite(dna_len_nm) else np.nan,
                    "dna_length_bp": round(float(dna_len_bp), 3) if np.isfinite(dna_len_bp) else np.nan,
                    "touches_edge_dna": bool(touches_edge) if isinstance(touches_edge, (bool, np.bool_))
                                        else (bool(touches_edge) if np.isfinite(touches_edge) else np.nan),

                    # coords
                    "attachment_y": attach_global_y,
                    "attachment_x": attach_global_x,
                })

        # Save overlay
        if save_overlay and fig is not None:
            if overlay_dir is not None:
                os.makedirs(overlay_dir, exist_ok=True)
                out_pdf = os.path.join(
                    overlay_dir,
                    os.path.splitext(os.path.basename(annotated))[0] + ".pdf"
                )
            else:
                out_pdf = _out_img_path(annotated)
            fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')

    finally:
        if fig is not None:
            plt.close(fig)

    # ---- px -> nm -> bp for per-loop distances (stored in rows) ----
    if rows:
        nm_per_px2 = _lookup_nm_per_px(px_map, annotated) if px_map else None
        if nm_per_px2 is not None:
            for r in rows:
                ln = r.get("loop_length_px", np.nan)
                rd = r.get("raw_dist_px", np.nan)
                di = r.get("dist_incl_loops_px", np.nan)
                r["loop_length_nm"]     = round(ln * nm_per_px2, 3) if np.isfinite(ln) else np.nan
                r["raw_dist_nm"]        = round(rd * nm_per_px2, 3) if np.isfinite(rd) else np.nan
                r["dist_incl_loops_nm"] = round(di * nm_per_px2, 3) if np.isfinite(di) else np.nan

            if (nm_per_bp_mean is not None) and np.isfinite(nm_per_bp_mean) and (nm_per_bp_mean > 0):
                for r in rows:
                    ln = r.get("loop_length_nm", np.nan)
                    rd = r.get("raw_dist_nm", np.nan)
                    di = r.get("dist_incl_loops_nm", np.nan)
                    r["loop_length_bp"]     = round(ln / nm_per_bp_mean, 3) if np.isfinite(ln) else np.nan
                    r["raw_dist_bp"]        = round(rd / nm_per_bp_mean, 3) if np.isfinite(rd) else np.nan
                    r["dist_incl_loops_bp"] = round(di / nm_per_bp_mean, 3) if np.isfinite(di) else np.nan

    return rows


def quantify_loops_for_folder(
    segmented_folder: str,
    output_folder: str,
    pixel_size_csv: str | None = None,
    nm_per_bp_mean: float | None = None,
    min_length: int = 10,
    loops_on_path_dist: int = 3,
    dilation_radius: int = 1,
    do_skeletonize: bool = False,
    save_overlays: bool = True,
    dna_intensity_radius_px: float = 0.5,
    anchor_intensity_radius_px: float = 0.5,
):
    """
    Run loop quantification over all TIFFs inside <segmented_folder> (expects ML_annotated).
    Writes:
      - per-file overlays to <output_folder>/loops/
      - combined CSV to <output_folder>/loops_summary.csv
    Returns: (DataFrame, combined_csv_path)

    Notes:
      - The final CSV uses column 'filename' (not 'file'), storing only the base name.
      - The column 'dna_id' is renamed to 'comp_id' for consistency.
    """
    os.makedirs(output_folder, exist_ok=True)
    overlay_dir = os.path.join(output_folder, "loops") if save_overlays else None
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    px_map = _auto_load_pixel_sizes(pixel_size_csv) if pixel_size_csv else {}

    # Gather .tif/.tiff files
    patt1 = os.path.join(segmented_folder, "*.tif")
    patt2 = os.path.join(segmented_folder, "*.tiff")
    files = sorted(glob.glob(patt1)) + sorted(glob.glob(patt2))

    if not files:
        raise FileNotFoundError(
            f"No .tif/.tiff files found in segmented_folder:\n  {segmented_folder}\n"
            "Make sure you passed the ML_annotated directory."
        )

    all_rows = []
    for i, f in enumerate(files, 1):
        try:
            rows = _quantify_loops_single(
                f,
                min_length=min_length,
                loops_on_path_dist=loops_on_path_dist,
                dilation_radius=dilation_radius,
                do_skeletonize=do_skeletonize,
                px_map=px_map,
                nm_per_bp_mean=nm_per_bp_mean,
                save_overlay=save_overlays,
                overlay_dir=overlay_dir,
                same_pos_thr=2.0,
                dna_intensity_radius_px=dna_intensity_radius_px,
                anchor_intensity_radius_px=anchor_intensity_radius_px
            )

            # ensure consistent naming inside collected rows
            for r in rows:
                # keep only base filename (not full path)
                if "file" in r:
                    r["filename"] = os.path.basename(r.pop("file"))
                elif "filename" in r:
                    r["filename"] = os.path.basename(r["filename"])
                else:
                    r["filename"] = os.path.basename(f)

                # rename dna_id -> comp_id if present
                if "dna_id" in r:
                    r["comp_id"] = r.pop("dna_id")

            all_rows.extend(rows)

        except Exception as e:
            print(f"  !!! Loop-quant skipped {os.path.basename(f)} due to: {e}")

    out_csv = _out_csv_path(output_folder)

    if all_rows:
        df = pd.DataFrame(all_rows)

        # rename any leftover dna_id/file columns at DataFrame level (failsafe)
        rename_map = {}
        if "dna_id" in df.columns:
            rename_map["dna_id"] = "comp_id"
        if "file" in df.columns:
            rename_map["file"] = "filename"
        df = df.rename(columns=rename_map)

        # ensure consistent column order: px - nm - bp, with intensity columns included
        preferred = [
            "filename", "comp_id", "loop_index",
            "loop_length_px", "raw_dist_px", "dist_incl_loops_px",
            "n_loops_added_inclusive", "n_loops_excluded_samepos",

            # intensity columns
            "mean_dna_intensity", "std_dna_intensity",
            "mean_anchor_intensity", "std_anchor_intensity",

            # nm/bp conversions of distances
            "loop_length_nm", "raw_dist_nm", "dist_incl_loops_nm",
            "loop_length_bp", "raw_dist_bp", "dist_incl_loops_bp",

            # DNA totals
            "dna_length_px", "dna_length_nm", "dna_length_bp",
            "touches_edge_dna",

            # coords
            "attachment_y", "attachment_x",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols]

        df.to_csv(out_csv, index=False)

    else:
        df = pd.DataFrame(columns=[
            "filename", "comp_id", "loop_index",
            "loop_length_px", "raw_dist_px", "dist_incl_loops_px",
            "mean_dna_intensity", "std_dna_intensity",
            "mean_anchor_intensity", "std_anchor_intensity",
            "attachment_y", "attachment_x"
        ])
        df.to_csv(out_csv, index=False)

    return df, out_csv

