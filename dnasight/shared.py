import numpy as np
import tifffile
from roifile import ImagejRoi
from skimage.morphology import binary_dilation, disk as morph_disk, skeletonize, dilation, disk
from skimage.measure import label, find_contours


def load_annotated_mask_2(tiff_path, dilation_radius=5, do_skeletonize=False):
    with tifffile.TiffFile(tiff_path) as tif:
        arr = tif.asarray()
        metadata = tif.imagej_metadata

    if arr.ndim == 3 and arr.shape[0] == 2:
        # ML-style 2-channel TIFF: [raw, annotation]
        raw_image = arr[0]
        annotation_mask = arr[1] > 0
        print(f"Loaded ML-style 2-channel TIFF: {tiff_path}")

    else:
        # ImageJ ROI overlay fallback
        raw_image = arr
        annotation_mask = np.zeros(raw_image.shape, dtype=np.uint8)
        if metadata and 'Overlays' in metadata:
            roi_data = metadata['Overlays']
            if isinstance(roi_data, list):
                for roi_bytes in roi_data:
                    roi = ImagejRoi.frombytes(roi_bytes)
                    x, y = roi.coordinates().T
                    x = np.clip(x.astype(int), 0, raw_image.shape[1] - 1)
                    y = np.clip(y.astype(int), 0, raw_image.shape[0] - 1)
                    annotation_mask[y, x] = 1
            print(f"Loaded ROI overlay from: {tiff_path}")
        else:
            raise ValueError(f"No annotation found in {tiff_path}")

    # Dilation and optional skeletonization
    if dilation_radius > 0:
        annotation_mask = binary_dilation(annotation_mask, morph_disk(dilation_radius))
    if do_skeletonize:
        annotation_mask = skeletonize(annotation_mask)

    return raw_image, annotation_mask.astype(bool)


def load_annotated_mask(tiff_path, dilation_radius=1, do_skeletonize=False):
    """
    Load a TIFF with either:
    - 2-channel format (ch0 = raw, ch1 = annotation):
        * If ch1 already contains unique IDs (non-binary), PRESERVE them.
        * If ch1 is binary (0/1 or 0/255), relabel connected components 1..N.
    - ImageJ-style ROI overlays:
        * Build a binary mask from overlays, then relabel 1..N.

    Returns:
        raw_image         : 2D ndarray (kept as-loaded; often uint8/uint16/uint32)
        annotation_mask   : 2D ndarray of labels (0 = background, 1..N = component IDs).
                            If the input had global IDs in ch1, those IDs are preserved.
    """

    def _is_binary(arr):
        # treat 0/1 or 0/255 as binary
        u = np.unique(arr)
        if len(u) <= 2 and set(u.tolist()).issubset({0, 1}):
            return True
        if len(u) <= 2 and set(u.tolist()).issubset({0, 255}):
            return True
        return False

    def _per_label_op(id_map, op):
        """Apply a binary op per label and write back the same label."""
        if id_map.dtype != np.uint32:
            id_map = id_map.astype(np.uint32, copy=False)
        out = np.zeros_like(id_map, dtype=np.uint32)
        for gid in np.unique(id_map):
            if gid == 0:
                continue
            m = (id_map == gid)
            mm = op(m)  # binary -> binary
            out[mm] = gid
        return out

    def _maybe_dilate(id_map):
        if dilation_radius and dilation_radius > 0:
            se = disk(int(dilation_radius))
            return _per_label_op(id_map, lambda m: dilation(m, se))
        return id_map

    def _maybe_skeletonize(id_map):
        if do_skeletonize:
            return _per_label_op(id_map, lambda m: skeletonize(m))
        return id_map

    with tifffile.TiffFile(tiff_path) as tif:
        arr = tif.asarray()
        metadata = tif.imagej_metadata

    # ---------- Case A: 2-channel (ML) ----------
    if arr.ndim == 3 and arr.shape[0] == 2:
        raw_image = arr[0]
        ann = arr[1]

        # If ch1 is already an ID map (non-binary), preserve as-is.
        if not _is_binary(ann):
            # Ensure integer type (prefer uint32 for IDs)
            if not np.issubdtype(ann.dtype, np.integer):
                ann = ann.astype(np.uint32)
            else:
                # upcast smaller ints to uint32 (safer for large IDs)
                if ann.dtype.itemsize < np.dtype(np.uint32).itemsize:
                    ann = ann.astype(np.uint32)
            # Apply per-label ops without destroying IDs
            ann = _maybe_dilate(ann)
            ann = _maybe_skeletonize(ann)
            #print(f"Loaded ML 2-channel with preserved IDs: {tiff_path}")
            return raw_image, ann

        # Binary channel -> relabel 1..N
        bin_mask = (ann > 0)
        if dilation_radius and dilation_radius > 0:
            bin_mask = dilation(bin_mask, disk(int(dilation_radius)))
        # label connected components
        lbl = label(bin_mask, connectivity=2).astype(np.uint32)
        if do_skeletonize:
            lbl = _per_label_op(lbl, lambda m: skeletonize(m))
        #print(f"Loaded ML 2-channel (binary -> relabeled 1..N): {tiff_path}")
        return raw_image, lbl

    # ---------- Case B: ImageJ Overlays ----------
    # Fallback: single-plane raw + ROI overlays
    raw_image = arr
    H, W = raw_image.shape[-2], raw_image.shape[-1]
    bin_mask = np.zeros((H, W), dtype=bool)

    if metadata and 'Overlays' in metadata and metadata['Overlays']:
        # Build a binary mask from all overlay ROIs
        for roi_bytes in metadata['Overlays']:
            roi = ImagejRoi.frombytes(roi_bytes)
            xy = roi.coordinates()
            if xy.size == 0:
                continue
            # rasterscan the polygon/trace into a mask
            # For simplicity: mark the coordinates; optional: thicken lines
            x, y = xy[:, 0], xy[:, 1]
            x = np.clip(x.astype(int), 0, W - 1)
            y = np.clip(y.astype(int), 0, H - 1)
            bin_mask[y, x] = True

        # optional dilation before labeling to thicken traces
        if dilation_radius and dilation_radius > 0:
            bin_mask = dilation(bin_mask, disk(int(dilation_radius)))

        lbl = label(bin_mask, connectivity=2).astype(np.uint32)
        if do_skeletonize:
            lbl = _per_label_op(lbl, lambda m: skeletonize(m))
        print(f"Loaded ROI overlays (relabeled 1..N): {tiff_path}")
        return raw_image, lbl

    # No overlays found
    raise ValueError(f"No valid annotation found in {tiff_path} (no 2-channel mask or ROI overlays)")
