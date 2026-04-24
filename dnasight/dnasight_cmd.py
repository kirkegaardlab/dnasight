# These imports are for PyInstaller
import imagecodecs
import imagecodecs._imcd
import imagecodecs._shared
#

import sys  # To ensure print statements are immediate
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import yaml
import urllib.request
import ssl
import certifi
from pathlib import Path


from dnasight.dataset import DNAClusterDataset
from dnasight.train import train_unet
from dnasight.unet import UNet
from dnasight.dna import run_model_on_unannotated, calibrate_multiple_folders, analyze_rg_branch_shape, quantify_dna_lengths_bp
from dnasight.cluster import (
    normalize_cluster_metrics, link_clusters_to_dna,
    build_cluster_centered_summary, process_folder_clusters_dispatch, summarize_and_make_overlays
)
import warnings
warnings.filterwarnings("ignore")  # hide all warnings


def cmd_train_unet(args):
    args.save_dir = args.save_dir.rstrip('/')
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_plots:
        save_plots = args.save_dir + '/unet'
        os.makedirs(save_plots, exist_ok=True)
    else:
        save_plots = None

    unet = UNet(n_channels=1, n_classes=1).to(args.device)
    dataset = DNAClusterDataset(args.folder, augment=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train_unet(unet, train_loader, device=args.device, epochs=args.epochs, lr=args.lr, save_plots=save_plots)
    torch.save(unet.state_dict(), f'{args.save_dir}/unet.pt')
    print('Trained model saved to', f'{args.save_dir}/unet.pt')

def merge_quant_and_geo_on_comp_id(quant_csv: str, geo_csv: str, out_dir: str) -> str: #Should expanded to also include looping features
    """
    Merge DNA quantification and geometric-feature outputs on `comp_id` only,
    then:
      - keep a SINGLE `touches_edge`
      - keep a SINGLE `length_px` (prefer quant's length_px; else geo's total_length_px; else geo's length_px)
      - keep a SINGLE `pixel_size_nm` (prefer quant's, else geo's)
      - keep ONLY the 'stem' name and expose it as `filename`
      - remove filename_* columns and the redundant source columns above
      - drop duplicate rows using (filename, comp_id) as the key (fall back to comp_id if filename missing)
      - write out as dna_summary.csv

    Returns: path to the written dna_summary.csv
    """

    if not (os.path.isfile(quant_csv) and os.path.isfile(geo_csv)):
        missing = [p for p in [quant_csv, geo_csv] if not os.path.isfile(p)]
        raise FileNotFoundError(f"Cannot merge; missing: {missing}")

    os.makedirs(out_dir, exist_ok=True)

    q = pd.read_csv(quant_csv)
    g = pd.read_csv(geo_csv)

    # Ensure comp_id is numeric for both
    q["comp_id"] = pd.to_numeric(q.get("comp_id"), errors="coerce")
    g["comp_id"] = pd.to_numeric(g.get("comp_id"), errors="coerce")

    # Inner merge on comp_id only (as requested)
    merged = q.merge(g, on="comp_id", how="inner", suffixes=("_quant", "_geo"))

    # Helper: first non-null across several columns
    def coalesce(df, cols, default=np.nan):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(default, index=df.index)
        out = df[present[0]].copy()
        for c in present[1:]:
            out = out.where(out.notna(), df[c])
        if default is not np.nan:
            out = out.fillna(default)
        return out

    # Build canonical filename from stem (prefer quant's)
    filename = coalesce(
        merged,
        ["stem", "stem_quant", "stem_geo", "filename_quant", "filename_geo"]
    )

    # Unified pixel size
    pixel_size_nm = coalesce(
        merged,
        ["pixel_size_nm_quant", "pixel_size_nm_geo", "pixel_size_nm"]
    )

    # Unified touches_edge
    touches_edge = coalesce(
        merged,
        ["touches_edge_quant", "touches_edge_geo", "touches_edge"]
    ).astype("boolean")  # preserves NA as <NA>

    # Unified length_px (quant length_px > geo total_length_px > geo length_px)
    length_px = coalesce(
        merged,
        ["length_px_quant", "total_length_px_geo", "length_px_geo", "total_length_px", "length_px"]
    )

    # Attach canonical columns
    merged["filename"] = filename
    merged["pixel_size_nm"] = pd.to_numeric(pixel_size_nm, errors="coerce")
    merged["touches_edge"] = touches_edge
    merged["length_px"] = pd.to_numeric(length_px, errors="coerce")

    # Drop the now-redundant source columns
    drop_cols = []
    drop_cols += [c for c in ["stem", "stem_quant", "stem_geo",
                              "filename_quant", "filename_geo"]
                  if c in merged.columns]
    drop_cols += [c for c in ["pixel_size_nm_quant", "pixel_size_nm_geo"]
                  if c in merged.columns]
    drop_cols += [c for c in ["touches_edge_quant", "touches_edge_geo"]
                  if c in merged.columns]
    drop_cols += [c for c in ["length_px_quant", "total_length_px_geo",
                              "length_px_geo", "total_length_px"]
                  if c in merged.columns]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Final column order (canonical first, then everything else as-is)
    canonical_first = [c for c in ["filename", "comp_id", "pixel_size_nm", "touches_edge", "length_px"]
                       if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in canonical_first]
    merged = merged[canonical_first + other_cols]

    # Deduplicate rows: keep the row with the most non-null values per (filename, comp_id)
    # Fallback to comp_id only if filename missing entirely.
    key_cols = ["filename", "comp_id"] if merged["filename"].notna().any() else ["comp_id"]
    merged["__nnz__"] = merged.notna().sum(axis=1)
    merged.sort_values(key_cols + ["__nnz__"], ascending=[True] * len(key_cols) + [False], inplace=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="first").drop(columns="__nnz__")

    # Write summary CSV
    geom_dir = os.path.join(out_dir, "geometric_features")
    os.makedirs(geom_dir, exist_ok=True)

    out_path = os.path.join(geom_dir, "dna_summary.csv")

    #merged.to_csv(out_path, index=False) #turns up empty!!!
    print(f"Wrote {out_path}  (rows={len(merged)})")
    return out_path

class RunTracker:
    def __init__(self):
        self.errors = []
    def error(self, msg: str):
        print(f"ERROR: {msg}")
        self.errors.append(msg)


def cmd_run_unet(args):
    args.output = args.output.rstrip('/') + '/'
    os.makedirs(args.output, exist_ok=True)

    rt = RunTracker()   # <— add this


    ## DNA
    if args.dna_segmentation:
        try:
            print("  === RUNNNING DNA SEGMENTATION === ")
            unet = UNet(n_channels=1, n_classes=1)
            unet.load_state_dict(torch.load(args.unet, map_location=args.device))
            unet.to(args.device)
            unet.eval()
            run_model_on_unannotated(
                unet,
                args.folder,                # raw input images for your main dataset
                args.output,                # will write to <output>/ML_annotated
                args.device,
                batch_size=1,
                threshold=args.dna_segmentation_threshold,
                peak_threshold=0.03,
                min_distance=5
            )
        except Exception as e:
            rt.error(f"DNA segmentation failed: {e}")


    ## DNA length quantification (Module 1)
    if args.dna_quantification is not None:
        print("  === RUNNING DNA QUANTIFICATION ===")

        # parse optional config
        quant_config = {}
        for entry in args.dna_quantification:
            quant_config.update(parse_kv_string(entry))

        # after building quant_config
        if "min_component_area_px" in quant_config:
            quant_config["min_component_area_px"] = int(float(quant_config["min_component_area_px"]))

        if "exclude_edge_touching" in quant_config:
            v = str(quant_config["exclude_edge_touching"]).strip().lower()
            quant_config["exclude_edge_touching"] = v in ("1", "true", "yes", "y")

        # Defaults
        min_area   = quant_config.get("min_component_area_px", 5)
        excl_edge  = quant_config.get("exclude_edge_touching", False)
        overlay    = quant_config.get("overlay", False)
        debug      = quant_config.get("debug", False)

        # --- Ensure DNA segmentation exists BEFORE quantification ---
        segmented_main = os.path.join(args.output, "ML_annotated")
        if not os.path.isdir(segmented_main):
            rt.error("DNA quantification requested but DNA segmentation output was not found (run DNA segmentation first).")
        else:

            # --- Default: use fixed nm/bp unless we successfully compute a calibration ---
            nm_per_bp_mean = None
            nm_per_bp_sem  = None
            try:
                if args.nm_per_bp is not None:
                    nm_per_bp_mean = float(args.nm_per_bp)
                    nm_per_bp_sem = 0.0
            except Exception:
                pass

            if args.dna_calibration or getattr(args, "dna_calibration_folders", None):
                print("  === SEGMENTING CALIBRATION FOLDERS ===")

                # Normalize entries -> list of dicts
                calib_entries = args.dna_calibration or []
                if isinstance(calib_entries, (str, dict)):
                    calib_entries = [calib_entries]

                parsed_specs = []
                for i, entry in enumerate(calib_entries, 1):
                    try:
                        if isinstance(entry, str):
                            spec = parse_kv_string(entry)           # "k=v,..." -> dict
                        elif isinstance(entry, dict):
                            spec = dict(entry)                       # copy
                        else:
                            print(f"[{i}] Unknown calibration spec type: {type(entry)}. Skipping.")
                            continue

                        # Coerce some fields if present
                        numeric_keys = (
                            "dna_bp",
                            "pixel_size_nm",
                            "perc_low",
                            "perc_high",
                            "dna_calibration_threshold",
                            "threshold",
                        )
                        for k in numeric_keys:
                            if k in spec and spec[k] is not None and spec[k] != "":
                                try:
                                    if k == "dna_bp":
                                        spec[k] = int(float(spec[k]))
                                    else:
                                        spec[k] = float(spec[k])
                                except Exception:
                                    pass
                        parsed_specs.append(spec)
                    except Exception as e:
                        print(f"[{i}] Skipping invalid calibration spec '{entry}': {e}")

                explicit_paths = getattr(args, "dna_calibration_folders", None) or []
                dna_bps = getattr(args, "dna_calibration_dna_bp", None) or []
                px_sizes = getattr(args, "dna_calibration_pixel_size_nm", None) or []
                perc_lows = getattr(args, "dna_calibration_perc_low", None) or []
                perc_highs = getattr(args, "dna_calibration_perc_high", None) or []

                for idx, folder in enumerate(explicit_paths):
                    if not folder:
                        continue
                    spec = {"path": folder}
                    if idx < len(dna_bps) and dna_bps[idx] is not None:
                        try:
                            spec["dna_bp"] = int(float(dna_bps[idx]))
                        except Exception:
                            spec["dna_bp"] = dna_bps[idx]
                    if idx < len(px_sizes) and px_sizes[idx] is not None:
                        try:
                            spec["pixel_size_nm"] = float(px_sizes[idx])
                        except Exception:
                            spec["pixel_size_nm"] = px_sizes[idx]
                    if idx < len(perc_lows) and perc_lows[idx] is not None:
                        try:
                            spec["perc_low"] = float(perc_lows[idx])
                        except Exception:
                            spec["perc_low"] = perc_lows[idx]
                    if idx < len(perc_highs) and perc_highs[idx] is not None:
                        try:
                            spec["perc_high"] = float(perc_highs[idx])
                        except Exception:
                            spec["perc_high"] = perc_highs[idx]
                    parsed_specs.append(spec)

                # (Re)load a UNet so calibration segmentation always works
                calib_unet = UNet(n_channels=1, n_classes=1)
                calib_unet.load_state_dict(torch.load(args.unet, map_location=args.device))
                calib_unet.to(args.device)
                calib_unet.eval()

                segmented_calibration_specs = []
                calib_root = os.path.join(args.output, "dna_calibration")
                os.makedirs(calib_root, exist_ok=True)

                thresholds_from_args = list(getattr(args, "dna_calibration_threshold", []) or [])
                threshold_defaults = list(getattr(args, "dna_calibration_threshold_defaults", []) or [])
                try:
                    thresholds_from_args = [float(t) for t in thresholds_from_args if t is not None]
                except Exception:
                    thresholds_from_args = [t for t in thresholds_from_args if isinstance(t, (int, float))]
                try:
                    threshold_defaults = [float(t) for t in threshold_defaults if t is not None]
                except Exception:
                    threshold_defaults = [t for t in threshold_defaults if isinstance(t, (int, float))]

                fallback_threshold = threshold_defaults[-1] if threshold_defaults else 0.8

                for i, spec in enumerate(parsed_specs, start=1):
                    raw_path = spec.get("path")
                    if not raw_path or not os.path.isdir(raw_path):
                        print(f"[{i}] Skipping invalid calibration path: {raw_path}")
                        continue

                    spec = dict(spec)
                    raw_threshold = spec.pop("threshold", None)
                    if raw_threshold in (None, ""):
                        raw_threshold = spec.pop("dna_calibration_threshold", None)
                    if raw_threshold in (None, ""):
                        raw_threshold = spec.pop("calibration_threshold", None)
                    threshold = None
                    if raw_threshold not in (None, ""):
                        try:
                            threshold = float(raw_threshold)
                        except Exception:
                            pass

                    idx = i - 1
                    if threshold is None:
                        if idx < len(thresholds_from_args):
                            threshold = thresholds_from_args[idx]
                        elif idx < len(threshold_defaults):
                            threshold = threshold_defaults[idx]
                        elif thresholds_from_args:
                            threshold = thresholds_from_args[-1]
                        elif threshold_defaults:
                            threshold = threshold_defaults[-1]
                        else:
                            threshold = fallback_threshold

                    try:
                        threshold = float(threshold)
                    except Exception:
                        threshold = fallback_threshold

                    # Folder name: "<dna_bp>bp_<pixel_size_nm>nm"
                    dna_bp = spec.get("dna_bp")
                    px_nm  = spec.get("pixel_size_nm") or spec.get("pixelsize_nm") or spec.get("pixel_size")
                    parts = []
                    try:
                        if dna_bp is not None:
                            parts.append(f"{int(float(dna_bp))}bp")
                    except Exception:
                        pass
                    try:
                        if px_nm is not None:
                            parts.append(f"{float(px_nm):g}nm")
                    except Exception:
                        pass
                    out_name = "_".join(parts) if parts else os.path.basename(os.path.abspath(raw_path))

                    calib_out = os.path.join(calib_root, out_name)
                    os.makedirs(calib_out, exist_ok=True)

                    print(f"    [{i}] Segmenting calibration folder → {out_name}")
                    run_model_on_unannotated(
                        calib_unet,
                        raw_path,       # RAW calibration images
                        calib_out,      # writes <calib_out>/ML_annotated
                        args.device,
                        batch_size=1,
                        threshold=threshold,
                        peak_threshold=0.03,
                        min_distance=5,
                    )

                    # Point spec to segmented output for calibrate_multiple_folders
                    seg_spec = dict(spec)
                    seg_spec["path"] = os.path.join(calib_out, "ML_annotated")
                    segmented_calibration_specs.append(seg_spec)

                if segmented_calibration_specs:
                    try:
                        print(f"  === RUNNING CALIBRATION on {len(segmented_calibration_specs)} segmented folders ===")
                        nm_per_bp_mean, nm_per_bp_sem = calibrate_multiple_folders(
                            segmented_calibration_specs,
                            calib_root,
                        )
                    except Exception as e:
                        print(f"Calibration failed ({e}); falling back to fixed nm/bp = {nm_per_bp_mean}")
                else:
                    print(f"No valid calibration folders; using fixed nm/bp = {nm_per_bp_mean}")
        try:
            # --- Quantify DNA (pass None for nm/bp if no calibration) ---
            df, out_csv = quantify_dna_lengths_bp(
                segmented_folder=os.path.join(args.output, "ML_annotated"),
                pixel_size_csv=args.pixel_size_csv,
                nm_per_bp_mean=nm_per_bp_mean,   # None if calibration absent/failed
                nm_per_bp_sem=nm_per_bp_sem,     # None if calibration absent/failed
                output_folder=os.path.join(args.output, "dna_quantification"),
                min_component_area_px=min_area,
                exclude_edge_touching=excl_edge,
                overlay=overlay,
                debug=debug,
            )
            print(f"DNA quantification succesful: {out_csv}")
        except Exception as e:
            rt.error(f"DNA quantification failed: {e}")



    # --- GEOMETRIC FEATURES (Module 2) ---
    if args.geometric_features is not None:
        try:
            print("  === RUNNNING GEOMETRIC QUANTIFICATION ===")

            # parse key=value pairs into dict
            geo_config = {}
            for entry in args.geometric_features:
                geo_config.update(parse_kv_string(entry))

            if getattr(args, "pixel_size_csv", None) and "pixel_size_csv" not in geo_config:
                geo_config["pixel_size_csv"] = args.pixel_size_csv

            if args.geo_exclude_edge_touching is not None:
                geo_config["exclude_edge_touching"] = bool(args.geo_exclude_edge_touching)
            if args.geo_bend_angle_deg is not None:
                geo_config["bend_angle_deg"] = args.geo_bend_angle_deg
            if args.geo_bend_min_span_px is not None:
                geo_config["bend_min_span_px"] = args.geo_bend_min_span_px
            if args.geo_bend_span_nm_ref is not None:
                geo_config["bend_span_nm_ref"] = args.geo_bend_span_nm_ref

            # capture outputs
            df_geo, geo_out_csv = analyze_rg_branch_shape(
                folder=os.path.join(args.output, "ML_annotated"),
                output_folder=os.path.join(args.output, "geometric_features"),
                **geo_config,
            )
            print(f"Geometric feature analysis successfull: {geo_out_csv}")
        except Exception as e:
            rt.error(f"Geometric feature analysis failed: {e}")


    ## Cluster quantification (Module 4)
    if args.cluster_segmentation:
        try:
            print("  === RUNNNING CLUSTER SEGMENTATION === ")
            cluster_output_folder = os.path.join(args.output, 'cluster_segmentation')
            os.makedirs(cluster_output_folder, exist_ok=True)

            # Parse any key=value strings into a single dict
            cluster_cfg = {}
            if args.cluster_cfg:
                for entry in args.cluster_cfg:
                    cluster_cfg.update(parse_kv_string(entry))

            selected_model = cluster_cfg.get("model", args.cluster_model or "rw")
            selected_model = (selected_model or "rw").lower()
            if selected_model == "large":
                selected_model = "rw"
            elif selected_model == "small":
                selected_model = "trackpy"
            cluster_cfg["model"] = selected_model

            if selected_model == "rw":
                cluster_cfg.setdefault("threshold_factor", args.cluster_large_threshold_factor)
                cluster_cfg.setdefault("dilation_foreground", args.cluster_large_dilation_foreground)
                cluster_cfg.setdefault("dilation_background", args.cluster_large_dilation_background)
                cluster_cfg.setdefault("min_area", args.cluster_large_min_area)
                cluster_cfg.setdefault("beta", args.cluster_large_beta)
            elif selected_model in ("trackpy", "tp"):
                cluster_cfg.setdefault("diameter", args.cluster_small_diameter)
                cluster_cfg.setdefault("minmass", args.cluster_small_minmass)
                cluster_cfg.setdefault("min_area_filter", args.cluster_small_min_area_filter)
                cluster_cfg.setdefault("max_area_filter", args.cluster_small_max_area_filter)

            # Run via dispatcher (bare keys auto-map to the chosen model)
            df_seg, seg_csv = process_folder_clusters_dispatch(
                model=selected_model,
                input_folder=args.folder,
                output_folder=cluster_output_folder,
                **{k: v for k, v in cluster_cfg.items() if k != "model"}
            )
            print(f"Cluster segmentation successfull ({len(df_seg)} clusters). CSV: {seg_csv}")
        except Exception as e:
            rt.error(f"Cluster segmentation failed: {e}")

    # --- Cluster quantification (Module 4) ---
    if args.cluster_quantification:
        print("  === RUNNING CLUSTER QUANTIFICATION ===")

        cluster_seg_out = os.path.join(args.output, "cluster_segmentation")
        seg_csv = os.path.join(cluster_seg_out, "segmentation_results.csv")

        # define this OUTSIDE the branch so it's always in scope (no UnboundLocalError)
        quant_out = os.path.join(args.output, "cluster_quantification")

        if not os.path.isfile(seg_csv):
            rt.error("Cluster quantification requested but cluster segmentation results were not found (run cluster segmentation first).")
        else:
            try:
                os.makedirs(quant_out, exist_ok=True)
                out_csv = os.path.join(quant_out, "cluster_quantification.csv")

                normalize_cluster_metrics(
                    csv_path=seg_csv,
                    pixelsize_csv=args.pixel_size_csv,
                    output_path=out_csv,
                    image_roots=[args.folder],
                )
                print(f"Cluster quantification succesful: {out_csv}")
            except Exception as e:
                rt.error(f"Cluster quantification failed: {e}")


    
    ## LOOP QUANTIFICATION (Module 3)
    if args.loop_quantification is not None:
        try:
            print("  === RUNNING LOOP QUANTIFICATION ===")
            loop_cfg = {}
            for entry in args.loop_quantification:
                loop_cfg.update(parse_kv_string(entry))

            if args.loop_min_length is not None:
                loop_cfg["min_length"] = args.loop_min_length

            # defaults (override by passing key=value on CLI)
            min_length        = int(loop_cfg.get("min_length", 10))
            loops_on_path_dist= int(loop_cfg.get("loops_on_path_dist", 3))
            dilation_radius   = int(loop_cfg.get("dilation_radius", 1))
            do_skeletonize    = bool(loop_cfg.get("do_skeletonize", False))
            save_overlays     = bool(loop_cfg.get("save_overlays", True))

            # Use calibration result if available from earlier step
            # (nm_per_bp_mean was computed above in the DNA quant section if args.dna_calibration)
            nm_per_bp_mean_for_loops = locals().get("nm_per_bp_mean", None)
            if nm_per_bp_mean_for_loops is None:
                try:
                    nm_per_bp_mean_for_loops = float(args.nm_per_bp)
                except Exception:
                    nm_per_bp_mean_for_loops = None

            seg_folder = os.path.join(args.output, "ML_annotated")
            loops_out  = os.path.join(args.output, "loop_quantification")
            os.makedirs(loops_out, exist_ok=True)

            from dnasight.dna import quantify_loops_for_folder
            df_loops, csv_loops = quantify_loops_for_folder(
                segmented_folder=seg_folder,
                output_folder=loops_out,
                pixel_size_csv=args.pixel_size_csv,
                nm_per_bp_mean=nm_per_bp_mean_for_loops,
                min_length=min_length,
                loops_on_path_dist=loops_on_path_dist,
                dilation_radius=dilation_radius,
                do_skeletonize=do_skeletonize,
                save_overlays=save_overlays,
            )
            print(f"Loop quantification done: {csv_loops} (rows={len(df_loops)})")
        except Exception as e:
            rt.error(f"Loop quantification failed: {e}")

    # If both modules ran in this invocation, merge on comp_id
    if args.dna_quantification and args.geometric_features:
        print("  === MERGING DNA FILES ===")
        try:
            merged_csv = merge_quant_and_geo_on_comp_id(
                quant_csv=out_csv,                     # from quantify_dna_lengths_bp
                geo_csv=geo_out_csv,                   # from analyze_rg_branch_shape
                out_dir=args.output
            )
            print(f"Merging DNA quantification and geometric features successfull: {merged_csv}")
        except Exception as e:
            rt.error(f"Merging DNA quantification and geometric features failed: {e}")
    
    # --- LINK CLUSTERS TO DNA when BOTH segmentations were requested ---
    links_csv = None  # track so we can delete later

    if args.dna_segmentation and args.cluster_segmentation:
        try:
            print("  === LINKING CLUSTERS TO DNA ===")

            dna_annotation_folder  = os.path.join(args.output, "ML_annotated")
            cluster_seg_folder     = os.path.join(args.output, "cluster_segmentation")
            seg_results_csv        = os.path.join(cluster_seg_folder, "segmentation_results.csv")

            out_dir = os.path.join(args.output, "cluster_quantification")
            os.makedirs(out_dir, exist_ok=True)
            links_csv = os.path.join(out_dir, "cluster_to_dna_links.csv")  # ephemeral

            # sanity checks
            if not os.path.isdir(dna_annotation_folder):
                raise FileNotFoundError(f"DNA annotations not found: {dna_annotation_folder}")
            if not os.path.isdir(cluster_seg_folder):
                raise FileNotFoundError(f"Cluster segmentations not found: {cluster_seg_folder}")
            if not os.path.isfile(seg_results_csv):
                raise FileNotFoundError(f"Missing segmentation_results.csv: {seg_results_csv}")

            links_df = link_clusters_to_dna(
                annotation_folder=dna_annotation_folder,
                cluster_seg_folder=cluster_seg_folder,
                segmentation_results_csv=seg_results_csv,
                output_csv=links_csv,
                dilation_px=args.dna_protein_dilation,
            )
            print(f"Cluster-DNA linking successfull: {links_csv} (rows={len(links_df)})")
        except Exception as e:
            rt.error(f"Cluster–DNA linking failed: {e}")


    # ---- After SEGMENT + LINK + CLUSTER QUANT + (DNA QUANT) ----
    if args.cluster_segmentation and args.cluster_quantification and args.dna_quantification:
        cluster_seg_out   = os.path.join(args.output, "cluster_segmentation")
        seg_csv           = os.path.join(cluster_seg_out, "segmentation_results.csv")

        quant_out         = os.path.join(args.output, "cluster_quantification")
        cluster_quant_csv = os.path.join(quant_out, "cluster_quantification.csv")

        # Prefer the link CSV from the previous step; fall back to default path
        links_csv_path = links_csv or os.path.join(quant_out, "cluster_to_dna_links.csv")

        # Prefer the canonical DNA-quant file, fall back to None if missing
        dna_quant_csv = os.path.join(args.output, "dna_quantification", "lengths_per_component.csv")
        if not os.path.isfile(dna_quant_csv):
            dna_quant_csv = None
            print("No DNA quant CSV found; proceeding without per-DNA lengths/edge flags.")

        # Sanity checks (do not enforce permanence of links_csv; we’ll delete it later)
        for pth, msg in [
            (seg_csv,           "segmentation results CSV"),
            (cluster_quant_csv, "cluster quantification CSV"),
            (links_csv_path,    "cluster→DNA links CSV"),
        ]:
            if not os.path.isfile(pth):
                raise FileNotFoundError(f"Missing {msg}: {pth}")

        # Build final cluster-centered summary
        final_csv = os.path.join(quant_out, "cluster_dna_summary.csv")
        print("  === BUILDING CLUSTER–DNA SUMMARY ===")
        summary_df = build_cluster_centered_summary(
            links_csv=links_csv_path,
            cluster_quant_csv=cluster_quant_csv,
            dna_quant_csv=dna_quant_csv,   # can be None
            out_csv=final_csv,
        )
        try:
            n_rows = len(summary_df)
        except Exception:
            n_rows = "?"
        print(f"Cluster-DNA link quantifications successfull: {final_csv}  (rows={n_rows})")

        # Clean up: remove the intermediate links CSV so it isn't saved in the end
        try:
            if os.path.isfile(links_csv_path):
                os.remove(links_csv_path)
                print("🧹 Removed intermediate cluster_to_dna_links.csv")
        except Exception as e:
            print(f"Could not remove cluster_to_dna_links.csv: {e}")

    try:
        dna_annot_folder   = os.path.join(args.output, "ML_annotated")
        cluster_seg_folder = os.path.join(args.output, "cluster_segmentation")
        lengths_csv_path   = os.path.join(args.output, "dna_quantification", "lengths_per_component.csv")
        seg_results_csv    = os.path.join(cluster_seg_folder, "segmentation_results.csv")

        have_all = (
            os.path.isdir(dna_annot_folder)
            and os.path.isdir(cluster_seg_folder)
            and os.path.isfile(seg_results_csv)
            and os.path.isfile(lengths_csv_path)
        )
        if have_all:
            try:
                print("  === BUILDING GROUP LINK SUMMARY + OVERLAYS ===")
                link_out_dir = os.path.join(args.output, "cluster_quantification")
                overlays_dir = os.path.join(link_out_dir, "overlays")
                os.makedirs(overlays_dir, exist_ok=True)

                group_csv_path = os.path.join(link_out_dir, "group_summary.csv")
                dna_centered_csv_path = os.path.join(link_out_dir, "dna_centered_summary.csv")

                df_groups = summarize_and_make_overlays(
                    dna_annot_folder=dna_annot_folder,
                    cluster_seg_folder=cluster_seg_folder,
                    lengths_csv_path=lengths_csv_path,
                    output_csv_path=group_csv_path,
                    output_overlay_folder=overlays_dir,
                    dna_centered_output_csv_path=dna_centered_csv_path,
                    dilation_radius_px=5,
                    min_overlap_px=0.5,
                    min_dna_component_area_px=1,
                    cluster_fill_alpha=0.25,
                    cluster_edge_alpha=0.85,
                    dna_fill_alpha=0.40,
                    dna_edge_color="purple",
                    dna_edge_lw=1.2,
                    text_size=4,
                    debug=True,
                )
                try:
                    print(f"Group summary saved: {group_csv_path} (rows={len(df_groups)})")
                except Exception:
                    print(f"Group summary saved: {group_csv_path}")
            except Exception as e:
                print(f"Group summary/overlays step failed: {e}")
            # --- Final success / error summary ---
            if rt.errors:
                print("\n=== DNAsight run finished with errors ===")
                for i, msg in enumerate(rt.errors, 1):
                    print(f"  {i}. {msg}")
                raise SystemExit(1)
            else:
                print("\nEverything ran successfully.")
                
        else:
            missing = []
            if not os.path.isdir(dna_annot_folder):   missing.append("ML_annotated")
            if not os.path.isdir(cluster_seg_folder): missing.append("cluster_segmentation")
            if not os.path.isfile(seg_results_csv):   missing.append("segmentation_results.csv")
            if not os.path.isfile(lengths_csv_path):  missing.append("lengths_per_component.csv")
            #print(f"  (Skipping group summary; missing: {', '.join(missing)})")

    except Exception as e:
        print(f"Group summary/overlays step failed: {e}")



def get_device():
    if torch.cuda.is_available():
        return 'cuda'

    if torch.backends.mps.is_available():
        return 'mps'

    return 'cpu'


def parse_dna_calibration_string(s):
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = [x.strip() for x in part.split("=", 1)]
            if value in ("", None):
                # Allow callers to omit optional values and fall back to defaults later.
                continue
            if key == "dna_bp":
                out[key] = int(float(value))
            elif key in (
                "pixel_size_nm",
                "perc_low",
                "perc_high",
                "dna_calibration_threshold",
                "threshold",
            ):
                out[key] = float(value)
            else:
                # path and any other string fields
                out[key] = value
        return out
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dna_calibration format: {s}. Error: {e}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_kv_string(s):
    """
    Parse 'key=value,key=value' into dict with proper Python types.
    """
    out = {}
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # try to coerce to int, float, bool
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
        out[k] = v
    return out

def download_with_certifi(url: str, dest_path: str):
    """
    Download `url` to `dest_path` using certifi CA bundle.
    Works inside PyInstaller *if* certifi is bundled.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Make sure anything using default SSL paths can also find this.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    ctx = ssl.create_default_context(cafile=certifi.where())

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r, open(dest_path, "wb") as f:
        f.write(r.read())


def load_config(path="config.yaml"):
    candidates = []

    # 1) as provided (cwd-relative or absolute)
    candidates.append(Path(path))

    # 2) next to this script (dev mode)
    candidates.append(Path(__file__).resolve().parent / path)

    # 3) next to the executable (PyInstaller)
    candidates.append(Path(sys.executable).resolve().parent / path)

    # 4) PyInstaller temp bundle dir (onefile)
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / path)

    for p in candidates:
        if p.is_file():
            with p.open("r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "config.yaml not found. Looked in:\n" + "\n".join(str(p) for p in candidates)
    )


def main():
    print('Loading arguments')
    config = load_config()

    parser = argparse.ArgumentParser(description="DNAsight command script")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Subcommand to run")

    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument('--device', type=str, default=config['shared']['device'], help='Device to use (e.g., cpu, cuda, mps)')

    ## Run
    run_parser = subparsers.add_parser('run', parents=[shared_parser], help='Run the model')
    run_parser.add_argument('--folder', required=True, type=str, help='folder containing unannotated TIFF images.')
    run_parser.add_argument('--output', type=str, default=config['run']['output'], help='Where to save output')
    run_parser.add_argument('--unet', type=str, default=config['run']['unet'], help='The trained Unet model to use (.pt file)')

    run_parser.add_argument('--dna_segmentation', action='store_true', default=config['run']['dna_segmentation'], help='Whether to run DNA segmentation')
    run_parser.add_argument('--dna_segmentation_threshold', type=float, default=config['run'].get('dna_segmentation_threshold', 0.7), help='Probability threshold for DNA segmentation masking')
    run_parser.add_argument(
        "--dna_quantification",
        nargs="*",
        metavar="key=value",
        default=None,
        help=(
            "Enable DNA quantification (Module 1). Optionally supply key=value overrides, "
            "e.g. min_component_area_px=75 exclude_edge_touching=False overlay=True"
        ),
    )
    run_parser.add_argument('--dna_quant_min_component_area', type=int, default=config['run'].get('dna_quant_min_component_area', 50), help='Minimum component area (px) for DNA quantification')
    run_parser.add_argument('--dna_quant_exclude_edge_touching', type=str2bool, choices=[True, False], default=config['run'].get('dna_quant_exclude_edge_touching', True), help='Exclude DNA components touching the image edge during quantification')

    run_parser.add_argument('--geometric_features',nargs="*",metavar="key=value",default=None,help="Optional key=value config for geometric analysis")
    run_parser.add_argument('--geo_exclude_edge_touching', type=str2bool, choices=[True, False], default=config['run'].get('geo_exclude_edge_touching', False), help='Exclude edge-touching components in geometric analysis')
    run_parser.add_argument('--geo_bend_angle_deg', type=float, default=config['run'].get('geo_bend_angle_deg', 60.0), help='Bend angle threshold in degrees')
    run_parser.add_argument('--geo_bend_min_span_px', type=float, default=config['run'].get('geo_bend_min_span_px', 5.0), help='Minimum span in pixels for bend detection')
    run_parser.add_argument('--geo_bend_span_nm_ref', type=float, default=config['run'].get('geo_bend_span_nm_ref', 10.0), help='Reference span in nm for bend detection')
    run_parser.add_argument('--loop_quantification', nargs="*", metavar="key=value", default=None,
                        help="Optional key=value config for loop quantification "
                             "(min_length,loops_on_path_dist,dilation_radius,do_skeletonize,save_overlays)")
    run_parser.add_argument('--loop_min_length', type=int, default=config['run'].get('loop_min_length', 10), help='Minimum loop length (px) to quantify')


    run_parser.add_argument('--cluster_segmentation', action='store_true', default=config['run']['cluster_segmentation'], help='Whether to run cluster analysis')
    run_parser.add_argument('--cluster_model', type=str, choices=['rw', 'trackpy', 'large', 'small'], default=config['run'].get('cluster_model', 'rw'), help="Cluster segmentation model: 'rw'/'large' or 'trackpy'/'small'")
    run_parser.add_argument('--cluster_large_threshold_factor', type=float, default=config['run'].get('cluster_large_threshold_factor', 1.5), help='Random-walker threshold factor (Large model)')
    run_parser.add_argument('--cluster_large_dilation_foreground', type=int, default=config['run'].get('cluster_large_dilation_foreground', 5), help='Foreground dilation radius (px) for Large model')
    run_parser.add_argument('--cluster_large_dilation_background', type=int, default=config['run'].get('cluster_large_dilation_background', 10), help='Background dilation radius (px) for Large model')
    run_parser.add_argument('--cluster_large_min_area', type=int, default=config['run'].get('cluster_large_min_area', 200), help='Minimum area (px) retained for Large model')
    run_parser.add_argument('--cluster_large_beta', type=float, default=config['run'].get('cluster_large_beta', 90), help='Beta parameter for Large model')
    run_parser.add_argument('--cluster_small_diameter', type=int, default=config['run'].get('cluster_small_diameter', 11), help='Trackpy diameter (Small model)')
    run_parser.add_argument('--cluster_small_minmass', type=float, default=config['run'].get('cluster_small_minmass', 300), help='Trackpy minmass (Small model)')
    run_parser.add_argument('--cluster_small_min_area_filter', type=float, default=config['run'].get('cluster_small_min_area_filter', 10), help='Minimum area filter (px) for Small model results')
    run_parser.add_argument('--cluster_small_max_area_filter', type=float, default=config['run'].get('cluster_small_max_area_filter', 2050), help='Maximum area filter (px) for Small model results')
    run_parser.add_argument('--cluster_min_area', type=float, default=config['run']['cluster_min_area'], help='nm2')
    run_parser.add_argument('--cluster_min_density', type=float, default=config['run']['cluster_min_density'], help='/nm2')
    run_parser.add_argument( '--cluster_cfg', nargs="*", metavar="key=value", default=config['run'].get('cluster_cfg', None), help="Cluster segmentation config. Example: 'model=trackpy,diameter=11,minmass=400'")


    run_parser.add_argument('--cluster_quantification', action='store_true', default=config['run']['cluster_quantification'], help='Whether to run cluster quantification')
    run_parser.add_argument('--pixel_size_csv', type=str, default=config['run']['pixel_size_csv'], help='csv file containing image pixel sizes')
    run_parser.add_argument('--dna_protein_dilation', type=int, default=config['run'].get('dna_protein_dilation', 3), help='Dilation radius (px) when linking DNA and protein clusters')

    run_parser.add_argument('--coverage_quantification', action='store_true', default=config['run']['coverage_quantification'], help='Whether to run DNA quantification')
    
    run_parser.add_argument('--nm_per_bp', type=float, default=config['run']['nm_per_bp'], help='nanometers per basepair (overwritten if data calibration is used)')
    run_parser.add_argument(
        '--dna_calibration',
        action='append',
        type=parse_dna_calibration_string,
        default=config['run']['dna_calibration'],
        help=(
            'Format: path=...,dna_bp=...,pixel_size_nm=... '
            '(perc_low/perc_high/dna_calibration_threshold are optional; defaults are used when omitted).'
        ),
    )
    run_parser.add_argument('--dna_calibration_threshold', action='append', type=float, default=None,
                            help='Segmentation threshold(s) applied to calibration folders; repeat to match each dataset')
    run_parser.add_argument('--dna_calibration_folders', nargs='*', default=config['run'].get('dna_calibration_folders', []), help='Calibration folders to segment')
    run_parser.add_argument('--dna_calibration_dna_bp', nargs='*', type=float, default=config['run'].get('dna_calibration_dna_bp', []), help='DNA base pairs corresponding to each calibration folder')
    run_parser.add_argument('--dna_calibration_pixel_size_nm', nargs='*', type=float, default=config['run'].get('dna_calibration_pixel_size_nm', []), help='Pixel size (nm) for each calibration folder')
    run_parser.add_argument('--dna_calibration_perc_low', nargs='*', type=float, default=config['run'].get('dna_calibration_perc_low', []), help='Lower percentile for calibration filtering per folder')
    run_parser.add_argument('--dna_calibration_perc_high', nargs='*', type=float, default=config['run'].get('dna_calibration_perc_high', []), help='Upper percentile for calibration filtering per folder')
    


    ## Train
    train_parser = subparsers.add_parser('train', parents=[shared_parser], help='Train the model')
    train_parser.add_argument('--folder', type=str, nargs='+', default=config['train']['folder'], help='One or more folders containing the training data')
    train_parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'], help='Batch size (reduce to lower RAM usage)')
    train_parser.add_argument('--lr', type=float, default=config['train']['lr'], help='Learning rate')
    train_parser.add_argument('--save_dir', type=str, default=config['train']['save_dir'], help='Path to save final model and outputs')
    train_parser.add_argument('--save_plots', action='store_true', default=config['train']['save_plots'], help='Whether to save training plots')


    # Run code
    args = parser.parse_args()
    if args.device == 'detect':
        args.device = get_device()
        print(f'Running on {args.device}')

    if args.command == 'train':
        cmd_train_unet(args)

    if args.command == 'run':
        def _coerce_float_list(values):
            out = []
            for v in values:
                try:
                    if v is None:
                        continue
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
            return out

        defaults = config['run'].get('dna_calibration_thresholds', []) or []
        if not isinstance(defaults, (list, tuple)):
            defaults = [defaults]
        legacy_default = config['run'].get('dna_calibration_threshold', None)
        if not defaults and legacy_default is not None:
            defaults = [legacy_default]
        defaults = _coerce_float_list(defaults)

        provided_thresholds = args.dna_calibration_threshold
        if provided_thresholds is None:
            provided_thresholds = defaults
        provided_thresholds = _coerce_float_list(provided_thresholds)

        args.dna_calibration_threshold = provided_thresholds
        args.dna_calibration_threshold_defaults = defaults

        # If --output is absolute, keep it. If it's relative, make it inside the folder.
        if os.path.isabs(args.output):
            out = args.output
        else:
            out = os.path.join(args.folder.rstrip('/'), args.output)
        args.output = out.rstrip('/') + '/'


        # Make sure unet file exists, otherwise download:
        if args.unet == 'model/unet.pt':
            os.makedirs(os.path.dirname(args.unet), exist_ok=True)
            if not os.path.exists(args.unet):
                print(f"{args.unet} not found. Downloading from server...")
                url = 'https://github.com/kirkegaardlab/dnasightsmodels/releases/download/model/unet.pt'
                # url = "https://sid.erda.dk/share_redirect/fJeJhVJugm"
                download_with_certifi(url, args.unet)
                print(f"Downloaded and saved to {args.unet}")

        cmd_run_unet(args)

if __name__ == '__main__':
    print('Running dnasight...')
    main()

