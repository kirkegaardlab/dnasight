import shlex, json, threading, subprocess, queue, sys
from pathlib import Path
from contextlib import contextmanager
import dearpygui.dearpygui as dpg
import subprocess
import platform
import os
import pandas as pd
import traceback

APP_TITLE = "DNAsight"
STATE = {
    "proc": None,
    "proc_thread": None,
    "log_queue": queue.Queue(),
    "running": False,
    "log": "",
}
SMALL_FONT = None
DEFAULT_FONT = None

LOGO_PATH_DEFAULT = "logo.png"
LOGO_TEX = None
LOGO_DISPLAY_H = 50  # px height in the header

CALIB_COLS = [
    ("path", "Path"),
    ("dna_bp", "DNA (bp)"),
    ("pixel_size_nm", "Pixel size (nm)"),
    ("perc_low", "Low %"),
    ("perc_high", "High %"),
    ("threshold", "Seg. thresh"),
]

CALIB_TOOLTIPS = {
    "Path": "Folder containing RAW calibration images for this dataset.\nThese will be segmented first, then used for nm/bp calibration.",
    "DNA (bp)": "Known length of the calibration DNA construct (in base pairs),\ne.g., 1059. Used to convert from pixels to nm/bp.",
    "Pixel size (nm)": "Instrument pixel size (nm per pixel) for these images.\nIf you pass a pixel-size CSV, that will override per-image values.",
    "Low %": "Lower percentile cut used when fitting the length distribution\n(e.g., 5) to remove short outliers and partial molecules.",
    "High %": "Upper percentile cut used when fitting the length distribution\n(e.g., 95) to remove long outliers and merged molecules.",
    "Seg. thresh": "UNet probability threshold used ONLY for segmenting the\ncalibration folders (not the main dataset).",
}

exe_mode = False
if getattr(sys, 'frozen', False):
    print('Frozen mode', getattr(sys, 'frozen', False))
    exe_mode = True
cmd_mode = False


# =========================
# Utilities & serialization
# =========================s
def bool_str(b: bool) -> str:
    return "True" if b else "False"


def fmt_float(value, precision: int = 3) -> str:
    try:
        num = float(value)
        return f"{num:.{precision}f}".rstrip('0').rstrip('.')
    except Exception:
        return str(value)


def build_dna_calibration_arg(row: dict):
    """Build a --dna_calibration specification string from a UI row."""

    # row keys: path, dna_bp, pixel_size_nm, perc_low, perc_high, threshold
    path = str(row.get("path", "")).strip()
    if not path:
        return None, "Calibration path is required."

    def _require_number(raw, caster, label):
        text = str(raw).strip()
        if text == "":
            return None, f"Calibration {label} is required."
        try:
            return caster(text), None
        except Exception:
            return None, f"Calibration {label} must be numeric."

    dna_bp, err = _require_number(row.get("dna_bp", ""), lambda v: int(float(v)), "DNA bp")
    if err:
        return None, err

    pixel, err = _require_number(row.get("pixel_size_nm", ""), float, "pixel size (nm)")
    if err:
        return None, err

    def _optional_number(raw, caster, label):
        text = str(raw).strip()
        if text == "":
            return None, None
        try:
            return caster(text), None
        except Exception:
            return None, f"Calibration {label} must be numeric."

    low, err = _optional_number(row.get("perc_low", ""), float, "lower percentile")
    if err:
        return None, err
    high, err = _optional_number(row.get("perc_high", ""), float, "upper percentile")
    if err:
        return None, err
    threshold, err = _optional_number(row.get("threshold", ""), float, "threshold")
    if err:
        return None, err

    parts = [
        f"path={path}",
        f"dna_bp={int(dna_bp)}",
        f"pixel_size_nm={fmt_float(pixel)}",
    ]
    if low is not None:
        parts.append(f"perc_low={fmt_float(low)}")
    if high is not None:
        parts.append(f"perc_high={fmt_float(high)}")
    if threshold is not None:
        parts.append(f"dna_calibration_threshold={fmt_float(threshold)}")

    spec = ",".join(parts)
    return spec, None


def get_table_rows(table_id):
    rows = []
    meta = dpg.get_item_user_data(table_id) or {}
    row_count = int(meta.get("rows", 0))
    keys = ["path", "dna_bp", "pixel_size_nm", "perc_low", "perc_high", "threshold"]
    for r in range(row_count):
        row = {}
        for col, key in enumerate(keys):
            field_id = meta.get(f"cell_{r}_{col}")
            row[key] = dpg.get_value(field_id) if field_id else ""
        rows.append(row)
    return rows


def set_table_rows(table_id, rows):
    # clear table
    meta = dpg.get_item_user_data(table_id) or {}
    current = int(meta.get("rows", 0))
    for r in range(current):
        # delete cells & row container
        for c in range(6):
            cid = meta.get(f"cell_{r}_{c}")
            if cid and dpg.does_item_exist(cid):
                dpg.delete_item(cid)
        rid = meta.get(f"row_{r}")
        if rid and dpg.does_item_exist(rid):
            dpg.delete_item(rid)
        del_btn = meta.get(f"del_{r}")
        if del_btn and dpg.does_item_exist(del_btn):
            dpg.delete_item(del_btn)
    meta.clear()
    meta["rows"] = 0
    dpg.set_item_user_data(table_id, meta)

    # add rows
    for dat in rows:
        add_calibration_row(table_id, preset=dat)

    preview_update()


# =========================
# Command building & preview
# =========================
def build_command():
    if exe_mode:
        if sys.platform.startswith('win'):
            cmd = ["dnasight-cmd.exe", "run"]
        else:
            cmd = ["./dnasight-cmd", "run"]

            sysname = platform.system()
            if sysname == "Darwin":  # macOS
                wd_file = sys.executable
                print('Setting working based on', wd_file)
                wd = os.path.abspath(os.path.join(os.path.dirname(wd_file)))
                print('Setting wd to', wd)
                os.chdir(wd)

    elif cmd_mode:
        cmd = ["dnasight-cmd", "run"]

    else:
        cmd = [sys.executable, "dnasight-cmd.py", "run"]

    # stages
    if dpg.get_value("dna_seg"):
        cmd.append("--dna_segmentation")
        threshold = float(dpg.get_value("dna_seg_threshold"))
        cmd += ["--dna_segmentation_threshold", fmt_float(threshold)]

    if dpg.get_value("cluster_seg"):
        cmd.append("--cluster_segmentation")
        cluster_model = (dpg.get_value("cluster_model") or "").strip().lower() or "large"
        cmd += ["--cluster_model", cluster_model]

        if cluster_model == 'large':
            # Large / random walker options
            cmd += [
                "--cluster_large_threshold_factor", fmt_float(dpg.get_value("cluster_large_threshold")),
                "--cluster_large_dilation_foreground", str(int(dpg.get_value("cluster_large_dilate_fg"))),
                "--cluster_large_dilation_background", str(int(dpg.get_value("cluster_large_dilate_bg"))),
                "--cluster_large_min_area", str(int(dpg.get_value("cluster_large_min_area"))),
                "--cluster_large_beta", fmt_float(dpg.get_value("cluster_large_beta")),
            ]
        else:
            # Small / trackpy options
            cmd += [
                "--cluster_small_diameter", str(int(dpg.get_value("cluster_small_diameter"))),
                "--cluster_small_minmass", fmt_float(dpg.get_value("cluster_small_minmass")),
                "--cluster_small_min_area_filter", fmt_float(dpg.get_value("cluster_small_min_area")),
                "--cluster_small_max_area_filter", fmt_float(dpg.get_value("cluster_small_max_area")),
            ]

    if dpg.get_value("cluster_quant"):
        cmd.append("--cluster_quantification")

    if dpg.get_value("dna_quant"):
        q = f"min_component_area_px={int(dpg.get_value('dq_min_area'))},"
        q += f"exclude_edge_touching={bool_str(dpg.get_value('dq_exclude_edge'))},"
        q += f"overlay={bool_str(dpg.get_value('dq_overlay'))}"
        cmd += ["--dna_quantification", q]

    if dpg.get_value("loop_quant"):
        q = f"min_length={int(dpg.get_value('lq_min_len'))}"
        cmd += ["--loop_quantification", q]

    if dpg.does_item_exist("lq_min_len"):
        cmd += ["--loop_min_length", str(int(dpg.get_value("lq_min_len")))]

    if dpg.get_value("geom_features"):
        # q = f"dilation_radius={int(dpg.get_value('gf_dilate'))},"
        # q = f"do_skeletonize={bool_str(dpg.get_value('gf_skel'))},"
        q = f"min_pixels={int(dpg.get_value('gf_min_px'))},"
        # q += f"cluster_eps={fmt_float(dpg.get_value('gf_eps'))},"
        q += f"exclude_edge_touching={bool_str(dpg.get_value('gf_exclude_edge'))},"
        q += f"bend_angle_deg={fmt_float(dpg.get_value('gf_bend_angle'))},"
        q += f"bend_min_span_px={fmt_float(dpg.get_value('gf_bend_span_px'))},"
        q += f"bend_span_nm_ref={fmt_float(dpg.get_value('gf_bend_span_nm'))}"
        cmd += ["--geometric_features", q]
        cmd += ["--geo_curvature_smoothing", str(int(dpg.get_value("gf_curvature_smoothing")))]

    # paths
    folder = dpg.get_value("in_folder").strip()
    output = dpg.get_value("out_folder").strip()
    # unet = dpg.get_value("unet_model").strip()
    pixel_csv = dpg.get_value("pixel_csv").strip()
    if folder:    cmd += ["--folder", folder]
    if output:    cmd += ["--output", output]
    # if unet:      cmd += ["--unet", unet]
    if pixel_csv:
        cmd += ["--pixel_size_csv", pixel_csv]
    else:
        cmd += ["--pixel_size_csv", folder + "/" + output + "/" + "pixel_size.csv"]

    # calibrations
    rows = get_table_rows("calib_table")
    if len(rows) == 0:
        cmd += ["--nm_per_bp", dpg.get_value("nm_per_bp").strip()]
    else:
        for row in rows:
            s, err = build_dna_calibration_arg(row)
            if err:
                cmd += ["--dna_calibration", "MISSING DATA"]
            else:
                cmd += ["--dna_calibration", s]

    if dpg.get_value("cluster_quant"):
        cmd += ["--dna_protein_dilation", str(int(dpg.get_value("dna_protein_dilation")))]

    return cmd, None


def preview_update():
    cmd, err = build_command()

    if (not exe_mode) and (not cmd_mode):
        cmd[0] = 'python'

    if err:
        dpg.set_value("preview", f"# Error: {err}")
        return

    grouped = []
    i = 0
    while i < len(cmd):
        token = cmd[i]

        # group "python dnasight-cmd.py run"
        if (
                token == "python"
                and i + 2 < len(cmd)
                and cmd[i + 1].endswith(".py")
                and not cmd[i + 2].startswith("-")
        ):
            grouped.append(f"{token} {shlex.quote(cmd[i + 1])} {shlex.quote(cmd[i + 2])}")
            i += 3
            continue

        # group flags and their immediate values
        if token.startswith("-") and i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
            if sys.platform.startswith('win'):
                grouped.append(f"{token} {subprocess.list2cmdline([cmd[i + 1]])}")
            else:
                grouped.append(f"{token} {shlex.quote(cmd[i + 1])}")
            i += 2
            continue

        grouped.append(shlex.quote(token))
        i += 1

    formatted = " \\\n    ".join(grouped)
    dpg.set_value("preview", formatted)


def _update_pixel_csv_enabled():
    """Disable the CSV group if 'Pixel size' is non-empty, re-enable if empty."""
    val = str(dpg.get_value("constant_pixel_size") or "").strip()
    if val != "":
        dpg.disable_item("pixel_size_csv")
    else:
        dpg.enable_item("pixel_size_csv")


def on_constant_pixel_size_change(sender=None, app_data=None, user_data=None):
    _update_pixel_csv_enabled()
    preview_update()


def on_any_change(sender=None, app_data=None, user_data=None):
    # Update which options are shown in Module section:
    for module in ["dna_seg", "cluster_seg", "cluster_quant",
                   "dna_quant", "loop_quant", "geom_features"]:
        if dpg.get_value(module):
            dpg.show_item(module + '_options')
        else:
            dpg.hide_item(module + '_options')

    # Calibration
    rows = get_table_rows("calib_table")
    if len(rows) == 0:
        dpg.enable_item("nm_per_bp")
    else:
        dpg.disable_item("nm_per_bp")

    _update_pixel_csv_enabled()

    # Update command preview:
    preview_update()


# =========================
# Running & logging
# =========================
def start_process():
    try:
        dpg.configure_item("run_btn", enabled=False)
    except Exception:
        pass

    if STATE["running"]:
        try:
            dpg.configure_item("run_btn", enabled=True)
        except Exception:
            pass

        log_line("Already running.\n")
        return

    check_and_create_pixel_size_csv_constant()

    STATE["running"] = True

    cmd, err = build_command()
    if err:
        log_line(f"[Invalid inputs] {err}\n")
        return

    dpg.set_value("log", "")
    STATE["log"] = ""

    def runner():
        STATE["log_queue"].put(f"Starting dnasight...\n")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True
            )
            STATE["proc"] = proc
            for line in proc.stdout:
                STATE["log_queue"].put(line)
                if not STATE["running"]:
                    break
            proc.wait()
            STATE["log_queue"].put(f"\n--- Process exited with code {proc.returncode} ---\n")

        except FileNotFoundError:
            STATE["log_queue"].put(
                "Error: Could not start process (python or script not found).\n"
                + traceback.format_exc()
            )
        except Exception:
            STATE["log_queue"].put(
                "Error while running:\n" + traceback.format_exc()
            )

        finally:
            STATE["running"] = False
            STATE["proc"] = None

            try:
                dpg.configure_item("run_btn", enabled=True)
            except Exception:
                pass

    t = threading.Thread(target=runner, daemon=True)
    STATE["proc_thread"] = t
    t.start()


def stop_process():
    if STATE["proc"] and STATE["running"]:
        try:
            STATE["proc"].terminate()
            STATE["running"] = False
            log_line("\n--- Termination requested ---\n")
        except Exception as e:
            log_line(f"\nError stopping process: {e}\n")
    else:
        log_line("No active process.\n")

    try:
        dpg.configure_item("run_btn", enabled=True)
    except Exception:
        pass


def wrap_text(text, width=100):
    return text
    # return '\n'.join(text[i:i+width] for i in range(0, len(text), width))


def log_line(s: str):
    STATE["log"] = STATE["log"] + s
    dpg.set_value("log", wrap_text(STATE["log"]))


def log_poller_callback(sender=None, app_data=None, user_data=None):
    drained = False
    while not STATE["log_queue"].empty():
        try:
            ln = STATE["log_queue"].get_nowait()
            log_line(ln)
            drained = True
        except queue.Empty:
            break
    if drained:
        try:
            dpg.configure_item("log_area", scroll_y=1e9)
        except Exception:
            pass


# =========================
# Portable timer shim
# =========================
_POLL_INTERVAL_FRAMES = 12  # ~0.2s at ~60 FPS


def _poller_tick(sender=None, app_data=None, user_data=None):
    log_poller_callback()
    # reschedule next tick
    dpg.set_frame_callback(dpg.get_frame_count() + _POLL_INTERVAL_FRAMES, _poller_tick)


def start_portable_timer():
    """Use add_timer if available; otherwise fall back to frame callbacks."""
    if hasattr(dpg, "add_timer"):
        try:
            dpg.add_timer(callback=log_poller_callback, delay=0.2, user_data=None, tag="log_timer")
            try:
                dpg.configure_item("log_timer", start=True)
            except Exception:
                pass
            return
        except Exception:
            pass
    # fallback that works across versions
    dpg.set_frame_callback(dpg.get_frame_count() + _POLL_INTERVAL_FRAMES, _poller_tick)


# =========================
# File dialogs
# =========================
def open_folder_dialog(target_id):
    dpg.configure_item("folder_dialog", show=True)
    dpg.set_item_user_data("folder_dialog", target_id)


def folder_selected(sender, app_data):
    # Prefer the explicit full path Dear PyGui gives us
    path = app_data.get("file_path_name")
    if not path:
        # Fallback to the first selection value if needed
        sel = app_data.get("selections", {})
        path = next(iter(sel.values()), None)

    target = dpg.get_item_user_data("folder_dialog")
    if target and path:
        dpg.set_value(target, str(Path(path).resolve()))
        preview_update()


def open_file_dialog(target_id):
    dpg.configure_item("file_dialog", show=True)
    dpg.set_item_user_data("file_dialog", target_id)


def show_dialog(dialog_tag):
    dpg.configure_item(dialog_tag, show=True)


def create_pixel_size_csv():
    print("Creating pixel size CSV...")  # Placeholder for actual implementation
    folder_path = dpg.get_value("in_folder")
    print("Input folder:", folder_path)

    # Get all .tif file names
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Create a DataFrame with filenames and pixelsize initialized to zero
    df = pd.DataFrame({
        'filename': tif_files,
        'pixelsize': [0] * len(tif_files)
    })

    # Save the DataFrame to a CSV file
    os.makedirs(dpg.get_value("out_folder"), exist_ok=True)
    output_csv = folder_path + "/" + dpg.get_value("out_folder") + "/pixel_size.csv"  # Replace with your desired path
    print("Output CSV path:", output_csv)
    os.makedirs(os.path.join(folder_path, dpg.get_value("out_folder")), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"CSV file created with {len(tif_files)} entries and saved to {output_csv}")


def check_and_create_pixel_size_csv_constant():
    if dpg.get_value("pixel_csv") != "":
        return

    if dpg.get_value("constant_pixel_size") == "":
        return

    if float(dpg.get_value("constant_pixel_size")) <= 0.0001:
        return

    print("Creating pixel size CSV...")  # Placeholder for actual implementation
    folder_path = dpg.get_value("in_folder")
    print("Input folder:", folder_path)

    # Get all .tif file names
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Create a DataFrame with filenames and pixelsize initialized to zero
    df = pd.DataFrame({
        'filename': tif_files,
        'pixelsize': [dpg.get_value("constant_pixel_size")] * len(tif_files)
    })

    # Save the DataFrame to a CSV file
    # output_csv = folder_path + "/" + dpg.get_value("out_folder") + "/pixel_size.csv"  # Replace with your desired path
    out_dir = folder_path + "/" + dpg.get_value("out_folder")
    os.makedirs(out_dir, exist_ok=True)
    output_csv = out_dir + "/pixel_size.csv"
    print("Output CSV path:", output_csv)
    df.to_csv(output_csv, index=False)

    print(f"CSV file created with {len(tif_files)} entries and saved to {output_csv}")


def file_selected_to_target(sender, app_data, target_id):
    # Prefer Dear PyGui's explicit full path
    path = app_data.get("file_path_name")
    if not path:
        sel = app_data.get("selections", {})
        path = next(iter(sel.values()), None)

    if target_id and path:
        dpg.set_value(target_id, str(Path(path).resolve()))
        preview_update()


def file_selected(sender, app_data):
    path = app_data.get("file_path_name")
    if not path:
        sel = app_data.get("selections", {})
        path = next(iter(sel.values()), None)

    target = dpg.get_item_user_data("file_dialog")
    if target and path:
        dpg.set_value(target, str(Path(path).resolve()))
        preview_update()


# =========================
# Presets
# =========================
def save_preset_cb():
    data = {
        "stages": {
            "dna_seg": dpg.get_value("dna_seg"),
            "cluster_seg": dpg.get_value("cluster_seg"),
            "cluster_quant": dpg.get_value("cluster_quant"),
            "dna_quant": dpg.get_value("dna_quant"),
            "loop_quant": dpg.get_value("loop_quant"),
            "geom_features": dpg.get_value("geom_features"),
        },
        "paths": {
            "folder": dpg.get_value("in_folder"),
            "output": dpg.get_value("out_folder"),
            # "unet": dpg.get_value("unet_model"),
            "pixel_csv": dpg.get_value("pixel_csv"),
        },
        "dna_segmentation": {
            "threshold": float(dpg.get_value("dna_seg_threshold")),
        },
        "cluster_segmentation": {
            "model": dpg.get_value("cluster_model"),
            "large": {
                "threshold_factor": float(dpg.get_value("cluster_large_threshold")),
                "dilation_foreground": int(dpg.get_value("cluster_large_dilate_fg")),
                "dilation_background": int(dpg.get_value("cluster_large_dilate_bg")),
                "min_area": int(dpg.get_value("cluster_large_min_area")),
                "beta": float(dpg.get_value("cluster_large_beta")),
            },
            "small": {
                "diameter": int(dpg.get_value("cluster_small_diameter")),
                "minmass": float(dpg.get_value("cluster_small_minmass")),
                "min_area_filter": float(dpg.get_value("cluster_small_min_area")),
                "max_area_filter": float(dpg.get_value("cluster_small_max_area")),
            },
        },
        "cluster_quant": {
            "dna_protein_dilation": int(dpg.get_value("dna_protein_dilation")),
        },
        "dna_quant": {
            "min_area": int(dpg.get_value("dq_min_area")),
            "exclude_edge": bool(dpg.get_value("dq_exclude_edge")),
            "overlay": bool(dpg.get_value("dq_overlay")),
        },
        "loop_quant": {
            "min_length": int(dpg.get_value("lq_min_len")),
        },
        "geom_features": {
            # "dilation_radius": int(dpg.get_value("gf_dilate")),
            # "do_skeletonize": bool(dpg.get_value("gf_skel")),
            "min_pixels": int(dpg.get_value("gf_min_px")),
            # "cluster_eps": float(dpg.get_value("gf_eps")),
            "exclude_edge_touching": bool(dpg.get_value("gf_exclude_edge")),
            "bend_angle_deg": float(dpg.get_value("gf_bend_angle")),
            "bend_min_span_px": float(dpg.get_value("gf_bend_span_px")),
            "bend_span_nm_ref": float(dpg.get_value("gf_bend_span_nm")),
            "curvature_smoothing": int(dpg.get_value("gf_curvature_smoothing")),
        },
        "calibrations": get_table_rows("calib_table"),
    }
    path = dpg.get_value("preset_path").strip() or "preset.json"
    try:
        Path(path).write_text(json.dumps(data, indent=2))
        log_line(f"Saved preset to {path}\n")
    except Exception as e:
        log_line(f"Failed to save preset: {e}\n")


def load_preset_cb():
    path = dpg.get_value("preset_path").strip() or "preset.json"
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:
        log_line(f"Failed to load preset: {e}\n")
        return

    s = data.get("stages", {})
    dpg.set_value("dna_seg", s.get("dna_seg", False))
    dpg.set_value("cluster_seg", s.get("cluster_seg", False))
    dpg.set_value("cluster_quant", s.get("cluster_quant", False))
    dpg.set_value("dna_quant", s.get("dna_quant", False))
    dpg.set_value("loop_quant", s.get("loop_quant", False))
    dpg.set_value("geom_features", s.get("geom_features", False))

    p = data.get("paths", {})
    dpg.set_value("in_folder", p.get("folder", ""))
    dpg.set_value("out_folder", p.get("output", ""))
    # dpg.set_value("unet_model", p.get("unet", ""))
    dpg.set_value("pixel_csv", p.get("pixel_csv", ""))

    ds = data.get("dna_segmentation", {})
    dpg.set_value("dna_seg_threshold", float(ds.get("threshold", 0.7)))

    cs = data.get("cluster_segmentation", {})
    dpg.set_value("cluster_model", cs.get("model", "large"))
    large = cs.get("large", {})
    dpg.set_value("cluster_large_threshold", float(large.get("threshold_factor", 1.5)))
    dpg.set_value("cluster_large_dilate_fg", int(large.get("dilation_foreground", 5)))
    dpg.set_value("cluster_large_dilate_bg", int(large.get("dilation_background", 10)))
    dpg.set_value("cluster_large_min_area", int(large.get("min_area", 200)))
    dpg.set_value("cluster_large_beta", float(large.get("beta", 90.0)))
    small = cs.get("small", {})
    dpg.set_value("cluster_small_diameter", int(small.get("diameter", 11)))
    dpg.set_value("cluster_small_minmass", float(small.get("minmass", 300.0)))
    dpg.set_value("cluster_small_min_area", float(small.get("min_area_filter", 10.0)))
    dpg.set_value("cluster_small_max_area", float(small.get("max_area_filter", 2050.0)))

    cq = data.get("cluster_quant", {})
    dpg.set_value("dna_protein_dilation", int(cq.get("dna_protein_dilation", 3)))

    dq = data.get("dna_quant", {})
    dpg.set_value("dq_min_area", int(dq.get("min_area", 5)))
    dpg.set_value("dq_exclude_edge", bool(dq.get("exclude_edge", False)))
    dpg.set_value("dq_overlay", bool(dq.get("overlay", True)))

    lq = data.get("loop_quant", {})
    dpg.set_value("lq_min_len", int(lq.get("min_length", 10)))

    gf = data.get("geom_features", {})
    # dpg.set_value("gf_dilate", int(gf.get("dilation_radius", 0)))
    # dpg.set_value("gf_skel", bool(gf.get("do_skeletonize", False)))
    dpg.set_value("gf_min_px", int(gf.get("min_pixels", 5)))
    # dpg.set_value("gf_eps", float(gf.get("cluster_eps", 3.5)))
    dpg.set_value("gf_exclude_edge", bool(gf.get("exclude_edge_touching", True)))
    dpg.set_value("gf_bend_angle", float(gf.get("bend_angle_deg", 60.0)))
    dpg.set_value("gf_bend_span_px", float(gf.get("bend_min_span_px", 5.0)))
    dpg.set_value("gf_bend_span_nm", float(gf.get("bend_span_nm_ref", 10.0)))
    dpg.set_value("gf_curvature_smoothing", int(gf.get("curvature_smoothing", 15)))

    set_table_rows("calib_table", data.get("calibrations", []))
    preview_update()
    log_line(f"Loaded preset from {path}\n")


# =========================
# Calibration table helpers
# =========================
def add_calibration_row(table_id, preset=None):
    defaults = {
        "path": "",
        "dna_bp": "",
        "pixel_size_nm": "",
        "perc_low": 25,
        "perc_high": 75,
        "threshold": 0.8,
    }
    if preset:
        for k in defaults:
            val = preset.get(k)
            if val not in (None, ""):
                defaults[k] = val
    meta = dpg.get_item_user_data(table_id) or {}

    def _add_calibration_cell(key, value, r):
        if key == "path":
            with dpg.group(horizontal=True) as grp:
                path_field = dpg.add_input_text(default_value=str(value), tag=f'path_{r}', width=200,
                                                callback=on_any_change)
                # Small browse buttons
                dpg.add_button(label="Browse", width=50,
                               callback=lambda s, a, u=path_field: open_folder_dialog(f'path_{r}'))
            # IMPORTANT: return the INPUT FIELD id (so your table meta points to the text field)
            return path_field

        if key == "threshold":
            try:
                default_val = float(value)
            except (TypeError, ValueError):
                default_val = 0.8
            return dpg.add_input_float(default_value=default_val, min_value=0.0, max_value=1.0, step=0.01,
                                       format="%.3f", width=-1, callback=on_any_change)

        return dpg.add_input_text(default_value=str(value), width=-1, callback=on_any_change)

    r = int(meta.get("rows", 0))
    with dpg.table_row(parent=table_id, tag=f"{table_id}_row_{r}"):
        keys = ["path", "dna_bp", "pixel_size_nm", "perc_low", "perc_high", "threshold"]
        for c, key in enumerate(keys):
            value = defaults.get(key, "")
            field = _add_calibration_cell(key, value, r)
            meta[f"cell_{r}_{c}"] = field
        # fixed-width button column fits into the fixed column defined in the table
        del_btn = dpg.add_button(label="Remove", width=90, callback=lambda s, a, u=r: del_calibration_row(table_id, r))
        meta[f"del_{r}"] = del_btn
        meta[f"row_{r}"] = f"{table_id}_row_{r}"
    meta["rows"] = r + 1
    dpg.set_item_user_data(table_id, meta)

    on_any_change()


def del_calibration_row(table_id, row_index):
    rows_data = get_table_rows(table_id)
    rows_data = [r for idx, r in enumerate(rows_data) if idx != row_index]
    set_table_rows(table_id, rows_data)

    on_any_change()


def _pick_font_path():
    sysname = platform.system()
    if sysname == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/DejaVuSans.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    elif sysname == "Windows":
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _asset_path(rel_path: str) -> Path:
    """Resolve asset path for both dev and frozen EXE modes."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) if exe_mode else Path(__file__).parent
    return (base / rel_path).resolve()


def _try_load_logo(path: str | Path) -> str | None:
    """Load logo image into a static texture and return its texture tag."""
    global LOGO_TEX

    def resource_path(rel: str) -> Path:
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return base / rel

    p = resource_path(path)

    if not p.exists():
        print(f"[DNAsight] Logo not found: {p}")
        return None

    # Ensure a texture registry exists
    if not dpg.does_item_exist("texture_registry"):
        with dpg.texture_registry(show=False, tag="texture_registry"):
            pass

    # Try Dear PyGui loader, else Pillow fallback
    w = h = c = None
    data = None
    try:
        if hasattr(dpg, "load_image"):
            w, h, c, data = dpg.load_image(str(p))
        else:
            raise AttributeError("dpg.load_image not available")
    except Exception:
        try:
            from PIL import Image
            im = Image.open(p).convert("RGBA")
            w, h = im.size
            raw = im.tobytes()
            data = [b / 255.0 for b in raw]  # floats 0..1
            c = 4
            print(f"[DNAsight] Loaded logo via Pillow (fallback): {p}")
        except Exception as e2:
            print(f"[DNAsight] Failed to load logo: {e2}")
            return None

    try:
        mtime = p.stat().st_mtime_ns
    except Exception:
        mtime = 0
    tex_tag = f"logo_tex_{w}x{h}_{mtime}"

    if dpg.does_item_exist(tex_tag):
        try:
            dpg.delete_item(tex_tag)
        except Exception:
            pass

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent="texture_registry")

    LOGO_TEX = tex_tag
    print(f"[DNAsight] Logo loaded: {p} ({w}x{h}, channels={c}) -> {tex_tag}")
    return tex_tag


def _header_logo_width_for_height(img_w: int, img_h: int, target_h: int) -> int:
    return max(1, int(round(img_w * (target_h / float(img_h)))))


# =========================
# Theme
# =========================
def make_theme():
    with dpg.theme(tag="dark_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 23, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 70, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 240, 255))

        with dpg.theme_component(dpg.mvInputText, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (60, 60, 60), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (120, 120, 120), category=dpg.mvThemeCat_Core)

    with dpg.theme(tag="info_icon_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 30, 30, 255))  # light blue
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 105, 225, 255))  # darker blue
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 144, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 149, 237, 255))  # white text
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 20)  # make it circular
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)


@contextmanager
def tooltip(text):
    with dpg.group(horizontal=True):
        with dpg.group() as content_group:
            yield content_group

        icon = dpg.add_button(label="?", width=20, height=20)
        dpg.bind_item_theme(icon, "info_icon_theme")
        with dpg.tooltip(icon):
            dpg.add_text(text)


# =========================
# UI
# =========================
def make_ui():
    # Use globals so we can bind after items exist
    global SMALL_FONT, DEFAULT_FONT

    dpg.create_context()

    # ---------- Fonts (load before setup) ----------
    with dpg.font_registry():
        font_path = _pick_font_path()
        DEFAULT_FONT = SMALL_FONT = None

        if font_path:
            try:
                DEFAULT_FONT = dpg.add_font(font_path, 18)  # main UI font
                SMALL_FONT = dpg.add_font(font_path, 16)  # for preview & log
                print(f"[DNAsight] Using font: {font_path}")
            except Exception as e:
                print(f"[DNAsight] Failed to load {font_path}: {e}")
        else:
            print("[DNAsight] No external font found; using Dear PyGui default")

    # ---------- Viewport / setup ----------
    dpg.create_viewport(title=APP_TITLE, width=1600, height=980, x_pos=100, y_pos=100)
    dpg.configure_viewport(0, small_icon="", large_icon="", vsync=True)
    dpg.setup_dearpygui()

    if DEFAULT_FONT:
        dpg.bind_font(DEFAULT_FONT)

    make_theme()

    with dpg.texture_registry(show=False, tag="texture_registry"):
        pass

    # Try default, but allow override via env var if you like
    _try_load_logo(Path(LOGO_PATH_DEFAULT))

    # ---------- File dialogs ----------
    with dpg.file_dialog(directory_selector=True, show=False,
                         callback=folder_selected, tag="folder_dialog",
                         height=420, width=760):
        pass  # folder selector doesn't need extensions

    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=lambda s, a: file_selected_to_target(s, a, "pixel_csv"),
                         tag="csv_dialog", height=420, width=760):
        dpg.add_file_extension(".csv")

    # ---------- Main Window ----------
    with dpg.window(tag="Primary", label=APP_TITLE, width=1400, horizontal_scrollbar=True):
        with dpg.group(horizontal=True):
            if LOGO_TEX:
                # Query the texture size to keep aspect ratio when scaling height
                tex_w = dpg.get_item_configuration(LOGO_TEX).get("width", 24)
                tex_h = dpg.get_item_configuration(LOGO_TEX).get("height", 24)
                disp_w = _header_logo_width_for_height(tex_w, tex_h, LOGO_DISPLAY_H)
                dpg.add_image(LOGO_TEX, width=disp_w, height=LOGO_DISPLAY_H)
                dpg.add_spacer(width=6)
            # dpg.add_text("DNAsight 0.1")

        dpg.add_separator()

        with dpg.group(horizontal=True):
            # ===== LEFT COLUMN =====
            with dpg.child_window(width=1024, height=780, border=True):
                # Stages
                with dpg.child_window(height=290, border=True):
                    dpg.add_text("Stages", color=(180, 200, 255));
                    dpg.add_separator()

                    with tooltip(
                            "Whether to run DNA segmentation module.\nIf you have already run this module and have segmentation masks saved, you can skip this step."):
                        dpg.add_checkbox(label="DNA segmentation", tag="dna_seg", callback=on_any_change)

                    with dpg.tree_node(label="DNA segmentation options", tag="dna_seg_options", show=False,
                                       default_open=False):
                        with tooltip(
                                "Segmentation threshold, leave as is to use the recommended value.\nLower values result in more identified DNA, but also more false positives.\nRefer to segmentation results to choose an appropriate value."):
                            dpg.add_input_float(label="Threshold", tag="dna_seg_threshold",
                                                default_value=0.8, min_value=0.0, max_value=1.0,
                                                step=0.01, format="%.3f", callback=on_any_change)

                    with tooltip(
                            "Whether to run cluster segmentation module.\nIf you have already run this module and have segmentation masks saved, you can skip this step."):
                        dpg.add_checkbox(label="Cluster segmentation", tag="cluster_seg", callback=on_any_change)

                    with dpg.tree_node(label="Cluster segmentation options", tag="cluster_seg_options", show=False,
                                       default_open=False):
                        with tooltip(
                                "Selects which cluster segmentation approach to use.\nThe large model is optimal for larger structures of varying shape.\nThe small model is optimal for small circular objects."):
                            dpg.add_combo(label="model", tag="cluster_model", items=["large", "small"],
                                          default_value="large", callback=on_any_change)

                        with dpg.tree_node(label="Large model", default_open=False):
                            with tooltip(
                                    "Intensity threshold for detected cluster.\nIncrease to make segmentation stricter, decrease to include more pixels."):
                                dpg.add_input_float(label="Intensity threshold", tag="cluster_large_threshold",
                                                    default_value=1.5, step=0.1, format="%.2f", callback=on_any_change)
                            with tooltip(
                                    "Pixel dilation of detected clusters, which are sure to be a part of the cluster.\nIncrease to detect larger clusters, decrease to be more conservative."):
                                dpg.add_input_int(label="Foreground dilation", tag="cluster_large_dilate_fg",
                                                  default_value=5, min_value=0, callback=on_any_change)
                            with tooltip(
                                    "Pixel dilation from detected clusters, from where it is sure to be background.\nIncrease for larger clusters, decrease to be more conservative."):
                                dpg.add_input_int(label="Background dilation", tag="cluster_large_dilate_bg",
                                                  default_value=10, min_value=0, callback=on_any_change)
                            with tooltip("Minimum area in pixels that a detected cluster must have to be kept."):
                                dpg.add_input_int(label="Minimum area", tag="cluster_large_min_area",
                                                  default_value=200, min_value=0, callback=on_any_change)
                            with tooltip(
                                    "Shaping parameter for cluster segmentation.\nHigher values favor smoother regions, lower values to catch more details in segmentation outline."):
                                dpg.add_input_float(label="Shape parameter", tag="cluster_large_beta",
                                                    default_value=90.0, step=1.0, format="%.1f", callback=on_any_change)

                        with dpg.tree_node(label="Small model", default_open=False):
                            with tooltip("Approximate diameter in pixels of expected protein clusters for detection."):
                                dpg.add_input_int(label="Diameter", tag="cluster_small_diameter",
                                                  default_value=11, min_value=1, callback=on_any_change)
                            with tooltip("Minimum integrated intensity required for a detection."):
                                dpg.add_input_float(label="Intensity threshold", tag="cluster_small_minmass",
                                                    default_value=300.0, step=10.0, format="%.1f",
                                                    callback=on_any_change)
                            with tooltip("Lower bound on detected cluster area (in pixels) retained after filtering."):
                                dpg.add_input_float(label="Minimum area", tag="cluster_small_min_area",
                                                    default_value=10.0, step=1.0, format="%.1f", callback=on_any_change)
                            with tooltip("Upper bound on detected cluster area (in pixels) retained after filtering."):
                                dpg.add_input_float(label="Maximum area", tag="cluster_small_max_area",
                                                    default_value=2050.0, step=10.0, format="%.1f",
                                                    callback=on_any_change)

                    with tooltip(
                            "Whether to perform quantification of protein clusters and protein-DNA associations after segmentation.\nIf no DNA segmentation is performed, only standalone protein cluster statistics will be computed."):
                        dpg.add_checkbox(label="Cluster quantification", tag="cluster_quant", callback=on_any_change)

                    with dpg.tree_node(label="Cluster quantification options", tag="cluster_quant_options", show=False,
                                       default_open=False):
                        with tooltip("Radius in pixels to dilate cluster masks to match with DNA segmentations."):
                            dpg.add_input_int(label="Dilation", tag="dna_protein_dilation",
                                              default_value=3, min_value=0, callback=on_any_change)

                    with tooltip(
                            "Whether to compute DNA component statistics from the segmentation results.\nApply pixel size CSV to convert pixel measurements to nm.\nApply DNA calibration to convert pixel measurements to base pairs."):
                        dpg.add_checkbox(label="DNA quantification", tag="dna_quant",
                                         default_value=False, callback=on_any_change)

                    with dpg.tree_node(label="DNA quantification options", default_open=False, tag="dna_quant_options",
                                       show=False):
                        with tooltip(
                                "Smallest DNA object area (in pixels) that will be included in the quantification table."):
                            dpg.add_input_int(label="Minimum area", tag="dq_min_area",
                                              default_value=5, min_value=0, step=1, callback=on_any_change)
                        with tooltip("Discard DNA components that touch the image border."):
                            dpg.add_checkbox(label="Exclude edge touching", tag="dq_exclude_edge",
                                             default_value=False, callback=on_any_change)
                        with tooltip("Render an overlay image highlighting quantified DNA components."):
                            dpg.add_checkbox(label="Image overlay", tag="dq_overlay",
                                             default_value=True, callback=on_any_change)

                    with tooltip("Whether to quantify loop features from the segmented DNA."):
                        dpg.add_checkbox(label="Loop quantification", tag="loop_quant",
                                         default_value=False, callback=on_any_change)

                    with dpg.tree_node(label="Loop quantification options", tag="loop_quant_options", show=False,
                                       default_open=False):
                        with tooltip("Minimum loop length in pixels required for a loop to be counted."):
                            dpg.add_input_int(label="Minimum length", tag="lq_min_len",
                                              default_value=10, min_value=0, callback=on_any_change)

                    with tooltip("Whether to compute geometric features of DNA after segmentation."):
                        dpg.add_checkbox(label="Spatial organization", tag="geom_features",
                                         default_value=False, callback=on_any_change)

                    with dpg.tree_node(label="Spatial organization options", show=False, tag="geom_features_options",
                                       default_open=False):
                        with tooltip(
                                "Minimum length in pixels a segmentation must contain to be considered for quantification."):
                            dpg.add_input_int(label="Minimum length", tag="gf_min_px",
                                              default_value=5, min_value=0, callback=on_any_change)
                        with tooltip(
                                "Threshold angle in degrees which DNA curvature has to\nbe above to be counted as a strong bend."):
                            dpg.add_input_float(label="Strong bend angle", tag="gf_bend_angle",
                                                default_value=60.0, step=1.0, format="%.1f", callback=on_any_change)
                        with tooltip(
                                "Distance in px at which a bend is evaluated\nif it considered strong.\nLeave as is to use recommended value."):
                            dpg.add_input_float(label="Strong bend spand px", tag="gf_bend_span_px",
                                                default_value=5.0, step=0.1, format="%.1f", callback=on_any_change)
                        with tooltip(
                                "Distance in nm at which a bend is evaluated\nif it considered strong.\nLeave as is to use recommended value."):
                            dpg.add_input_float(label="Strong bend spand nm", tag="gf_bend_span_nm",
                                                default_value=10.0, step=0.1, format="%.1f", callback=on_any_change)
                        with tooltip(
                                "If enabled, discard quantification of geometric features of DNA that touch the image border."):
                            dpg.add_checkbox(label="Exclude edge touching", tag="gf_exclude_edge",
                                             default_value=True, callback=on_any_change)
                        with tooltip(
                                "Savitzky–Golay filter window size to smooth pixels before estimating curvature (must be odd)."):
                            dpg.add_input_int(label="Pixel smoothing",
                                              tag="gf_curvature_smoothing",
                                              default_value=15, callback=on_any_change)

                dpg.add_spacer(height=8)

                # Paths & files
                with dpg.child_window(height=150, border=True):
                    dpg.add_text("Paths & files", color=(180, 200, 255));
                    dpg.add_separator()
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            with tooltip("Directory containing DNAsight input TIFF images of AFM data."):
                                dpg.add_input_text(label="Input folder", tag="in_folder",
                                                   width=480, callback=on_any_change)
                            dpg.add_button(label="Browse", callback=lambda: open_folder_dialog("in_folder"))

                        with dpg.group(horizontal=True):
                            with tooltip("Where DNAsight outputs (segmentations, measurements) will be written."):
                                dpg.add_input_text(label="Output folder", tag="out_folder",
                                                   default_value="output", width=480, callback=on_any_change)
                            dpg.add_button(label="Browse", callback=lambda: open_folder_dialog("out_folder"))

                        with dpg.group(horizontal=True):
                            with tooltip(
                                    "Input fixed pixel size if constant for all input files.\nAlternatively input or generate empty pixel size CSV."):
                                dpg.add_input_text(
                                    label="Pixel size",
                                    tag="constant_pixel_size",
                                    default_value="",
                                    width=50,
                                    callback=on_constant_pixel_size_change,
                                    # on_enter=False  # (default) fire on every edit; set True if you want only on Enter
                                )

                            with dpg.group(tag="pixel_size_csv", horizontal=True):
                                with tooltip("Optional CSV listing per-image pixel sizes to override defaults."):
                                    dpg.add_input_text(label="Pixel size CSV", tag="pixel_csv",
                                                       width=330, callback=on_any_change)
                                dpg.add_button(label="Browse", callback=lambda: show_dialog("csv_dialog"))

                                with tooltip(
                                        "Create a pixel size CSV file listing all TIFF files in the input folder with pixel size 0 nm/pixel.\nYou can then fill in the correct pixel sizes manually."):
                                    dpg.add_button(label="Generate", callback=create_pixel_size_csv)

                dpg.add_spacer(height=8)

                # Calibrations
                with dpg.child_window(height=-1, border=True):
                    dpg.add_text("DNA calibrations", color=(180, 200, 255))
                    dpg.add_separator()

                    with dpg.group(horizontal=True):
                        with tooltip("Static nm/bp calibration factor (used if no data calibration)."):
                            dpg.add_input_text(label="nm/bp calibration", tag="nm_per_bp", default_value="0.34",
                                               width=75, callback=on_any_change)
                        dpg.add_spacer(width=25)
                        with tooltip("Append a new DNA calibration row to the table below."):
                            dpg.add_button(label="Add data calibration",
                                           callback=lambda: add_calibration_row("calib_table"))
                        with tooltip("Remove all DNA calibration rows from the table."):
                            dpg.add_button(label="Clear data calibrations",
                                           callback=lambda: set_table_rows("calib_table", []))

                    # 2) Then add the table which will fill the remaining space
                    with dpg.table(
                            tag="calib_table",
                            header_row=False,
                            resizable=True,
                            policy=dpg.mvTable_SizingStretchProp,
                            width=-1,
                            height=-1,
                            freeze_rows=1
                    ):
                        dpg.add_table_column(tag='tabel_path', init_width_or_weight=0.8)
                        dpg.add_table_column(tag='tabel_dna_bp', init_width_or_weight=1)
                        dpg.add_table_column(tag='tabel_pixel_size_nm', init_width_or_weight=1)
                        dpg.add_table_column(tag='tabel_low_pct', init_width_or_weight=1.2)
                        dpg.add_table_column(tag='tabel_high_pct', init_width_or_weight=1.2)
                        dpg.add_table_column(tag='tabel_threshold', init_width_or_weight=0.8)
                        dpg.add_table_column(tag='tabel_spacer', init_width_or_weight=0.65)

                        # --- Pseudo-header row using your tooltip() helper ---
                        with dpg.table_row():
                            with tooltip("Path to calibration data"):
                                dpg.add_text("Input folder")

                            with tooltip("Known calibration construct length in base pairs."):
                                dpg.add_text("DNA length (bp)")

                            with tooltip("Instrument pixel size (nm per pixel)."):
                                dpg.add_text("Pixel size (nm)")

                            with tooltip("Lower percentile cut used in length fitting."):
                                dpg.add_text("Lower threshold (%)")

                            with tooltip("Upper percentile cut used in length fitting."):
                                dpg.add_text("Upper threshold (%)")

                            with tooltip(
                                    "Segmentation threshold, leave as is to use the recommended value.\nLower values result in more identified DNA, but also more false positives.\nRefer to segmentation results to choose an appropriate value."):
                                dpg.add_text("Threshold")

                            dpg.add_text("")  # spacer

                    dpg.set_item_user_data("calib_table", {"rows": 0})

            # ===== RIGHT COLUMN =====
            with dpg.child_window(width=512, height=780, border=True):
                # Preview
                with dpg.child_window(height=150, border=True):
                    dpg.add_text("Command Preview", color=(180, 200, 255));
                    dpg.add_separator()
                    dpg.add_input_text(tag="preview", multiline=True, readonly=True, width=-1, height=-1)
                    if SMALL_FONT:
                        dpg.bind_item_font("preview", SMALL_FONT)

                dpg.add_spacer(height=6)

                # Run controls
                with dpg.child_window(height=90, border=True):
                    dpg.add_text("Run controls", color=(180, 200, 255));
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Run", width=100, tag="run_btn", callback=start_process)
                        dpg.add_button(label="Stop", width=100, callback=stop_process)
                        dpg.add_spacer(width=16)
                        with tooltip("Path to store or read DNAsight GUI presets (JSON format)."):
                            dpg.add_input_text(label="Preset path", tag="preset_path", width=320, hint="preset.json")
                        with tooltip("Save the current GUI configuration to the preset file path."):
                            dpg.add_button(label="Save preset", callback=save_preset_cb)
                        with tooltip("Load settings from the preset file path into the GUI."):
                            dpg.add_button(label="Load preset", callback=load_preset_cb)

                dpg.add_spacer(height=6)

                # Log
                with dpg.child_window(tag="log_area", height=-1, border=True, autosize_x=True):
                    dpg.add_text("Run log", color=(180, 200, 255));
                    dpg.add_separator()
                    dpg.add_input_text(
                        tag="log",
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=-1,
                    )
                    if SMALL_FONT:
                        dpg.bind_item_font("log", SMALL_FONT)
                        dpg.bind_theme("dark_theme")

    # ---------- init preview + timer, then run ----------
    preview_update()
    start_portable_timer()
    _update_pixel_csv_enabled()

    dpg.show_viewport()
    dpg.set_primary_window("Primary", True)
    dpg.configure_item("Primary", no_scrollbar=False, horizontal_scrollbar=True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def cmd():
    global cmd_mode
    cmd_mode = True
    make_ui()


# =========================
# Main
# =========================
if __name__ == "__main__":
    make_ui()