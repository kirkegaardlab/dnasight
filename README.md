# DNAsight

DNAsight is a modular framework for automated segmentation and quantitative analysis of AFM images of DNA–protein complexes. It provides both:

- **DNAsight GUI** for interactive, point-and-click analysis
- **DNAsight CLI** for batch processing and reproducible pipelines

---

## Contents

- [Installation](#installation)
  - [Option 1 — Install with pip (recommended if you have Python)](#option-1--install-with-pip-recommended-if-you-have-python)
  - [Option 2 — Download the standalone app (no Python required)](#option-2--download-the-standalone-app-no-python-required)
- [Quick start](#quick-start)
- [Using the GUI](#using-the-gui)
- [Using the CLI](#using-the-cli)
- [Configuration](#configuration)
- [Example data](#example-data)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation

DNAsight can be installed in two ways:

1. **Python (pip) installation** *(recommended for macOS users who already have Python)*
2. **Standalone app download** *(recommended if you don’t want to install Python)*

### Option 1 — Install with pip (recommended if you have Python)

This option installs DNAsight as a Python package and is the easiest way to stay up to date. It also avoids macOS “unblock” steps that can occur when running downloaded applications.

#### Prerequisites

- Python (recommended: **Python 3.10+**)
- A clean virtual environment (**strongly recommended**)

#### 1) Create and activate a clean virtual environment

**macOS / Linux**

```bash
python3 -m venv dnasight-env
source dnasight-env/bin/activate
```
**Windows (PowerShell)**
```bash
python -m venv dnasight-env
.\dnasight-env\Scripts\Activate.ps1
```

#### 2) Install DNAsight
```bash
pip install dnasight
```
#### 3) Launch DNAsight
**GUI:**
```bash
dnasight-gui
```

**CLI:**
```bash
dnasight-cmd --help
```

> If the commands aren’t found, make sure your environment is activated.

---

### Option 2 — Download the standalone app (no Python required)

This option is best if you **don’t want to install Python**. Download the correct release for your operating system, unzip it, then run the GUI or CLI.

#### 1) Download the release

- Go to the GitHub **Releases** page
- Download the **macOS** or **Windows** zip (choose the one that matches your system)

#### 2) Unzip

Unzip the downloaded file. You should see a folder containing something like:

- `dnasight-gui` (GUI application)
- `dnasight-cmd` (command-line executable)
- `data/` (example datasets to test the workflow)
- `config.yaml` (configuration template)

#### 3) Run DNAsight

**GUI**

- Double click `dnasight-gui` to open DNAsight.

**CLI**

Open a terminal in that folder and run the command-line executable.

macOS (example):

```bash
./dnasight-cmd --help
```

Windows (example, PowerShell):

```bash
.\dnasight-cmd.exe --help
```

> Replace the commands above with the exact executable name on each platform if it differs.

#### macOS note: “Unblock” on first run

If macOS blocks the app the first time you try to open it:

1. Go to **System Settings → Privacy & Security**
2. Find the DNAsight warning
3. Click **Open Anyway**

---

## Quick start

This works for both installation methods.

1. Launch the **GUI** (`dnasight-gui`)
2. Load an image or folder (or use the example files in `data/`)
3. Run segmentation and analysis
4. Export results (CSV + overlays)

---

## Using the GUI

Typical workflow:

1. Choose segmentation and quantification modules, carefully adjusting settings to your experiment
   <img width="1009" height="301" alt="Screenshot 2026-04-17 at 09 59 06" src="https://github.com/user-attachments/assets/6f6a8721-52a8-4db6-95e3-5f0680373eb8" />
2. Select an input folder containing images in TIFF format and select fixed pixel size or generate and fill out pixel size csv
   <img width="1008" height="158" alt="Screenshot 2026-04-17 at 09 59 49" src="https://github.com/user-attachments/assets/429eddfd-edb4-40bd-b103-c2723f9ab79b" />
3. Set calibration options
   <img width="1009" height="146" alt="Screenshot 2026-04-17 at 10 00 17" src="https://github.com/user-attachments/assets/32dd2f8d-b85f-42b6-97b5-9776baec1c5f" />
4. Run DNAsight (press "Run" to start and "Stop" to stop the analysis before it finishes).
5. Check results in output folder

---

## Using the CLI

The CLI enables batch processing and reproducible runs.

Show CLI help:

```bash
dnasight-cmd --help
```

---

## Configuration

DNAsight can be configured with `config.yaml` (included in the standalone zip). This file stores defaults for common settings.

- If you are using the standalone app, you will see `config.yaml` next to the executables.
- If you installed via pip, DNAsight uses internal defaults and (optionally) a user-provided config.

> Add details here about where config is read from (working directory vs user directory) once finalized.

---

## Example data

The standalone download includes a `data/` folder with example datasets you can use to test the full workflow.

---

## Outputs

DNAsight generates csvs and/or image overlays and plots depending on the applied modules.

### S1: DNA segmentation

The DNA segmentation module outputs one `.tif` file for each input image containing two layers:

1. the raw image
2. the segmentation

The segmentation layer also contains the global ID of each DNA molecule in the individual masks. The output `.tif` files have `a_` added in front of the original filename to indicate that they are annotated files.

In addition, a folder called `segmentation_plots` is generated. This folder contains `.png` files with four plots for each input image and is intended to help the user assess DNA segmentation quality. These files are named in the same way as the annotated outputs.

### S2: Cluster segmentation and quantification

The cluster segmentation module outputs:

- an `.npy` file containing the segmentation mask and global ID for each cluster
- a `.png` file intended to help users assess whether the cluster segmentation parameters should be adjusted
- a `.csv` file called `segmentation_results`
- a separate `.csv` file called `cluster_quantification`

#### `segmentation_results.csv`

This file contains the following columns:

- `global_cluster_id`: global ID of each cluster within the provided files. If you run a different folder, the IDs may repeat. They are only global within the given folder.
- `file`: filename of the corresponding `.tif` file
- `local_id`: local cluster ID within the image file, used for internal reference for downstream quantifications
- `centroid_x`: x coordinate of the centroid of a detected cluster in pixels
- `centroid_y`: y coordinate of the centroid of a detected cluster in pixels
- `intensity`: summed intensity of pixels in the segmented cluster
- `area`: area of the segmented cluster in pixels

#### `cluster_quantification.csv`

This file contains the following columns:

- `cluster_id`: global ID of each cluster within the provided files. If you run a different folder, the IDs may repeat. They are only global within the given folder.
- `file`: filename of the corresponding `.tif` file
- `summed_intensity_px`: summed intensity of pixels within the cluster segmentation
- `bg_ring_mean`: local background around the segmented cluster
- `bg_corrected_summed_intensity_px`: background-corrected summed intensity of pixels
- `touches_edge_cluster`: `True` or `False` depending on whether the segmented cluster touches the image edge
- `centroid_x`: x coordinate of the centroid of a detected cluster in pixels
- `centroid_y`: y coordinate of the centroid of a detected cluster in pixels
- `pixel_size`: pixel size of the given image when applied; empty if no pixel size is given
- `cluster_area_nm2`: area of the segmented cluster in nm²; empty if no pixel size is given
- `summed_intensity_per_nm`: summed intensity of the segmented cluster normalized to pixel size; empty if no pixel size is given
- `summed_intensity_per_nm2`: summed intensity of the segmented cluster normalized to area; empty if no pixel size is given
- `bg_summed_intensity_per_nm`: local background-corrected summed intensity of the segmented cluster normalized to pixel size; empty if no pixel size is given
- `bg_summed_intensity_per_nm2`: local background-corrected summed intensity of the segmented cluster normalized to area; empty if no pixel size is given

### Q1: DNA length calculation

The DNA length calculation module outputs a folder called `overlays`, which contains `.png` files for each input image with the length of each segmented DNA component overlaid. The files are named as the annotated files, that is, with `a_` added in front of the original filename.

It also outputs a `.csv` file called `length_per_component`, which contains information about the length of each segmented DNA component.

#### `length_per_component.csv`

This file contains the following columns:

- `filename`: filename of the annotated file
- `comp_id`: global ID of each DNA molecule within the provided files. If you run a different folder, the IDs may repeat. They are only global within the given folder.
- `area_px`: area of the segmented DNA molecule in pixels
- `touches_edge_dna`: `True` or `False` depending on whether the segmented DNA molecule touches the image edge
- `length_px`: length of the segmented DNA molecule in pixels
- `length_nm`: length of the segmented DNA molecule in nm
- `length_bp`: length of the segmented DNA molecule in bp
- `length_bp_sem`: estimated error on the length of the segmented DNA molecule in bp based on the provided calibration
- `had_valid_pixel_size`: `True` or `False` indicating whether a valid pixel size was given for conversion
- `bp_calibration_used`: `True` or `False` indicating whether a calibration was applied

### Q2: DNA spatial organization

The DNA spatial organization module outputs a `.csv` file called `geometric_features_summary`, which contains information about the extracted geometric features.

#### `geometric_features_summary.csv`

This file contains the following columns:

- `filename`: filename of the annotated file
- `pixel_size`: pixel size of the given file, if provided
- `comp_id`: global ID of each DNA molecule within the provided files. If you run a different folder, the IDs may repeat. They are only global within the given folder.
- `touches_edge_dna`: `True` or `False` depending on whether the segmented DNA molecule touches the image edge
- `length_px`: length of the segmented DNA molecule in pixels
- `length_nm`: length of the segmented DNA molecule in nm
- `rg_px`: radius of gyration of the segmented DNA molecule in pixels
- `rg_nm`: radius of gyration of the segmented DNA molecule in nm
- `compaction`: reciprocal of the radius of gyration normalized to the length of the DNA molecule; unitless
- `n_branch_clusters`: number of crossings of the segmented DNA molecule
- `tortuosity`: tortuosity of the segmented DNA molecule; empty if fewer than one end was found, for example for plasmids
- `elongation`: elongation metric
- `mean_kappa_px_inv`: mean curvature of the segmented DNA molecule in pixel space
- `std_kappa_px_inv`: standard deviation of curvature of the segmented DNA molecule in pixel space
- `min_kappa_px_inv`: minimum curvature of the segmented DNA molecule in pixel space
- `max_kappa_px_inv`: maximum curvature of the segmented DNA molecule in pixel space
- `mean_kappa_nm_inv`: mean curvature of the segmented DNA molecule in real space, in 1/nm
- `std_kappa_nm_inv`: standard deviation of curvature of the segmented DNA molecule in real space, in 1/nm
- `min_kappa_nm_inv`: minimum curvature of the segmented DNA molecule in real space, in 1/nm
- `max_kappa_nm_inv`: maximum curvature of the segmented DNA molecule in real space, in 1/nm
- `no_strong_bends_px`: number of strong bends evaluated over a given pixel distance
- `no_strong_bends_nm`: number of strong bends evaluated over a given real-space distance in nm
- `lp_px`: persistence length in pixel space
- `lp_nm`: persistence length in real space, in nm
- `error`: an error message appears here if the calculation of a feature failed

### Q3: DNA loop quantification

The DNA loop quantification module outputs a folder called `loops`, which contains one `.pdf` file for each input image. Each `.pdf` shows detected loops overlaid with loop length and loop position given in pixels.

A yellow circle is used to indicate anchor points for loops where the intensity of the overlap exceeds the mean plus standard deviation of the intensity of the DNA molecule.

The module also outputs a `.csv` file containing information about each loop.

#### Loop output `.csv`

This file contains the following columns:

- `filename`: filename of the annotated file
- `comp_id`: global ID of each DNA molecule within the provided files. If you run a different folder, the IDs may repeat. They are only global within the given folder.
- `loop_index`: loop number within the image
- `loop_length_px`: length of the loop in pixels
- `raw_dist_px`: distance from the closest end to the loop, excluding loops along the way, in pixels
- `dist_incl_loops_px`: distance from the closest end to the loop, including loops along the way, in pixels
- `n_loops_added_inclusive`: number of other detected loops in the same DNA object that were added to the path length because they lie on the shortest path from a skeleton endpoint to the loop attachment point
- `n_loops_excluded_samepos`: number of other loops not counted because they are effectively at the same loop position as the current loop
- `mean_dna_intensity`: mean intensity of the full DNA molecule
- `std_dna_intensity`: standard deviation of intensity of the full DNA molecule
- `mean_anchor_intensity`: mean intensity at the loop anchor position
- `std_anchor_intensity`: standard deviation of intensity at the loop anchor position
- `loop_length_nm`: length of the loop in nm; empty if no pixel size is provided
- `raw_dist_nm`: distance from the closest end to the loop, excluding loops along the way, in nm; empty if no pixel size is provided
- `dist_incl_loops_nm`: distance from the closest end to the loop, including loops along the way, in nm; empty if no pixel size is provided
- `loop_length_bp`: length of the loop in bp; empty if no calibration is provided
- `raw_dist_bp`: distance from the closest end to the loop, excluding loops along the way, in bp; empty if no calibration is provided
- `dist_incl_loops_bp`: distance from the closest end to the loop, including loops along the way, in bp; empty if no calibration is provided
- `touches_edge_dna`: `True` or `False` depending on whether the segmented DNA molecule touches the image edge
- `attachment_x`: x coordinate of the loop anchor in pixels
- `attachment_y`: y coordinate of the loop anchor in pixels

### Q4: Protein-DNA associations

The protein-DNA associations module outputs two `.csv` files.

The first is centered on individual clusters and is called `cluster_dna_summary`. It contains the same columns as `cluster_quantification.csv`, with additional columns describing associated DNA.

#### `cluster_dna_summary.csv`

Additional columns:

- `sum_length_px`: total length of associated DNA in pixels; empty if no associated DNA
- `sum_length_nm`: total length of associated DNA in nm; empty if no associated DNA or if no pixel size is given
- `sum_length_bp`: total length of associated DNA in bp; empty if no associated DNA, if no pixel size is given, or if no calibration is applied
- `n_dna_linked`: number of separate DNA segmentations associated with the cluster
- `dna_ids_list`: list of global IDs of all associated DNA segmentations
- `touches_edge_dna`: `True` or `False` depending on whether one or more associated DNA segmentations touch the image edge

The second file is called `group_summary` and contains information about connected groups of DNA segmentations and clusters.

#### `group_summary.csv`

This file contains the following columns:

- `filename`: filename of the corresponding `.tif` file
- `n_clusters_in_group`: number of clusters in the group
- `cluster_ids`: list of global cluster IDs in the group
- `dna_ids`: list of global DNA IDs in the group
- `total_length_px`: total length of DNA in the group in pixels; empty if no DNA is present
- `total_length_nm`: total length of DNA in the group in nm; empty if no DNA is present or if no pixel size is given
- `total_length_bp`: total length of DNA in the group in bp; empty if no DNA is present, if no pixel size is given, or if no calibration is given
- `lengths_px_list`: list of DNA lengths in pixels, in the same order as the DNA IDs
- `lengths_nm_list`: list of DNA lengths in nm, in the same order as the DNA IDs; empty if no DNA is present or if no pixel size is given
- `lengths_bp_list`: list of DNA lengths in bp, in the same order as the DNA IDs; empty if no DNA is present, if no pixel size is given, or if no calibration is given

### Notes

- IDs described as global are only global within a single run or folder.
- If another folder is processed separately, IDs may repeat.
- Several outputs depend on whether pixel size and bp calibration were supplied.
- Empty fields generally indicate that the required metadata was not available or that the calculation was not applicable.

---

## Troubleshooting

**macOS: app is blocked on first open**

- Go to **System Settings → Privacy & Security**
- Click **Open Anyway** for DNAsight

**CLI command not found (pip install)**

- Make sure your virtual environment is activated
- Reinstall with `pip install dnasight` inside the environment

**Still stuck?**

Open an issue on GitHub with:

- your OS version
- how you installed DNAsight (pip vs standalone)
- the exact error message

---

## Development

For development installs (contributors):

```bash
git clone https://github.com/emilywinther/dnasight
cd dnasight
pip install -e .
```
---

## Citation

If you use DNAsight in academic work, please cite:

- DOI
---

## License

MIT

---

## Contact

For questions or bug reports, please open a GitHub issue.
