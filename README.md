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

1. Select an input image (or folder)
2. Set pixel size / calibration options
3. Run segmentation
4. Run analysis modules
5. Export results

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

DNAsight can generate:

- CSV summaries (per molecule / per image / per condition)
- Overlay images (segmentation outlines, masks, skeletons, labels)
- Plots (optional, depending on workflow)

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
