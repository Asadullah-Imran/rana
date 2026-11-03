# Kaggle Execution Guide for Multimodal Spatial Transcriptomics Benchmark

This repository contains benchmarking pipelines and visualization suites for 3 spatial transcriptomics & epigenomics multimodal clustering models:
1. **ARISE** (`new-scripts/arise.py`)
2. **SMART** (`new-scripts/smart.py`)
3. **SpatialGlue** (`new-scripts/spatialGlue.py`)

---

## 1. Benchmark Specifications

- **Exact Seeds (9 seeds)**:
  ```python
  SEEDS = [42, 0, 7, 123, 1234, 2022, 2023, 2024, 1337]
  ```
- **Evaluation Metrics Tracked**:
  - **Silhouette**: Silhouette Score on latent embeddings
  - **ARI**: Adjusted Rand Index (vs ground truth)
  - **NMI**: Normalized Mutual Information (vs ground truth)
  - **CHI**: Calinski-Harabasz Index
  - **DBI**: Davies-Bouldin Index
  - **Duration**: Execution runtime per seed (in seconds)
  - *(Also records AMI, Homogeneity, V-measure)*
- **Visualization Suites (Matching `visualization-add.txt`)**:
  - `*_cross_seed_linearity.png`: Side-by-side line & spread diagnostics (ARI vs Silhouette across seeds with ±1σ bands and linearity verdicts).
  - `*_spatial_domains_and_attention.png`: Unified 3-column spatial map (Ground Truth, Discovered Spatial Domains for Best ARI seed, Modality Attention Heatmap).
  - `*_latent_umap.png`: Unified 3-column latent UMAP manifold (Ground Truth, Discovered Domains, Modality Attention Heatmap).
  - `*_global_boxplot_comparison.png`: Metric distributions across datasets.

---

## 2. Kaggle Setup Instructions

### Step 1: Create a Kaggle Notebook
1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand panel under **Notebook options**:
   - **Accelerator**: Select **GPU T4 x 2** or **GPU P100**.
   - **Internet**: Toggle **Internet ON** (critical for downloading datasets and installing dependencies).

---

### Step 2: Clone Repository in Kaggle
In the first notebook cell, run:
```bash
!git clone https://github.com/Asadullah-Imran/result-analysis.git
%cd result-analysis
```

---

### Step 3: Install Dependencies
Run the following cell to install R prerequisites (required for `mclust` clustering in `SMART` and `SpatialGlue`) and Python dependencies:

```bash
# 1. Install R system package & mclust library
!apt-get update -qq
!apt-get install -y -qq r-base
!R -e 'if (!requireNamespace("mclust", quietly = TRUE)) install.packages("mclust", repos="https://cloud.r-project.org")'

# 2. Install PyTorch Geometric and required Python packages
!pip install -q torch-geometric
!pip install -q -r requirements.txt
```

---

### Step 4: Run a Model

You can run any of the three models directly, with full CLI argument support for quick testing:

#### Fast Test (e.g. Dataset 0 and Seed 0):
To quickly test any model on just **Dataset 0** with **Seed 0**:
```bash
# ARISE:
!python new-scripts/arise.py --dataset 0 --seed 0

# SMART:
!python new-scripts/smart.py --dataset 0 --seed 0

# SpatialGlue:
!python new-scripts/spatialGlue.py --dataset 0 --seed 0
```

#### Run Specific Datasets and Seeds:
```bash
# Run datasets 0 and 1 with seeds 42 and 0:
!python new-scripts/arise.py --dataset_indices 0 1 --seeds 42 0
```

#### Run Full Benchmark (All 6 Datasets & 9 Seeds):
If no flags are passed, the scripts run all 6 datasets across all 9 seeds:
```bash
!python new-scripts/spatialGlue.py
!python new-scripts/smart.py
!python new-scripts/arise.py
```

| Flag | Description | Example |
|---|---|---|
| `--dataset` | Single dataset index (0 to 5) | `--dataset 0` |
| `--dataset_indices` | Space-separated dataset indices | `--dataset_indices 0 1 2` |
| `--seed` | Single seed | `--seed 0` |
| `--seeds` | Space-separated seeds | `--seeds 42 0 7` |
| `--output_dir` | Custom output folder | `--output_dir my_test_run` |

**Dataset Index Mapping**:
- `0`: `10x_human_lymph_node_A1`
- `1`: `10x_human_lymph_node_D1`
- `2`: `Mouse_Brain_E11_S1`
- `3`: `Mouse_Brain_E13_S1`
- `4`: `Mouse_Brain_E15_S1`
- `5`: `Mouse_Brain_E18_S1`

---

### Step 5: How to See Visuals / Images in the Logs

In Kaggle notebooks, there are two easy ways to view the generated figures right inside your notebook output/logs:

#### Method A (Recommended: Auto-display inline with `%run`):
Instead of bash `!python`, use the IPython magic `%run`:
```python
%run new-scripts/arise.py --dataset 0 --seed 0
```
*Why this is great*: `%run` executes inside the notebook session, so all figures (`*_cross_seed_linearity.png`, `*_spatial_domains_and_attention.png`, `*_latent_umap.png`) automatically render inline directly in your cell output!

#### Method B (Display saved images after `!python` execution):
If you ran `!python new-scripts/arise.py --dataset 0 --seed 0`, display all generated figures in the very next notebook cell:
```python
from IPython.display import Image, display
import glob

# Display all generated figures in notebook cell logs
for img_path in sorted(glob.glob("*_outputs/visuals/*.png")):
    print(f"\n[Figure Log] {img_path}")
    display(Image(filename=img_path, width=950))
```

---

### Step 6: Output Structure & Visualization Artifacts

After execution, all files are saved in `/kaggle/working/result-analysis/` under:

```
SpatialGlue_outputs/
├── results/
│   ├── SpatialGlue_benchmark_results.csv   # Full results across all seeds & datasets
│   ├── SpatialGlue_summary_statistics.txt  # Mean ± Std for ARI, NMI, Sil, CHI, DBI, Duration
│   ├── SpatialGlue_adata_*.h5ad            # AnnData objects for best ARI seed
│   ├── metrics_*.csv                       # Per-dataset CSV results
│   └── summary_*.txt                       # Per-dataset summary text
└── visuals/
    ├── SpatialGlue_cross_seed_linearity.png           # Linearity & spread (ARI & Sil)
    ├── SpatialGlue_spatial_domains_and_attention.png  # Spatial domains & attention maps
    ├── SpatialGlue_latent_umap.png                    # Latent UMAP manifold projection
    ├── SpatialGlue_global_boxplot_comparison.png      # Global metric boxplots
    └── seed_plots/                                    # Per-seed UMAP plots
```

---

### Step 7: Download Results from Kaggle

To package all results and figures into a single downloadable zip file, run:

```python
import shutil

# Compress all outputs for easy download
shutil.make_archive("/kaggle/working/benchmark_outputs", "zip", ".")
print("Archive created at /kaggle/working/benchmark_outputs.zip")
```

You can then download `benchmark_outputs.zip` directly from Kaggle's right-hand Output panel.

---

### Step 8: Re-rendering Visualizations Standalone

If you want to re-generate the figures from saved CSVs and `.h5ad` files without re-training:
```bash
!python new-scripts/visualizations.py --model SpatialGlue --output_dir SpatialGlue_outputs
```
