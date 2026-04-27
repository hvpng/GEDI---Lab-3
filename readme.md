# GEDI - Lab 3 Clustering Reproduction

A lightweight, fully reproducible clustering reproduction package implementing the **GEDI** (Generative Energy-Based Deep Clustering) algorithm from scratch for educational purposes.

## Overview

This project provides a complete re-implementation of the GEDI clustering algorithm including:
- **Core Models**: GEDI deep clustering and baseline clustering methods
- **Metrics**: Comprehensive clustering evaluation metrics (ACC, NMI, ARI, Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Experiments**: Synthetic datasets, image benchmarks , and text clustering
- **Analysis**: Ablation studies and comparative evaluations

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # On Windows
# or: source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

Execute notebooks in order from the project root:

```bash
# Method 1: Using VS Code or Jupyter Lab (interactive)
jupyter lab notebooks/

# Method 2: Command line execution
jupyter nbconvert --to notebook --execute notebooks/01_main_reproduction.ipynb --output notebooks/01_main_reproduction.ipynb
```

## Project Structure

```
├── src/                          # Source code
│   ├── model.py                 # GEDI model & clustering algorithms
│   ├── metrics.py               # Evaluation metrics
│   ├── utils.py                 # Utilities & data loaders
│   └── __init__.py
│
├── notebooks/                    # Jupyter experiments
│   ├── 01_main_reproduction.ipynb      # Core GEDI vs baselines (Synthetic + SVHN)
│   ├── 02_ablation_study.ipynb         # Loss component ablation analysis
│   ├── 03_new_dataset_evaluation.ipynb # Evaluation on new datasets
│   └── 04_text_clustering.ipynb        # Text clustering experiments
│
├── data/                         # Dataset storage (auto-downloaded)
│   ├── test_32x32.mat
│   ├── train_32x32.mat
│   └── FashionMNIST/
│   
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── paper/                        # Reference materials
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.26, <2.0 | Numerical computing & array operations |
| **scipy** | ≥1.11 | Scientific functions (distance, optimization) |
| **pandas** | ≥2.2 | Data manipulation & result aggregation |
| **scikit-learn** | ≥1.5 | Clustering algorithms & metrics |
| **matplotlib** | ≥3.8 | Data visualization & plotting |
| **seaborn** | ≥0.13 | Enhanced visualization styling |
| **jupyter** | ≥1.1 | Interactive notebook environment |
| **torch** | ≥2.2 | Deep learning framework (GEDI model) |
| **torchvision** | ≥0.17 | Dataset loading & model utilities |
| **tqdm** | ≥4.66 | Progress bars for training loops |
| **sentence-transformers** | ≥2.6 | Text embeddings (notebook 04) |

## Notebooks Overview

### 📊 Notebook 1: Main Reproduction
**File**: `notebooks/01_main_reproduction.ipynb`

Reproduces core GEDI results from the original paper:
- Synthetic datasets (moons, circles)
- SVHN image dataset experiments
- Comparison: GEDI vs JEM vs Barlow vs SwAV
- Reference values from Table 3 of the paper

### 🔬 Notebook 2: Ablation Study
**File**: `notebooks/02_ablation_study.ipynb`

Analyzes contribution of each loss component:
- Invariance loss ($L_{INV}$)
- Prior loss ($L_{PRIOR}$)
- Contrastive-divergence loss ($L_{GEN}$)
- Impact on clustering performance

### 📈 Notebook 3: New Dataset Evaluation
**File**: `notebooks/03_new_dataset_evaluation.ipynb`

Evaluates GEDI on additional datasets not in the original paper:
- CIFAR-10 & CIFAR-100 image clustering
- Custom or transfer-learning scenarios
- Generalization assessment

### 📝 Notebook 4: Text Clustering
**File**: `notebooks/04_text_clustering.ipynb`

Applies clustering to text data using sentence embeddings.

## Data

### Automatic Download
Image datasets are automatically downloaded on first run:
- **SVHN**: ~400 MB → `data/SVHN/`
- **CIFAR-10**: ~170 MB → `data/cifar-10-batches-py/`
- **CIFAR-100**: ~170 MB → `data/cifar-100-python/`
- **FashionMNIST**: Auto-downloaded → `data/FashionMNIST/`

### Pre-loaded Data
- `data/train_32x32.mat` - Pre-processed SVHN training data
- `data/test_32x32.mat` - Pre-processed SVHN test data

### Synthetic Data
Moons and circles datasets are generated on-the-fly via scikit-learn (no files needed).

## Core Implementation

### Key Modules

**src/model.py**
- `GEDIModel`: Main PyTorch model with encoder, projector, and cluster centers
- `train_gedi()`: Full training loop with energy-based learning
- `gedi_predict()`: Hard cluster assignment
- `run_clustering_suite()`: Baseline comparison (KMeans, Spectral, etc.)
- `run_ablation_study()`: Loss component analysis

**src/metrics.py**
- `clustering_accuracy()`: Hungarian matching (ACC)
- `normalized_mutual_info_score()`: Mutual information (NMI)
- `adjusted_rand_score()`: Adjusted Rand Index (ARI)
- `silhouette_score()`: Silhouette coefficient
- `davies_bouldin_score()`: Davies-Bouldin index
- `calinski_harabasz_score()`: Calinski-Harabasz index

**src/utils.py**
- Random seed initialization
- Synthetic dataset generation
- Data loaders for all benchmarks
- Reference score retrieval from paper

## Important Notes

⚠️ **Path Resolution**: Always run notebooks from the project root directory, not from the `notebooks/` folder. The code automatically detects and adjusts for the working directory.

🔒 **Reproducibility**: Random seed is fixed at 42 throughout the project — all results are deterministic and reproducible.

📋 **Results Reporting**: Results are separated into two tables:
- **Table 1 (Paper Reproduction)**: GEDI vs JEM/Barlow/SwAV (reference values from paper)
- **Table 2 (Baseline Comparison)**: GEDI vs KMeans/Spectral/GaussianMixture/Agglomerative

## Troubleshooting

**Issue**: Import errors or missing modules
- **Solution**: Ensure virtual environment is activated and `pip install -r requirements.txt` completed successfully

**Issue**: CUDA errors (if GPU available)
- **Solution**: CPU-only mode is default. For GPU, modify `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` in notebooks

**Issue**: Data download timeout
- **Solution**: Datasets can be pre-downloaded manually and placed in the `data/` folder

## References

- Original GEDI paper: [Reference details]
- Lab assignment details: See `notes.txt`
- Implementation notes: See `notes.txt`

## Author

Implementation for educational purposes (Coursework Lab 3)
