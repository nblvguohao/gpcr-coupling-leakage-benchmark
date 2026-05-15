# GPCR-G Protein Coupling Prediction — Benchmark & Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Leakage-controlled benchmark and software toolkit for predicting GPCR–G protein coupling specificity using protein language models and topology-aware feature engineering.

## Repository Structure

```
├── code/                       # Research pipeline + CLI tool
│   ├── gpcr_coupling/          # Installable Python package
│   │   ├── __init__.py
│   │   ├── cli.py              # Command-line interface
│   │   ├── predict.py          # Cross-attention model + predictor
│   │   └── features.py         # ESM-2 + ICL feature extraction
│   ├── build_dataset.py        # GPCRdb data acquisition + pairing matrix
│   ├── extract_features.py     # ESM-2 / G protein / ICL feature extraction
│   ├── train_models.py         # Train CA, MLP, RF, XGBoost (cluster-aware CV)
│   ├── generate_predictions.py # Generate held-out CV predictions
│   ├── bootstrap.py            # Cluster-level bootstrap statistics
│   ├── ablation.py             # Control experiments + label quality audit
│   ├── make_figures.py         # All manuscript figures
│   ├── statistical_tests.py    # Paired t-test + Wilcoxon
│   └── style_config.py         # Shared matplotlib style
├── data/                       # Datasets, features, and results
├── sample_data/                # Example dataset (for format reference)
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark.git
cd gpcr-coupling-leakage-benchmark

# Install the CLI tool
pip install -e .
```

## Quick Start

```bash
# Predict coupling for a GPCR
gpcr-coupling predict --gpcr example.fasta --gprotein Gq

# Extract features from sequences
gpcr-coupling extract-features --input sequences.fasta --output-dir features/

# Evaluate predictions
gpcr-coupling evaluate --predictions output.json --labels test.csv
```

## CLI Commands

```
predict           Predict GPCR-G protein coupling from FASTA sequences
extract-features  Extract ESM-2 and ICL features from sequences
evaluate          Evaluate model performance on labeled data
train             Train cross-attention model on custom data
```

## Reproducing the Benchmark

Run the pipeline scripts in order:

```bash
cd code/

# 1. Build dataset
python build_dataset.py

# 2. Extract features (requires GPU for ESM-2 650M)
python extract_features.py --step all

# 3. Train models
python train_models.py --model all

# 4. Generate CV predictions
python generate_predictions.py --model all

# 5. Compute bootstrap statistics
python bootstrap.py --mode all

# 6. Run ablation experiments
python ablation.py --step all

# 7. Make figures
python make_figures.py all
```

Pre-computed features and model weights are included in `data/` to skip steps 2–3.

## Data Format

The benchmark uses paired GPCR–G protein data with the following schema:

| Column | Description |
|--------|-------------|
| `gpcr_id` | UniProt accession |
| `gpcr_sequence` | Amino acid sequence |
| `g_protein_family` | Gi, Gs, Gq, G12/13 |
| `coupling` | Binary label (0 or 1) |
| `source` | Data source (gpcrdb_iuphar, local_seed) |
| `cluster_id` | 3-mer Jaccard single-linkage cluster assignment |

Sample data is provided in `sample_data/` for format reference.

## Key Dependencies

- Python 3.8+
- PyTorch 2.0+
- fair-esm (ESM-2 650M embeddings)
- scikit-learn
- xgboost
- NumPy, SciPy, pandas

## Citation

If you use this benchmark or toolkit, please cite:

Lü G, Xia Y, Liu H, Zhu X, Yang S, Gu L, Wang Q. Paired prediction of GPCR-G protein coupling specificity using protein language models and topology-aware feature engineering. *Briefings in Bioinformatics*. 2026.

## License

MIT
