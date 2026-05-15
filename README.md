# GPCR-G Protein Coupling Prediction — Benchmark & Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Leakage-controlled benchmark and software toolkit for predicting GPCR–G protein coupling specificity using protein language models and topology-aware feature engineering.

## Repository Structure

```
├── code/                  # Core scripts (31 files)
│   ├── cross_validation.py         # Main CV: cluster-aware + LOGPSO
│   ├── train_cross_attention.py    # Cross-attention model training
│   ├── train_baselines.py          # SVM/MLP/RF/XGBoost baselines
│   ├── extract_esm.py              # ESM-2 650M feature extraction
│   ├── extract_icl.py              # ICL2/3 feature extraction
│   ├── extract_gprotein.py         # G protein ESM extraction
│   ├── run_ablation.py             # Ablation experiments
│   ├── run_gprotein.py             # G protein experiment
│   ├── run_analysis.py             # Calibration + promiscuity analysis
│   ├── gradient_attribution.py     # Feature importance
│   ├── statistical_tests.py        # Significance tests
│   ├── build_dataset.py            # Dataset construction
│   ├── fetch_data.py               # GPCRdb data acquisition
│   ├── bootstrap_statistics.py     # Bootstrap confidence intervals
│   ├── classical_baselines.py      # Classical ML baselines
│   ├── cluster_sensitivity.py      # Cluster threshold sensitivity
│   ├── compute_prauc.py            # Precision-Recall AUC
│   ├── compute_tiered_metrics.py   # Tiered evaluation metrics
│   ├── dimension_alignment_controls.py  # ICL dimension controls
│   ├── extended_bootstrap.py       # Extended bootstrap analysis
│   ├── generate_cv_predictions.py  # CV prediction generation
│   ├── generate_mlp_predictions.py # MLP prediction generation
│   ├── gprot_onehot_ablation.py    # G protein one-hot ablation
│   ├── label_audit.py              # Data label audit
│   ├── minimal_family_classifier.py # Minimal family classifier
│   ├── fig_*.py                    # Figure generation (5 scripts)
│   └── style_config.py            # Plotting style config
├── src/gpcr_coupling/     # Python package (CLI tool)
│   ├── __init__.py
│   ├── cli.py                     # Command-line interface
│   ├── features.py                # Feature extraction module
│   └── predict.py                 # Model architecture + predictor
├── sample_data/           # Example dataset (for format reference)
│   ├── pairing_matrix_sample.csv         # Sample pairs (50 rows)
│   ├── sequence_clusters_sample.json     # Sample clusters
│   ├── ablation_results_sample.json      # Sample ablation results
│   ├── cv_results_sample.json            # Sample CV results
│   └── prauc_results_sample.json         # Sample PRAUC results
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone repo
git clone https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark.git
cd gpcr-coupling-leakage-benchmark

# Install package
pip install -e .
```

## Usage

```bash
# Predict coupling
gpcr-coupling predict --gpcr example.fasta --gprotein Gq

# Extract features
gpcr-coupling extract-features --input sequences.fasta --output-dir features/
```

## Software Tool Commands

```
predict         Predict GPCR-G protein coupling from FASTA sequences
extract-features  Extract ESM-2 and ICL features from sequences
evaluate        Evaluate model performance on labeled data
train           Train cross-attention model on custom data
```

## Data Format

The benchmark uses paired GPCR–G protein data with the following schema:

| Column | Description |
|--------|-------------|
| `gpcr_id` | UniProt accession (optionally prefixed for isoform/allele) |
| `gpcr_sequence` | Amino acid sequence |
| `g_protein_family` | Gi, Gs, Gq, G12/13 |
| `coupling` | Binary label (0 or 1) |
| `source` | Data source (gpcrdb_iuphar, local_seed) |
| `cluster_id` | 3-mer Jaccard single-linkage cluster assignment |

Sample data is provided in `sample_data/` for format reference. The full benchmark dataset is available upon request.

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
