# Paired GPCR-G Protein Coupling Prediction

**Target Journal**: *Briefings in Bioinformatics* (Oxford, IF ~13.4)

## Key Results

- **Best AUC**: 0.8619 ± 0.0249 (Cross-Attention + ESM-2 650M + ICL features, cluster-aware CV)
- **Dataset**: 1,647 pairs, 431 GPCRs, 4 G protein families, 387 sequence clusters
- **Brier score**: 0.008 (CA) vs 0.039 (SVM) — well-calibrated probabilities
- **Key finding**: ICL local features must match global ESM embedding dimension
- **Negative result**: AlphaFold structural descriptors provide no gain beyond ESM-2 650M

## Repository Structure

```
├── manuscript/           # Manuscript source + compiled PDF
│   ├── main.tex          # LaTeX source (BIB formatted)
│   ├── main.pdf          # Compiled manuscript (21 pages)
│   ├── references.bib    # 45 references
│   ├── supplementary.tex # Supplementary materials
│   └── supplementary.pdf
├── code/                 # Core scripts (18 files)
│   ├── cross_validation.py      # Main CV: cluster-aware + LOGPSO
│   ├── train_cross_attention.py # Cross-attention model training
│   ├── train_baselines.py       # SVM/MLP/RF/XGBoost baselines
│   ├── extract_esm.py           # ESM-2 650M feature extraction
│   ├── extract_icl.py           # ICL2/3 feature extraction
│   ├── run_ablation.py          # Ablation experiments
│   ├── run_gprotein.py          # G protein experiment
│   ├── run_analysis.py          # Calibration + promiscuity analysis
│   ├── gradient_attribution.py  # Feature importance
│   ├── statistical_tests.py     # Significance tests
│   ├── build_dataset.py         # Dataset construction
│   ├── fetch_data.py            # GPCRdb data acquisition
│   └── fig_*.py                 # Figure generation (5 scripts)
├── data/                 # Dataset + results (21 files)
│   ├── pairing_matrix_raw.csv   # 1,647 labeled pairs
│   ├── sequence_clusters.json   # 387 CD-HIT clusters
│   ├── gpcr_esm_features_650m.json     # GPCR ESM-2 650M embeddings
│   ├── g_protein_esm_features_650m.json # G protein ESM-2 embeddings
│   ├── icl_features_650m.json          # ICL2/3 features (1280-d)
│   ├── cv_results.json                 # Cross-validation results
│   ├── baseline_results.json           # Baseline model results
│   ├── ca_predictions.json             # Cross-attention predictions
│   └── ...
├── figures/              # Publication figures (14 files)
├── src/gpcr_coupling/    # Software tool (4 files)
├── submission/           # Cover letter + README
├── .gitignore
└── README.md
```

## Reproduction

```bash
# Clone repo
git clone https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark.git
cd gpcr-coupling-leakage-benchmark

# Reproduce main CV results
python code/cross_validation.py

# Generate figures
python code/fig_bib.py
python code/fig_manuscript.py

# Run analysis
python code/run_analysis.py
```

Pre-computed features and results are in `data/` — no GPU required for reproduction.

## Software Tool

```bash
# Install
pip install -e .

# Predict coupling
gpcr-coupling predict --gpcr example.fasta --gprotein Gq

# Extract features
gpcr-coupling extract-features --input sequences.fasta --output-dir features/
```

## Citation

Lü G, Xia Y, Liu H, Gu L, Wang Q. Paired prediction of GPCR-G protein coupling specificity using protein language models and topology-aware feature engineering. *Briefings in Bioinformatics*. 2026.

## License

MIT
