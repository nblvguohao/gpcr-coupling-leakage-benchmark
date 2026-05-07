# Reproducibility Guide: GPCR–G Protein Coupling Prediction

This repository contains all data, code, and results for the manuscript:

> **Large-scale paired prediction of GPCR–G protein coupling specificity via cross-attention and protein language models**

---

## Quick Start

All core experiments can be reproduced with Python 3.10+ and the following key dependencies:
- `torch >= 2.0`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `biopython` (for DSSP-based secondary structure analysis)
- `pydssp` (pure-Python DSSP alternative, used for geometric feature extraction)

---

## 1. Dataset

The paired dataset is located at:
```
paired_dataset/pairing_matrix_raw.csv
```

It contains 1,639 rows, each representing a (GPCR, G protein family) pair with:
- `gpcr_id`: UniProt-style identifier
- `g_protein_family`: One of {Gq, Gi, Gs, G12_13}
- `coupling`: Binary label (1 = coupling annotated, 0 = non-coupling)
- `cluster_id`: CD-HIT cluster assignment (0–386)

Sequence clusters are defined in:
```
paired_dataset/sequence_clusters.json
```

---

## 2. Feature Files

| Feature | File | Dimensions |
|---------|------|------------|
| GPCR ESM-2 8M (mean-pool) | `server_sync/extended_data/features/esm_features_100samples.json` | 320-d |
| GPCR ESM-2 650M (mean-pool) | `server_sync/extended_data/features/esm_features_650m_meanpool.json` | 1280-d |
| G protein ESM-2 8M | `paired_dataset/g_protein_esm_features.json` | 320-d |
| ICL 8M | `paired_dataset/icl_features.json` | 320-d ESM + 8 stats per loop |
| ICL 650M | `paired_dataset/icl_features_650m.json` | 1280-d ESM + 8 stats per loop |
| AlphaFold-ICL | `paired_dataset/alphafold_icl_features.json` | 38-d |

---

## 3. Running the Main Experiments

### 3.1 8M SVM ablations
```bash
python paired_cross_validation_enhanced.py
```
Output: `paired_dataset/paired_cv_enhanced_results.json`

### 3.2 650M SVM ablations
```bash
python paired_cross_validation_enhanced_v2_650m.py
```
Output: `paired_dataset/paired_cv_enhanced_v2_650m_results.json`

### 3.3 8M Cross-Attention
```bash
python train_paired_cross_attention.py
```
Output: `paired_dataset/paired_cross_attention_results.json`

### 3.4 650M Cross-Attention
```bash
python train_paired_cross_attention_650m.py
```
Output: `paired_dataset/paired_cross_attention_650m_results.json`

### 3.5 Generate report and figures
```bash
python generate_result_report.py
python generate_figures_for_manuscript.py
```
Outputs: `RESULTS_REPORT.md`, `figures/`

---

## 4. Gradient Attribution (Interpretability)

```bash
python gradient_attribution_650m.py
```
Outputs:
- `figures/figure5_feature_group_importance.png`
- `figures/gradient_attribution_650m.json`

---

## 5. Wet-Lab Candidate Generation

```bash
python generate_wetlab_candidates_650m.py
```
Output: `paired_dataset/wetlab_candidates_650m.json`

---

## 6. AlphaFold Feature Extraction (if needed)

The 38-d AlphaFold features were generated in two stages:

### Geometric features
```bash
python compute_geometric_alphafold_features.py
```
Output: `paired_dataset/alphafold_geometric_features.json`

### PAE features
```bash
python compute_pae_features.py
```
Output: `paired_dataset/alphafold_pae_features.json`

### Merge into final AlphaFold-ICL file
```bash
python compute_icl_plddt.py
```
Output: `paired_dataset/alphafold_icl_features.json`

**Note**: `compute_icl_plddt.py` requires `pydssp` to be installed (`pip install pydssp`) because the system `mkdssp` 4.x often fails on Windows due to missing CIF dictionary files.

---

## 7. Expected Results Summary

| Model | Embedding | Baseline AUC | ICL-full AUC | Alpha (38-d) AUC |
|-------|-----------|--------------|--------------|------------------|
| SVM | 8M | 0.7972 | 0.8301 | 0.8285 |
| SVM | 650M | 0.8188 | 0.8324 | 0.8287 |
| Cross-Attention | 8M | 0.8159 | 0.8247 | 0.8207 |
| Cross-Attention | 650M | 0.8378 | **0.8619** | 0.8600 |

All AUCs are cluster-aware 5-fold CV means. Standard deviations are reported in the corresponding JSON output files.

---

## 8. Manuscript Files

- `MANUSCRIPT.md` — Full manuscript draft (Abstract, Introduction, Results, Discussion, Methods, References)
- `SUBMISSION_PACKAGE.md` — Cover letter draft, submission checklist, and journal recommendations
- `RESULTS_REPORT.md` — Internal technical summary of all experiments

---

## 9. Contact

For questions regarding reproducibility, please contact the corresponding author listed in `MANUSCRIPT.md`.

---

*Last updated: 2026-04-17*
