# Topology-Aware GPCR-G Protein Coupling Prediction

Reproducible code for the manuscript:
**"Topology-Aware GPCR-G Protein Coupling Prediction via ESM-2 Embeddings and Curated Transmembrane Annotations"**

## Overview

This repository contains the complete pipeline for reproducing the main results of the paper:

1. **ESM-2 feature extraction** (`src/extract_esm_features.py`)
2. **UniProt transmembrane annotation fetching** (`src/fetch_uniprot_tm_annotations.py`)
3. **ICL2/3 feature extraction** (`src/extract_icl_features.py`)
4. **Paired cross-validation** (`src/paired_cross_validation.py`) - Random CV, Cluster-aware CV, LOGPSO
5. **SHAP attribution analysis** (`src/shap_attribution.py`)
6. **Permutation importance** (`src/permutation_importance.py`)
7. **Figure generation** (`src/generate_figures.py`)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data

Small metadata files are included in `data/`:
- `pairing_matrix_raw.csv` - 1,639 (GPCR, G-protein-family) pairs with labels
- `sequence_clusters.json` - 387 homology clusters (30% sequence identity)
- `uniprot_topology.json` - UniProt-curated transmembrane annotations
- `extended_sequences.json` - GPCR sequences
- `gpcrdb_coupling_long.csv` - Raw GPCRdb coupling annotations
- `g_protein_esm_features.json` - Pre-computed G-protein ESM-2 features (320-d)

> **Note**: GPCR ESM-2 features are ~1.5GB and not included. Generate them with Step 1 below, or download from [releases](https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark/releases).

### Reproduction Steps

#### Step 1: Extract ESM-2 features for GPCRs

```bash
python src/extract_esm_features.py \
    --input data/extended_sequences.json \
    --output data/gpcr_esm_features.json \
    --model esm2_t6_8M_UR50D
```

This generates 320-dimensional mean-pooled ESM-2 embeddings for all GPCR sequences.
For 650M model (1280-d), use `--model esm2_t33_650M_UR50D`.

#### Step 2: Fetch UniProt transmembrane annotations

```bash
python src/fetch_uniprot_tm_annotations.py \
    --sequences data/extended_sequences.json \
    --output data/uniprot_topology.json
```

This queries the UniProt REST API for curated transmembrane annotations.

#### Step 3: Extract ICL2/3 features

```bash
python src/extract_icl_features.py \
    --sequences data/extended_sequences.json \
    --topology data/uniprot_topology.json \
    --esm-features data/gpcr_esm_features.json \
    --output data/icl_features.json
```

This extracts:
- ICL2/3 physicochemical statistics (16-d): length, hydrophobicity, charge, etc.
- ICL2/3 local ESM embeddings (640-d): mean-pooled ESM tokens within exact loop boundaries

#### Step 4: Run cross-validation

```bash
python src/paired_cross_validation.py \
    --pairing data/pairing_matrix_raw.csv \
    --clusters data/sequence_clusters.json \
    --gpcr-features data/gpcr_esm_features.json \
    --g-protein-features data/g_protein_esm_features.json \
    --icl-features data/icl_features.json \
    --output results/cv_results.json
```

This runs three evaluation protocols:
- **Random CV**: 5-fold stratified cross-validation
- **Cluster-aware CV**: Homology-cluster-based splitting (no sequence leakage)
- **LOGPSO**: Leave-one-G-protein-family-out

#### Step 5: SHAP attribution analysis

```bash
python src/shap_attribution.py \
    --pairing data/pairing_matrix_raw.csv \
    --gpcr-features data/gpcr_esm_features.json \
    --g-protein-features data/g_protein_esm_features.json \
    --icl-features data/icl_features.json \
    --output-dir results/shap
```

#### Step 6: Permutation importance

```bash
python src/permutation_importance.py \
    --pairing data/pairing_matrix_raw.csv \
    --gpcr-features data/gpcr_esm_features.json \
    --g-protein-features data/g_protein_esm_features.json \
    --icl-features data/icl_features.json \
    --output results/permutation_importance.json
```

#### Step 7: Generate figures

```bash
python src/generate_figures.py \
    --cv-results results/cv_results.json \
    --permutation results/permutation_importance.json \
    --shap-dir results/shap \
    --output-dir figures/
```

## Repository Structure

```
.
├── data/                          # Small metadata files (included)
│   ├── pairing_matrix_raw.csv
│   ├── sequence_clusters.json
│   ├── uniprot_topology.json
│   ├── extended_sequences.json
│   ├── gpcrdb_coupling_long.csv
│   └── g_protein_esm_features.json
├── src/                           # Source code
│   ├── extract_esm_features.py
│   ├── fetch_uniprot_tm_annotations.py
│   ├── extract_icl_features.py
│   ├── paired_cross_validation.py
│   ├── shap_attribution.py
│   ├── permutation_importance.py
│   └── generate_figures.py
├── figures/                       # Output directory for figures
├── results/                       # Output directory for results
├── requirements.txt
└── README.md
```

## Key Parameters

- **SVM**: RBF kernel, C=10.0, class_weight='balanced'
- **Homology threshold**: 30% sequence identity (CD-HIT)
- **ESM-2 model**: esm2_t6_8M_UR50D (320-d) / esm2_t33_650M_UR50D (1280-d)
- **G-protein families**: Gq, Gi, Gs, G12/13

## Citation

If you use this code or dataset, please cite:

```
Lü G, Xia Y, Liu H, et al. Topology-Aware GPCR-G Protein Coupling Prediction 
via ESM-2 Embeddings and Curated Transmembrane Annotations. 
Briefings in Bioinformatics. 2026.
```

## License

This code is provided for academic research purposes.

## Contact

- Corresponding authors: wqy@ahau.edu.cn; glc@ahau.edu.cn
- Issues: https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark/issues
