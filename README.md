# Topology-Aware GPCR-G Protein Coupling Prediction via ESM-2 Embeddings and Curated Transmembrane Annotations

**Target Journal**: *Briefings in Bioinformatics* (中科院一区, IF ~13+)

---

## Project Overview

Large-scale paired prediction of GPCR–G protein coupling specificity using scaled protein language models (ESM-2) and topology-aware features.

**Key Results**:
- **Best AUC**: 0.8619 ± 0.0249 (Cross-Attention + ESM-2 650M + ICL features)
- **Dataset**: 1,639 (GPCR, G protein family) pairs, 431 GPCRs, 387 sequence clusters
- **Novel finding**: ICL local features must match global embedding dimension
- **Negative result**: AlphaFold structural features do not improve beyond ESM-2 + ICL

---

## Project Structure

```
├── MANUSCRIPT.md / .tex / .pdf     # Manuscript source files
├── README.md                       # This file
├── real_references.bib             # BibTeX references
│
├── *.py (40 scripts)              # Core code (root level for path compatibility)
│
├── paired_dataset/                 # Main dataset (231MB)
│   ├── alphafold_*                # AlphaFold structural features
│   ├── icl_features_*             # ICL ESM embeddings
│   ├── g_protein_esm_features_*   # G protein ESM features
│   ├── pairing_matrix_raw.csv     # 1,639 pairs with labels
│   ├── sequence_clusters.json     # 387 homology clusters
│   └── cv_results/*               # Cross-validation results
│
├── submission_package/             # Journal submission package
│   ├── main_text/                 # Manuscript LaTeX
│   ├── supplementary/             # Supplementary materials
│   ├── cover_letter/              # Cover letter
│   └── figures/                   # Publication figures
│
├── reproducible_package/           # Standalone reproducible code
│   ├── src/                       # Compact reproduction scripts
│   ├── data/                      # Core data files
│   └── README.md                  # Reproduction instructions
│
├── figures/                       # Generated figures
├── results/                       # Experiment results
├── dssp_data/                     # DSSP structural data
│
├── docs/                          # Documentation
│   ├── plans/                     # Technical design documents
│   ├── reports/                   # Experiment reports
│   └── archive/                   # Old drafts and review records
│
├── src/                           # Organized code reference
│   └── legacy/                    # Obsolete/superseded scripts
│
└── bin/                           # Binary executables (DSSP, etc.)
```

---

## Key Code Files by Function

### Models & Training
| Script | Description |
|--------|-------------|
| `train_paired_cross_attention_650m.py` | Cross-Attention + 650M ESM training |
| `train_paired_baselines_650m.py` | MLP/RF/XGBoost baselines |
| `train_paired_cross_attention.py` | 8M ESM version |
| `train_gsca.py` / `train_ipl.py` / `train_ipl_v2.py` | Exploratory models |

### Cross-Validation
| Script | Description |
|--------|-------------|
| `paired_cross_validation_enhanced_v2_650m.py` | **Main CV** (paper results) |
| `paired_cross_validation_enhanced.py` / `_v2.py` | Enhanced CV variants |
| `paired_cross_validation_650m.py` | 650M CV runner |
| `paired_cross_validation.py` | 8M CV runner |
| `run_final_ablation.py` | Final ablation experiments |
| `run_gprot_650m_experiment.py` | Full G protein experiment |

### Feature Extraction
| Script | Description |
|--------|-------------|
| `extract_650m_features.py` | ESM-2 650M features |
| `extract_icl_features_650m.py` | ICL2/3 local features |
| `extract_alphafold_features_paired.py` | AlphaFold structure features |
| `extract_gprotein_650m.py` | G protein features |
| `compute_geometric_alphafold_features.py` | Geometric descriptors |
| `compute_pae_features.py` | PAE flexibility features |
| `build_paired_matrix.py` | Pairing matrix construction |

### Analysis & Interpretability
| Script | Description |
|--------|-------------|
| `gradient_attribution_650m.py` | Gradient-based feature sensitivity |
| `shap_attribution_paired.py` | SHAP analysis |
| `shap_attribution_icl.py` | ICL-specific SHAP |
| `map_shap_to_residues.py` | Residue-level mapping |
| `analyze_logpso_failure.py` | LOGPSO failure analysis |

### Figures & Results
| Script | Description |
|--------|-------------|
| `generate_figures_for_manuscript.py` | Main results figures |
| `generate_schematic_figure.py` | Figure 1 architecture schematic |
| `generate_supplementary_materials.py` | Supplementary figures/tables |
| `generate_wetlab_candidates_650m.py` | Wet-lab candidate sets |
| `statistical_significance_test_paired.py` | Statistical tests |

---

## Quick Start (Reproduction)

See `reproducible_package/README.md` for step-by-step reproduction.

```bash
# Install dependencies
pip install -r reproducible_package/requirements.txt

# Run cross-validation
python paired_cross_validation_enhanced_v2_650m.py
```

---

## Citation

```
Lü G, Xia Y, Liu H, et al. Topology-Aware GPCR-G Protein Coupling Prediction 
via ESM-2 Embeddings and Curated Transmembrane Annotations. 
Briefings in Bioinformatics. 2026.
```

---

## Key Findings

1. **Paired formulation**: Reformulating coupling prediction as (GPCR, G protein) pairwise classification is biologically more faithful
2. **Dimension alignment**: ICL local features must match ESM-2 embedding dimension (1280-d)
3. **AlphaFold redundancy**: Structural features provide no gain beyond ESM-2 650M
4. **Cross-family gap**: LOGPSO AUC ~0.60 indicates family-specific learned patterns
5. **GPCR-centric**: GPCR features dominate prediction (5-7× higher sensitivity than G protein)
