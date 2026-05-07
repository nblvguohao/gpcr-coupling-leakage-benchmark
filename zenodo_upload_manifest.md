# Zenodo Upload Manifest

> This manifest lists all files and directories to be uploaded to Zenodo for the
> GPCR-G protein coupling prediction study. A DOI will be requested upon
> acceptance and inserted into the manuscript.

## Metadata

- **Title**: Large-scale paired prediction of GPCR-G protein coupling specificity via cross-attention and protein language models
- **Authors**: [To be added]
- **Affiliation**: [To be added]
- **License**: CC BY 4.0
- **Keywords**: GPCR, G protein, ESM-2, cross-attention, protein language model, machine learning

## Files to Upload

### 1. Core Dataset & Annotations

| File | Description |
|------|-------------|
| `paired_dataset/pairing_matrix_raw.csv` | 1,639 GPCR-G protein pairs with binary coupling labels |
| `paired_dataset/sequence_clusters.json` | 387 CD-HIT sequence cluster assignments |
| `paired_dataset/wetlab_candidates_650m.json` | Prioritized wet-lab candidate predictions |

### 2. Features

| File | Description |
|------|-------------|
| `paired_dataset/g_protein_esm_features.json` | G-protein ESM-2 mean-pooled embeddings |
| `paired_dataset/icl_features_650m.json` | ICL2/3 ESM-2 and physicochemical features (650M) |
| `paired_dataset/icl_features_8m.json` | ICL2/3 ESM-2 and physicochemical features (8M) |
| `paired_dataset/alphafold_icl_features.json` | 38-dimensional AlphaFold structural descriptors |

### 3. Model Evaluation Results

| File | Description |
|------|-------------|
| `paired_dataset/paired_cv_enhanced_results.json` | SVM 8M cluster-aware CV and LOGPSO results |
| `paired_dataset/paired_cv_enhanced_v2_650m_results.json` | SVM 650M cluster-aware CV and LOGPSO results |
| `paired_dataset/paired_cross_attention_results.json` | Cross-attention 8M results |
| `paired_dataset/paired_cross_attention_650m_results.json` | Cross-attention 650M results |
| `paired_dataset/paired_baselines_650m_results.json` | MLP, Random Forest, XGBoost baseline results |
| `paired_dataset/statistical_tests_results.json` | Paired statistical test outputs |
| `shap_results_paired/global_shap_summary.json` | Global SHAP summary statistics |

### 4. Python Scripts

| File | Description |
|------|-------------|
| `train_paired_cross_attention_650m.py` | Cross-attention model training (650M) |
| `train_paired_cross_attention.py` | Cross-attention model training (8M) |
| `paired_cross_validation_enhanced_v2_650m.py` | SVM evaluation script (650M) |
| `paired_cross_validation_enhanced.py` | SVM evaluation script (8M) |
| `train_paired_baselines_650m.py` | MLP / RF / XGBoost baseline training |
| `statistical_significance_test_paired.py` | Paired statistical significance tests |
| `generate_schematic_figure.py` | Figure 1 schematic generation |
| `generate_figures_for_manuscript.py` | Main results figure generation |
| `gradient_attribution_650m.py` | Gradient-based feature importance (650M) |
| `compute_geometric_alphafold_features.py` | AlphaFold geometric feature extraction |
| `compute_pae_features.py` | PAE matrix feature extraction |
| `generate_wetlab_candidates_650m.py` | Wet-lab candidate generation |
| `shap_analysis_paired.py` | SHAP analysis for SVM baseline |

### 5. Manuscript & Documentation

| File | Description |
|------|-------------|
| `MANUSCRIPT.md` | Full manuscript draft (Markdown) |
| `MANUSCRIPT.tex` | Full manuscript draft (LaTeX, if generated) |
| `REPRODUCIBILITY_README.md` | Step-by-step reproduction instructions |
| `zenodo_upload_manifest.md` | This file |

### 6. Figures

| File | Description |
|------|-------------|
| `figures/figure1_schematic.png` | Schematic overview (300 dpi) |
| `figures/figure1_schematic.pdf` | Schematic overview (vector) |
| `figures/figure2_cluster_auc_comparison.png` | Cluster-aware CV AUC comparison |
| `figures/figure3_scaling_effect.png` | Effect of PLM scale |
| `figures/figure4_dimension_alignment.png` | ICL dimension-alignment results |
| `figures/figure5_alphafold_ablation.png` | AlphaFold ablation results |
| `figures/figure6_feature_importance.png` | Feature-group sensitivity |
| `figures/table_statistical_tests.png` | Statistical test results table |

## DOI Placeholder

- Zenodo DOI: `10.5281/zenodo.XXXXXXX` (to be inserted after upload)
