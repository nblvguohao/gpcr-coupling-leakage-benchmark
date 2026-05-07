# Topology-Aware GPCR-G Protein Coupling Prediction - Submission Package

Complete submission package for the manuscript:
**"Topology-Aware GPCR--G Protein Coupling Prediction via ESM-2 Embeddings and Curated Transmembrane Annotations"**

Target journal: *Briefings in Bioinformatics*

---

## Package Structure

```
submission_package/
├── main_text/
│   ├── main_manuscript.tex       # Main manuscript LaTeX source
│   ├── main_manuscript.pdf       # Compiled PDF (9 pages)
│   └── references.bib            # Bibliography file
├── supplementary/
│   ├── supplementary_materials.tex  # Supplementary materials LaTeX
│   └── supplementary_materials.pdf  # Compiled PDF (6 pages)
├── cover_letter/
│   ├── cover_letter.tex          # Cover letter LaTeX
│   └── cover_letter.pdf          # Compiled PDF
├── figures/
│   ├── figure1_performance_comparison.{pdf,png}
│   ├── figure2_icl_importance.{pdf,png}
│   ├── figure3_shap_regions.{pdf,png}
│   ├── figure4_dataset_distribution.{pdf,png}
│   └── archive/                  # Obsolete figures from previous drafts
├── generate_figures.py           # Python script to regenerate figures
├── compile_latex.py              # Python script to compile LaTeX files
├── highlights.txt                # Journal highlights (5 bullet points)
└── README.md                     # This file
```

---

## Quick Start

### Option 1: Use Pre-generated PDFs

The figures are already generated in `figures/` directory. If you have a LaTeX distribution installed, you can compile the manuscripts:

```bash
# Compile all LaTeX files
python compile_latex.py
```

### Option 2: Full Reproduction

To regenerate all figures and compile LaTeX files:

```bash
# Step 1: Generate figures (requires matplotlib, pandas, numpy)
python generate_figures.py

# Step 2: Compile LaTeX files (requires pdflatex)
python compile_latex.py
```

---

## Requirements

### For Figure Generation

```bash
pip install matplotlib pandas numpy
```

### For LaTeX Compilation

**Windows (MiKTeX):**
- Download and install MiKTeX from https://miktex.org/
- Ensure `pdflatex` is in your PATH

**macOS (MacTeX):**
```bash
brew install --cask mactex
```

**Linux (TeX Live):**
```bash
sudo apt-get install texlive-full
```

---

## Files Description

### Main Manuscript (`main_text/`)

- **Abstract**: Large-scale pairwise prediction (1,639 pairs, 448 GPCRs), topology-aware feature engineering, rigorous cluster-aware evaluation
- **Introduction**: GPCR-G protein coupling, limitations of single-protein classification and flat embeddings
- **Methods**: Dataset construction, UniProt TM annotation pipeline, ICL2/3 feature extraction, SVM-RBF baselines
- **Results**: Cluster-aware CV AUC 0.797 -> 0.830, SHAP attention redistribution, LOGPSO feature-representation boundary
- **Discussion**: Implications, limitations (structural features, representation degeneracy, wet-lab validation)
- **Conclusion**: Summary of contributions

### Supplementary Materials (`supplementary/`)

- Pairing matrix construction details
- Cluster-aware CV algorithm
- UniProt API pipeline and ICL2/3 extraction
- Complete ablation metrics (AUC, Acc, Prec, Rec)
- LOGPSO per-family breakdown with threshold-tuning note
- Permutation-importance table for ICL statistics
- Supplementary Figures S1-S3

### Cover Letter (`cover_letter/`)

- Summary of three key contributions
- Fit for *Briefings in Bioinformatics*
- Suggested reviewers (to be filled)
- Conflict of interest statement

### Figures (`figures/`)

| Figure | Description | Format |
|--------|-------------|--------|
| Figure 1 | Performance comparison across CV strategies | PDF, PNG |
| Figure 2 | ICL2/3 permutation importance | PDF, PNG |
| Figure 3 | SHAP residue-region attention mapping | PDF, PNG |
| Figure 4 | Dataset distribution across G-protein families | PDF, PNG |

---

## Key Results Summary

| Strategy | Dim. | Random CV | Cluster-aware CV | LOGPSO |
|----------|------|-----------|------------------|--------|
| Baseline (global ESM-2) | 640 | 0.815 +/- 0.031 | 0.797 +/- 0.022 | 0.638 |
| Global + ICL2/3 stats | 656 | 0.832 +/- 0.017 | 0.810 +/- 0.030 | 0.628 |
| Global + ICL2/3 local (full) | 1,296 | 0.839 +/- 0.014 | **0.830 +/- 0.041** | 0.609 |
| ICL2/3 local only | 656 | 0.598 +/- 0.025 | 0.683 +/- 0.035 | 0.551 |

**Main Finding**: UniProt-curated transmembrane annotations enable precise ICL2/3 extraction that redirects pre-trained ESM-2 attention toward the true G-protein binding interface, raising homology-aware AUC from 0.797 to 0.830. Cross-family generalization (LOGPSO) is limited by probability degeneracy in concatenated mean-pooled embeddings when an entire G-protein family is unseen.

---

## Citation

If you use this code or dataset, please cite the final published article (DOI to be added upon acceptance).

---

## Contact

For questions about this submission package:
- Corresponding authors: wqy@ahau.edu.cn; glc@ahau.edu.cn
- Code repository: https://github.com/nblvguohao/topology-aware-gpcr-coupling

---

## License

This submission package is provided for academic review purposes.

---

*Package last updated: 2026-04-12*
