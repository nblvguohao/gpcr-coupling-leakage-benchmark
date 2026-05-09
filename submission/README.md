# Briefings in Bioinformatics — Submission Package

## Manuscript Status (2026-05-09)

- **Target**: *Briefings in Bioinformatics* (Oxford, IF ~13.4, CAS 一区)
- **Article type**: Original Research Article
- **Pages**: 17 (main text + references + figure pages)
- **Figures**: 6 main + 3 supplementary
- **References**: 45
- **LaTeX compilation**: `pdflatex → bibtex → pdflatex × 2`

## Files to Submit

### Main Manuscript
| File | Description |
|------|-------------|
| `BIOINFORMATICS_MANUSCRIPT.tex` | Main LaTeX source (BIB-formatted) |
| `BIOINFORMATICS_MANUSCRIPT.pdf` | 17 pages compiled |
| `bioinformatics_references.bib` | 45 references |
| `figures/figure1_schematic.pdf` | Figure 1: Study schematic |
| `figures/figure1_auc_comparison.pdf` | Figure 2: AUC performance |
| `figures/promiscuity_analysis.pdf` | Figure 3: Promiscuity-stratified |
| `figures/figure3_icl_alignment.pdf` | Figure 4: ICL dimension alignment |
| `figures/figure4_alphafold_ablation.pdf` | Figure 5: AlphaFold ablation |
| `figures/calibration_analysis.pdf` | Figure 6: Calibration analysis |

### Supplementary Materials
| File | Description |
|------|-------------|
| `supplementary_materials_BIB.pdf` | Expanded supplementary (tables + figures + sensitivity analysis) |

### Cover Letter
| File | Description |
|------|-------------|
| `submission_package/cover_letter/cover_letter.pdf` | Cover letter (update as needed) |

### Software
| File | Description |
|------|-------------|
| `src/gpcr_coupling/` | Python package (CLI tool) |
| `src/gpcr_coupling/predict.py` | Model architecture + predictor |
| `src/gpcr_coupling/features.py` | Feature extraction module |
| `src/gpcr_coupling/cli.py` | Command-line interface |

## Key Improvements for BIB (from previous Bioinformatics version)

1. **Key Points box** — BIB-required structured highlights
2. **Software tool** — `gpcr-coupling` CLI with predict/extract/evaluate/train commands
3. **Comprehensive benchmarking** — SVM, MLP, RF, XGBoost, Cross-Attention compared
4. **Promiscuity-stratified analysis** — Cross-attention advantage quantified by GPCR coupling complexity
5. **Calibration analysis** — Brier score, ECE, confidence stratification (critical for practical deployment)
6. **Case study** — Orphan GPCR deorphanization predictions
7. **Expanded discussion** — Tutorial, practical guidelines, community benchmark proposal
8. **References** — 18 → 45, covering all relevant domains
9. **Supplementary materials** — Extended with parameter sensitivity, training stability, computational resources
10. **Feature importance** — Integrated gradient attribution with per-group breakdown

## Compilation

```bash
pdflatex BIOINFORMATICS_MANUSCRIPT
bibtex BIOINFORMATICS_MANUSCRIPT
pdflatex BIOINFORMATICS_MANUSCRIPT
pdflatex BIOINFORMATICS_MANUSCRIPT
```

## Submission System Checklist

- [ ] Manuscript PDF uploaded
- [ ] Figures uploaded as separate PDFs
- [ ] Cover letter uploaded
- [ ] Supplementary materials uploaded
- [ ] Author names, affiliations, ORCIDs entered
- [ ] Competing interests declared
- [ ] Data availability statement confirmed
- [ ] GitHub repository public: https://github.com/nblvguohao/gpcr-coupling-leakage-benchmark
- [ ] Zenodo DOI reserved (upon acceptance)
- [ ] Software tool documentation link provided
- [ ] Suggested reviewers provided (optional)

## Manuscript Highlights

1. **Novelty**: Reformulates GPCR coupling as paired prediction (not single-protein classification)
2. **Method finding**: ICL local features must dimension-match global PLM embeddings
3. **Negative result**: AlphaFold structural features redundant with ESM-2 650M
4. **Benchmark**: LOGPSO proposed as community challenge (~0.60 AUC ceiling across all methods)
5. **Practical utility**: Well-calibrated (Brier 0.008), high-confidence subset at 99.94% accuracy
