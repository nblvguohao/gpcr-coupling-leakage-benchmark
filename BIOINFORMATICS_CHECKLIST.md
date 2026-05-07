# Bioinformatics Submission Checklist

## Submission Requirements

- [ ] Manuscript formatted to Bioinformatics guidelines (max 8 pages, structured abstract)
- [ ] All authors' contact information and affiliations
- [ ] Competing interests statement
- [ ] Data availability statement
- [ ] Code availability statement with repository URL

## Code & Data Readiness

- [ ] GitHub repository created/pushed
- [ ] Zenodo DOI reserved
- [ ] All features can be regenerated (`extract_650m_meanpool.py`)
- [ ] Main experiments reproducible (`paired_cross_validation_enhanced_v2_650m.py`)
- [ ] `reproducible_package/` standalone and documented
- [ ] Multi-task model code (`train_multi_task_650m.py`) documented
- [ ] Requirements file complete

## Paper Components

- [ ] `BIOINFORMATICS_MANUSCRIPT.md` - Main manuscript text (Bioinformatics format)
- [ ] Supplementary materials with full results table
- [ ] Figures (schematic, results, comparison)
- [ ] Cover letter

## Submission Pipeline

1. Run `train_multi_task_650m.py` to get multi-task results
2. Run `run_ensemble_analysis.py` for consolidated table
3. Generate final figures with `generate_figures_for_manuscript.py`
4. Compile manuscript
5. Push code to GitHub
6. Submit via Bioinformatics online system
