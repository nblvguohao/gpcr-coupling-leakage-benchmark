# Submission Package: GPCR–G Protein Coupling Prediction

**Target Journal**: *Bioinformatics* (Methods track) or *Briefings in Bioinformatics*  
**Manuscript File**: `MANUSCRIPT.md`  
**Submission Date**: 2026-04-17

---

## 1. Cover Letter (Draft)

---

**To the Editors of *[Journal Name]*:**

We are pleased to submit our manuscript entitled **"Large-scale paired prediction of GPCR–G protein coupling specificity via cross-attention and protein language models"** for consideration as a [Methods/Research Article] in *[Journal Name]*.

G protein-coupled receptors (GPCRs) and heterotrimeric G proteins constitute the largest signal-transduction machinery in eukaryotes, yet accurate computational prediction of their coupling specificity remains challenging. Existing methods typically reduce the task to single-protein classification, neglecting the G protein partner and lacking rigorous homology-aware validation. In this study, we reformulate the problem as a paired (GPCR, G protein family) prediction task over a curated dataset of 1,639 experimentally annotated pairs spanning 431 GPCRs and 4 G protein families.

Our key findings are threefold. First, a cross-attention deep neural network coupled with ESM-2 650M embeddings achieves a cluster-aware CV AUC of 0.8619, outperforming the best SVM baseline (0.8331) by a substantial margin. Second, we identify a critical embedding-dimension alignment requirement: local intracellular-loop (ICL) features must match the global ESM-2 dimension (1280-d) to unlock the full potential of scaled protein language models. Third, through a controlled ablation of 38-dimensional AlphaFold structural descriptors (including geometric distances, dihedral angles, DSSP secondary-structure ratios, and PAE flexibility metrics), we show that these hand-crafted features provide *no incremental predictive signal* beyond the ESM-2 650M + ICL baseline, suggesting that large protein language models already implicitly encode sufficient structural information for this task.

We believe this work will be of broad interest to the *[journal's audience]* because it establishes a reproducible, large-scale benchmark for GPCR–G protein coupling prediction, delivers practical methodological insights for pairing protein language models with topology-aware local features, and provides a cautionary case against indiscriminate structural-feature augmentation.

All data and code are available in the accompanying repository for reproducibility. The manuscript is original, has not been published previously, and is not under consideration elsewhere.

We suggest the following experts as potential reviewers:
- **Dr. [Expert 1]**, [Institution], [specialty: protein language models / GPCR biology]
- **Dr. [Expert 2]**, [Institution], [specialty: computational structural biology / AlphaFold applications]
- **Dr. [Expert 3]**, [Institution], [specialty: machine learning for protein-protein interactions]

We would prefer to exclude the following individuals as reviewers due to potential conflicts of interest:
- [To be determined]

Thank you for considering our submission. We look forward to your response.

Sincerely,  
[Corresponding Author Name]  
[Affiliation]  
[Email]

---

## 2. Submission Checklist

### Manuscript Components
- [x] **Title**: Clear, informative, and reflects the paired-prediction focus
- [x] **Abstract**: Structured (Background, Methods, Results, Conclusions), ≤250 words
- [x] **Introduction**: Motivation, gap, and contributions
- [x] **Results**: 6 subsections with clear findings
- [x] **Discussion**: 4 thematic subsections + limitations
- [x] **Methods**: Dataset, features, models, evaluation, candidate selection
- [x] **References**: 15 core references (to be expanded if needed)
- [x] **Figure Legends**: All 5 figures described
- [x] **Supplementary Information**: Framework defined

### Figures
- [x] **Figure 1**: `figure1_auc_comparison.png/pdf` — Overall AUC comparison
- [x] **Figure 2**: `figure2_embedding_scale.png/pdf` — 8M vs 650M scale effect
- [x] **Figure 3**: `figure3_icl_alignment.png/pdf` — ICL dimension alignment insight
- [x] **Figure 4**: `figure4_alphafold_ablation.png/pdf` — AlphaFold negative result
- [x] **Figure 5**: `figure5_feature_group_importance.png/pdf` — Gradient-based interpretability

### Data & Code
- [x] **Dataset**: `paired_dataset/pairing_matrix_raw.csv` (1,639 pairs)
- [x] **Results JSONs**: All 4 ablation result files generated
- [x] **Candidate sets**: `wetlab_candidates_650m.json`
- [x] **Feature files**: ESM-2, ICL, AlphaFold features extracted
- [x] **Code scripts**: Training, evaluation, and figure generation scripts

### Still Needed / Optional Enhancements
- [ ] **Wet-lab validation**: Not yet performed; can be mentioned as future work
- [ ] **Per-residue attention maps**: Architecture limitation acknowledged; deferred to future work
- [ ] **Statistical significance tests**: Paired t-tests between model configurations (can be added if requested by reviewers)
- [ ] **Expanded reference list**: Add 5–10 more recent GPCR/PLM papers if targeting higher-impact journal

---

## 3. Recommended Journal Tiering

### Tier 1 (First Choice)
**Bioinformatics** (Oxford, Methods track)
- Fit: Strong methodological contribution, rigorous CV, clear ablations
- Word limit: ~4,000 words main text
- Acceptance rate: Moderate
- Impact factor: ~4.4

### Tier 2 (If rejected from Tier 1)
**Briefings in Bioinformatics** (Oxford)
- Fit: Review-like scope, but accepts original methods with strong biological context
- Requires more extensive literature positioning

### Tier 3 (Safe fallback)
**BMC Bioinformatics**
- Fit: Methodological focus, rigorous benchmarking
- Open access, moderate impact

---

## 4. Response to Anticipated Reviewer Concerns

**Q1: "The paired formulation still shows that the model relies mainly on GPCR features, not true partner-aware interaction."**
> **Response prepared in manuscript**: We explicitly acknowledge this limitation in §2.5 and §3.4. The gradient-attribution analysis shows G-protein feature sensitivity is ~5× lower than GPCR feature sensitivity. Under the current mean-pooled architecture, the task partially degenerates to GPCR-centric classification. We frame the paired formulation as a necessary but not sufficient step toward true interaction modeling, and we identify family-aware multi-task learning as a future direction.

**Q2: "How does 0.8619 AUC compare to the state of the art?"**
> **Response prepared in manuscript**: In §3.1, we directly address the comparison with Miglionico *et al.* (2025), who reported 0.87 AUC using AlphaFold3 on a different dataset. We emphasize that our study’s contribution lies not in a marginal AUC improvement but in the *rigorous paired formulation*, *large-scale cluster-aware validation*, and *systematic ablations* that reveal the redundancy of AlphaFold features when large PLMs are used.

**Q3: "Why did AlphaFold features not help?"**
> **Response prepared in manuscript**: §2.4 and §3.3 provide a detailed discussion. The gradient-attribution analysis (Fig. 5) shows AlphaFold features are actively used by the network (comparable sensitivity to ICL features), yet they do not improve AUC because ESM-2 650M already implicitly encodes the structural constraints captured by geometric descriptors.

**Q4: "LOGPSO AUC is only ~0.60. Does the model have any practical utility?"**
> **Response prepared in manuscript**: §3.4 acknowledges this limitation. The low LOGPSO performance reflects the inherent difficulty of cross-family generalization and the family-specific nature of coevolutionary signals. We frame the model as most useful for *within-family* prediction and candidate prioritization, rather than de novo cross-family extrapolation.

**Q5: "Where is the wet-lab validation?"**
> **Response prepared in manuscript**: §2.6 describes the generation of wet-lab candidate sets, and §3.4 explicitly states that experimental validation is outstanding future work. We provide a structured candidate list for BRET/Co-IP assays.

---

## 5. Action Items for Final Polish

1. [ ] Add 5–10 additional references to broaden the literature base (especially 2023–2025 PLM and GPCR papers).
2. [ ] Format manuscript to target journal's LaTeX/Word template.
3. [ ] Write a 150-word "Highlights" box if required by journal.
4. [ ] Prepare a 1-slide graphical abstract if required.
5. [ ] Run supplementary statistical tests (paired t-test on fold AUCs) if time permits.
6. [ ] Clean up repository and add a comprehensive README for reproducibility reviewers.

---

*Compiled 2026-04-17*
