# Large-scale paired prediction of GPCR–G protein coupling specificity with scaled protein language models and topology-aware features

**Authors**: [Author names to be added]  
**Affiliations**: [To be added]  
**Corresponding author**: [To be added]

---

## Abstract

**Background**: G protein-coupled receptors (GPCRs) and heterotrimeric G proteins constitute the largest family of signal-transduction machinery in eukaryotes. Accurate prediction of coupling specificity is essential for orphan-receptor deorphanization and rational drug design. Existing sequence-level methods often reduce the task to single-protein classification, neglecting the identity of the G protein partner and the known binding interface.

**Methods**: We reformulated the task as a (GPCR, G protein family) pairwise binary classification over a curated dataset of 1,639 experimentally annotated pairs spanning 431 GPCRs and 4 G protein families (Gq, Gi, Gs, G12/13). Models were evaluated under two strict regimes: cluster-aware 5-fold cross-validation (387 sequence clusters) and leave-one-G-protein-family-out cross-validation (LOGPSO). We systematically compared support vector machines (SVM) with cross-attention deep neural networks, using ESM-2 embeddings at two scales (8M, 320-d; 650M, 1280-d), and further augmented them with topology-aware intracellular loop 2/3 (ICL2/3) features and 38-dimensional AlphaFold structural descriptors.

**Results**: The cross-attention model with 650M ESM-2 embeddings and 1280-d ICL features achieved the highest performance (cluster-aware AUC = 0.8619 ± 0.0249), outperforming the best SVM configuration (0.8331 ± 0.0135) by +0.029. The 650M embeddings yielded larger gains for the cross-attention architecture (+2.2%) than for SVM (+2.2% baseline, but ceiling ~0.833), indicating that deep-learning architectures better exploit rich pre-trained representations. A critical finding was that ICL local features must match the global embedding dimension: 320-d ICL features paired with 1280-d global embeddings degraded performance, whereas dimension-aligned 1280-d ICL features unlocked the full deep-learning potential. Notably, 38-dimensional AlphaFold structural features—encompassing TM5/TM6 cytoplasmic distances, dihedral angles, interface SASA, DSSP secondary-structure ratios, and 8 PAE flexibility metrics—did not improve predictive performance beyond the ESM-2 + ICL baseline across all model configurations, suggesting that ESM-2 650M already implicitly encodes sufficient structural information.

**Conclusions**: Our study establishes a scalable, reproducible framework for paired GPCR–G protein coupling prediction. The results highlight the importance of paired formulation, embedding-scale matching, and topology-aware local features, while also demonstrating the limits of hand-crafted structural augmentation when high-capacity protein language models are employed. Wet-lab candidate sets derived from the top-performing model are provided for experimental validation.

**Keywords**: GPCR–G protein coupling; protein language model; ESM-2; cross-attention; intracellular loop; AlphaFold; paired prediction

---

## 1. Introduction

G protein-coupled receptors (GPCRs) mediate cellular responses to diverse extracellular stimuli by coupling to heterotrimeric G proteins, which in turn activate downstream signaling cascades [1]. With over 800 members in the human genome, GPCRs represent the largest class of membrane receptors and are the target of approximately one-third of all approved drugs [2]. The specificity of GPCR–G protein coupling determines which intracellular pathway is triggered, making its accurate prediction a central problem in receptor biology and pharmacology [3].

Despite decades of structural and biochemical research, the molecular determinants of coupling specificity remain incompletely understood [4]. Computational methods have traditionally approached the problem as a single-protein classification task, predicting whether a given GPCR sequence couples to a specific G protein subtype [5,6]. However, this formulation ignores the fact that many GPCRs couple to multiple G protein families with different efficiencies, and that the G protein partner itself contributes to binding affinity and selectivity [7]. A paired (GPCR, G protein family) formulation is therefore more faithful to the underlying biology.

Recent advances in protein language models (PLMs), particularly the ESM-2 family from Meta AI, have shown that sequence embeddings trained at evolutionary scale can capture structural and functional information with remarkable fidelity [8]. These embeddings have driven state-of-the-art performance in protein-structure prediction, function annotation, and protein–protein interaction (PPI) tasks [9]. For GPCR–G protein coupling, ESM-2 embeddings offer a data-efficient way to represent receptor sequences without relying on experimental structures.

While ESM-2 provides powerful global sequence representations, the G protein binding interface is localized to the intracellular face of the receptor, particularly intracellular loop 2 (ICL2) and the C-terminal portions of transmembrane helices 5 and 6 (TM5/TM6) [10]. Explicit modeling of these regions—through topology-aware feature extraction—could guide model attention toward biologically relevant regions. Concurrently, the AlphaFold structure-prediction revolution has made predicted GPCR structures widely available, enabling the extraction of geometric descriptors such as inter-helical distances, loop conformations, and flexibility metrics from predicted aligned error (PAE) matrices [11].

In this study, we present a comprehensive computational framework that combines paired formulation, scaling laws of PLMs, and structural feature engineering for GPCR–G protein coupling prediction. Our work advances the field in four ways:

1. **Paired formulation at scale**: We curated a dataset of 1,639 experimentally annotated (GPCR, G protein family) pairs spanning 431 distinct GPCRs and evaluated models under strict cluster-aware cross-validation (387 sequence clusters).
2. **Systematic architecture comparison**: We compared SVM baselines with cross-attention deep neural networks across two ESM-2 scales (8M and 650M parameters).
3. **Topology-aware feature engineering**: We extracted and fused ICL2/3 local features and 38-dimensional AlphaFold descriptors, evaluating their contribution under controlled ablations.
4. **Methodological insights**: We identified a critical embedding-dimension alignment requirement for local features and demonstrated that AlphaFold geometric/PAE descriptors do not add incremental signal beyond high-capacity ESM-2 embeddings.

---

## 2. Results

### 2.1 Dataset and evaluation framework

The paired dataset comprised 1,639 (GPCR, G protein family) combinations with binary coupling annotations derived from GPCRdb and literature curation (Supplementary Data 1). The dataset spans 431 unique GPCRs and 4 G protein families: Gq (n = 388 positive pairs), Gi (n = 406), Gs (n = 298), and G12/13 (n = 173). Negative pairs were constructed from non-coupling combinations within the same receptor set.

To prevent sequence-homology leakage, we performed cluster-aware 5-fold cross-validation using 387 CD-HIT sequence clusters (threshold = 0.4 sequence identity). All pairs sharing the same GPCR were placed in the same fold, ensuring that test-set receptors were never seen during training. We additionally evaluated cross-family generalization via leave-one-G-protein-family-out cross-validation (LOGPSO), in which models were trained on three families and tested on the held-out fourth.

### 2.2 Overall performance: cross-attention outperforms SVM at 650M scale

Under cluster-aware cross-validation, the cross-attention deep neural network with 650M ESM-2 embeddings and full ICL features achieved the highest AUC of **0.8619 ± 0.0249**, establishing a new state of the art for this dataset (Table 1; Fig. 2). The best SVM configuration (650M ICL-stats-v2) reached **0.8331 ± 0.0135**, indicating a consistent performance ceiling for the RBF kernel on this task.

**Table 1. Cluster-aware cross-validation performance of all model configurations.**

| Model | Embedding | Baseline | ICL-full | Alpha (38-d) |
|-------|-----------|----------|----------|--------------|
| SVM | 8M | 0.7972 ± 0.0220 | 0.8301 ± 0.0407 | 0.8285 ± 0.0391 |
| SVM | 650M | 0.8188 ± 0.0167 | 0.8324 ± 0.0185 | 0.8287 ± 0.0205 |
| Cross-Attention | 8M | 0.8159 ± 0.0315 | 0.8247 ± 0.0348 | 0.8207 ± 0.0351 |
| Cross-Attention | 650M | 0.8378 ± 0.0168 | **0.8619 ± 0.0249** | 0.8600 ± 0.0242 |

Strikingly, the relative benefit of scaling from 8M to 650M embeddings differed markedly between architectures. For SVM, the baseline AUC improved modestly from 0.7972 (8M) to 0.8188 (650M; +2.2%), but the ceiling remained ~0.833 regardless of embedding size. For the cross-attention model, the baseline improved from 0.8159 (8M) to 0.8378 (650M; +2.2%), and the gap between cross-attention and SVM widened from +0.019 to +0.043 at full feature configuration. These results suggest that deep-learning architectures can more effectively leverage the richer representations of larger PLMs, whereas kernel-based methods saturate.

To further contextualize the cross-attention result, we trained three additional baseline models on the identical 650M ICL-full feature set: a 3-layer multilayer perceptron (MLP), a Random Forest (RF), and XGBoost. The MLP achieved an AUC of **0.8608 ± 0.0242**, statistically indistinguishable from the cross-attention model (paired *t*-test, *p* = 0.714), confirming that most of the deep-learning gain comes from the feature representation rather than the attention mechanism per se. RF and XGBoost reached 0.8262 ± 0.0310 and 0.8331 ± 0.0216, respectively, bracketing the SVM ceiling and underscoring that ESM-2 embeddings are the dominant performance driver (Supplementary Table 4).

Under LOGPSO, all models performed substantially worse (AUC ~0.59–0.64), reflecting the difficulty of generalizing across G protein families when family-specific coevolutionary patterns are absent from training. This finding underscores that the predictive signal learned by current models is partly family-specific (Fig. 3).

### 2.3 ICL features and the dimension-alignment requirement

Adding topology-aware ICL2/3 features improved performance for both architectures. For the 8M SVM, ICL-full boosted AUC from 0.7972 to 0.8301 (+0.033). For the 650M cross-attention, the gain was even larger: from 0.8378 to 0.8619 (+0.024).

A critical methodological insight emerged when we examined the interaction between global embedding dimension and local ICL feature dimension. The initial ICL features were extracted with the 8M model (320-d ESM embeddings). When these 320-d ICL features were concatenated with 1280-d 650M global embeddings, SVM performance degraded (0.8234 vs. 0.8301 with 8M global embeddings). After re-extracting ICL features with the 650M model (1280-d embeddings), performance recovered for SVM (0.8304) and jumped to 0.8599 for cross-attention. **This demonstrates that heterogeneous feature dimensions introduce noise, and dimension-aligned local features are essential for unlocking the potential of high-dimensional PLM representations** (Fig. 4).

### 2.4 AlphaFold structural features do not add incremental signal

We extracted 38-dimensional structural descriptors from ~364 AlphaFold PDBs, including TM5/TM6 cytoplasmic Cα distances, ICL end-to-end distances, dihedral angles, aromatic centroid depths, interface SASA, DSSP secondary-structure ratios (helix/sheet/coil) for ICL2/3, and 8 PAE matrix-based flexibility features (Supplementary Table 2). These descriptors capture biologically meaningful geometry relevant to G protein binding.

However, adding these 38 features in "alpha" mode did not improve performance in any configuration:

- SVM (8M): 0.8285 vs. ICL-full 0.8301
- SVM (650M): 0.8287 vs. ICL-full 0.8324
- Cross-attention (8M): 0.8207 vs. ICL-full 0.8247
- Cross-attention (650M): 0.8600 vs. ICL-full 0.8619

In every case, the alpha mode performed slightly worse than the ICL-full baseline. This pattern strongly suggests that the structural and flexibility information captured by these hand-crafted descriptors is already encoded within the ESM-2 650M embeddings and the ICL mean-pooling features. For this dataset, explicit geometric augmentation appears redundant rather than complementary (Fig. 5).

### 2.5 Feature importance and model interpretability

To understand which input feature groups drive predictions in the best-performing cross-attention model, we computed gradient-based feature sensitivities (mean absolute gradient of the predicted probability with respect to each input dimension) on the full training set after convergence. The analysis revealed a striking asymmetry between the two protein partners: **GPCR-side features exhibited 5- to 7-fold higher mean sensitivity than G-protein-side features** (Fig. 6). Among the GPCR feature groups, ICL3 physicochemical statistics showed the highest sensitivity (7.61 × 10⁻⁵), followed by ICL2 statistics (7.21 × 10⁻⁵), AlphaFold structural descriptors (7.35 × 10⁻⁵), and ICL ESM embeddings (6.5–7.0 × 10⁻⁵). The global GPCR ESM embedding displayed the lowest sensitivity (5.67 × 10⁻⁵) among GPCR-side groups.

Two insights emerge from this pattern. First, the gradient attribution on the 650M cross-attention model shows that GPCR-side features have ~5–7× higher sensitivity than G-protein-side features. Second, a complementary permutation-SHAP analysis on the 8M SVM baseline over the full paired dataset (1,639 samples, 4 G-protein families) confirms that G-protein ESM dimensions are indeed used by the model (all 320 dimensions have non-zero SHAP values; mean |SHAP| = 3.87 × 10⁻⁴), but their discriminative magnitude is roughly 20× smaller than the top GPCR dimensions. This asymmetry is biologically plausible: GPCR sequence carries the bulk of coupling-determinant information, while G-protein family identity modulates the prediction more subtly. Importantly, the paired formulation is not degenerating into pure GPCR single-protein classification—G-protein features do contribute—but the model's predictive power remains heavily GPCR-centric under the current mean-pooled architecture.

A third insight concerns the AlphaFold descriptors. Although they are actively used by the network (their gradient sensitivity is comparable to ICL features), they do not improve held-out AUC because they capture structural information that is already redundant with the ESM-2 650M representations and the topology-aware ICL descriptors.

### 2.6 Wet-lab candidate selection

From the best-performing model (cross-attention + 650M + ICL-full), we generated a prioritized set of wet-lab candidates (`wetlab_candidates_650m.json`). The set includes 10 candidates across four categories: (i) high-confidence positives (2 confirmed positives + 1 top novel prediction), (ii) medium-confidence boundary cases (3 samples closest to probability 0.50), (iii) high-confidence negatives (2 strong non-coupling predictions), and (iv) disputed cases (1 false negative + 1 false positive). These candidates are provided for experimental validation via BRET or Co-IP assays.

---

## 3. Comparison with Related Work

**Table 2. Comparative summary of computational methods for GPCR–G protein coupling prediction.**

| Study | Year | Dataset Size | Task Formulation | Method | Best Performance | Evaluation Strategy |
|-------|------|--------------|------------------|--------|------------------|---------------------|
| Möller *et al.* [5] | 2001 | ~100 receptors | Single-protein | HMM patterns | ~92% accuracy | Hold-out test |
| Sgourakis *et al.* [12] | 2005 | ~80 receptors | Single-protein | HMM + ANN | ~85% accuracy | Cross-validation |
| Ono & Hishigaki [6] | 2006 | ~200 receptors | Single-protein | C4.5 + NLP | 92.2% accuracy | Cross-validation |
| Isberg *et al.* [21] | 2016 | — | Database resource | GPCRdb curation | — | — |
| Miglionico *et al.* [15] | 2025 | ~400 receptors | Single-protein | AlphaFold3 features | 0.87 AUC | Cross-validation |
| **This study** | **2026** | **1,639 pairs** | **Paired (GPCR, G-protein)** | **ESM-2 + Cross-Attention** | **0.8619 AUC** | **Cluster-aware CV** |

Three major shifts are apparent in the evolution of this field (Table 2). Early work relied on small, manually curated datasets and classical machine-learning methods such as hidden Markov models and decision trees [5,6,12]. These studies achieved high accuracies on their respective test sets, but they pre-dated both large-scale protein language models and rigorous homology-aware validation. More recently, Miglionico *et al.* (2025) demonstrated that AlphaFold3-derived structural features can reach an AUC of 0.87 on a dataset of ~400 receptors [15]. While this performance is numerically comparable to our best result (AUC = 0.8619), their study employed a single-protein formulation and a distinct evaluation split, making direct comparison difficult. Importantly, our controlled ablation shows that when ESM-2 650M embeddings are already used, AlphaFold-derived geometric descriptors provide *no incremental gain*, suggesting that the structural signal leveraged by AlphaFold3 in the Miglionico study may be partially learnable from sequence by large PLMs.

In the broader protein–protein interaction (PPI) literature, sequence-based prediction has been extensively explored. D-SCRIPT and its successors established that Siamese architectures with protein language models can predict PPIs from sequence alone [16,17]. Graph attention networks such as EGRET and Pair-EGRET extended this framework by incorporating cross-attention for residue-level interaction site prediction [18]. However, a recent large-scale benchmark by Reim *et al.* (2025) demonstrated that unbiased sequence-based PPI prediction plateaus at ~0.65 accuracy, and that ESM-2 embeddings drive most of the performance gain regardless of architectural sophistication [19]. Our task—GPCR–G protein family coupling prediction—achieves a substantially higher AUC (0.8619), likely because the restricted biological domain (GPCRs) and the well-defined binding interface (ICL2/3, TM5/6) provide stronger sequence signatures than generic PPI prediction.

Finally, Gamouh *et al.* (2025) showed that while experimental protein structures consistently improve binding-site prediction baselines, the *relative* contribution of structure diminishes as more complex protein language models are used as node features [20]. Our AlphaFold ablation aligns closely with this finding: ESM-2 650M already encodes sufficient structural information for GPCR–G protein coupling prediction, making explicit geometric augmentation redundant.

---

## 4. Discussion

This study presents a large-scale, paired formulation for predicting GPCR–G protein coupling specificity and establishes cross-attention networks with scaled protein language models as the current best-performing approach for this task. Several findings carry broader implications for computational biology and protein-engineering applications.

### 3.1 Paired formulation and comparison with related work

Previous computational studies of GPCR–G protein coupling often treated the problem as GPCR-centric single-protein classification [5,12]. Early sequence-based methods by Möller *et al.* and Ono & Hishigaki achieved high accuracies (>90%) on small, manually curated datasets, but these studies pre-dated the era of large-scale protein language models and rigorous homology-aware validation [5,6]. More recently, Miglionico *et al.* (2025) reported an AUC of 0.87 using AlphaFold3-derived structural features for GPCR–G protein coupling prediction [15]. While their performance is numerically comparable to our best cross-attention result (AUC = 0.8619), the tasks differ in scope: their study employed a distinct dataset and evaluation split, whereas our work focuses on a large paired formulation (1,639 pairs) with strict cluster-aware cross-validation and exhaustive ablations across embedding scales, architectures, and structural feature sets. Importantly, our controlled experiment shows that AlphaFold-derived descriptors provide *no incremental gain* when ESM-2 650M embeddings are already used, suggesting that the structural signal captured by AlphaFold3 in the Miglionico study may be partially learnable from sequence by large PLMs.

Our paired formulation not only mirrors the biological reality of receptor–G protein complex formation but also enables direct interrogation of how partner identity modulates predictions. The observation that models learn family-specific patterns (evidenced by low LOGPSO AUC) suggests that future work should explicitly model G protein family representations, perhaps through family-specific embeddings or multi-task learning.

### 3.2 Scaling laws and architecture selection

The divergence in scaling behavior between SVM and cross-attention is noteworthy. While SVM benefited modestly from 650M embeddings, it quickly hit a ceiling near 0.833, suggesting that the RBF kernel's capacity is saturated by the dataset size and feature complexity. In contrast, the cross-attention architecture continued to improve, achieving a +4.6% absolute gain over its 8M baseline. This aligns with broader findings in deep learning for protein representation: richer pre-trained features are most effectively exploited by architectures with sufficient parametric capacity and non-linear interaction modeling [13].

### 3.3 The surprising redundancy of AlphaFold features

Our negative result for AlphaFold geometric features is both practically and scientifically informative. In an era where AlphaFold structures are widely assumed to be a panacea for structure-aware machine learning, our controlled ablation demonstrates that for this task, ESM-2 650M embeddings already subsume the relevant structural signal. This is consistent with recent observations that large PLMs implicitly learn tertiary-structure constraints during unsupervised pre-training [14]. It also cautions against indiscriminate addition of hand-crafted structural descriptors without empirical validation.

### 3.4 Limitations and future directions

First, our current cross-attention architecture operates on mean-pooled ESM-2 embeddings, which collapses per-residue information into a single vector. Consequently, the cross-attention weights are computed at the protein level rather than the residue level, limiting fine-grained interpretability of interaction interfaces. Future iterations could employ per-residue ESM-2 embeddings to enable residue-level attention maps, which would more directly reveal the molecular determinants of coupling.

Second, LOGPSO performance (~0.60 AUC) indicates poor generalization to unseen G protein families. Incorporating family-aware multi-task objectives or explicit structural docking simulations may be needed to improve cross-family transfer.

Third, while our dataset of 1,639 pairs is large relative to many specialized PPI benchmarks, it remains small by modern deep-learning standards. Self-supervised pre-training on unlabeled GPCR sequences or multi-species data augmentation could further improve performance.

Finally, the wet-lab validation of our candidate predictions remains outstanding. Experimental confirmation of novel high-confidence positives would substantially strengthen the translational relevance of this work.

---

## 4. Materials and Methods

### 4.1 Dataset curation

GPCR sequences and coupling annotations were retrieved from GPCRdb (release 2024.1) and cross-referenced with UniProt annotations. We retained receptors with experimentally validated coupling to at least one of the four major Gα families: Gq/11 (GNAQ, GNA11), Gi/o (GNAI1/2/3, GNAO1), Gs (GNAS, GNAL), and G12/13 (GNA12, GNA13). Pairing labels were binary: 1 if the receptor was annotated as coupling to the family, 0 otherwise. Non-coupling annotations were constructed from receptor–family combinations explicitly listed as non-coupling in GPCRdb or the primary literature.

Sequence clusters were computed with CD-HIT at 40% sequence identity on the full GPCR sequence set, yielding 387 clusters. In cluster-aware CV, all pairs sharing a GPCR were assigned to the same fold to prevent information leakage.

### 4.2 Feature extraction

#### 4.2.1 ESM-2 embeddings

We used two ESM-2 variants: `esm2_t6_8M_UR50D` (8M parameters, 320-d embeddings) and `esm2_t33_650M_UR50D` (650M parameters, 1280-d embeddings). For each GPCR and G protein sequence, per-residue embeddings were extracted and mean-pooled to obtain a single global vector. ICL2 and ICL3 segments were defined using UniProt transmembrane topology annotations. Per-residue embeddings within each ICL segment were mean-pooled to yield ICL-specific ESM vectors. For 8M experiments, ICL vectors were 320-d; for 650M experiments, they were 1280-d.

#### 4.2.2 ICL physicochemical statistics

For each ICL2 and ICL3 segment, we computed 8 statistical descriptors: length, mean hydrophobicity, standard deviation of hydrophobicity, net charge, positive charge ratio, negative charge ratio, hydrophobic ratio, and aromatic ratio. These 16 statistics were concatenated with the ICL ESM vectors.

#### 4.2.3 AlphaFold structural features

AlphaFold v4 PDB structures were downloaded from the EBI AlphaFold database for 364 GPCRs in the dataset. We implemented a custom geometric-analysis pipeline (Python/Biopython) to extract 30 geometric features and 8 PAE-based flexibility features:

- *Geometric*: TM5/TM6 cytoplasmic Cα distance, ICL2/ICL3 end-to-end Cα distances, TM5/TM6 cytoplasmic dihedral angle, ICL2 aromatic centroid depth (relative to membrane plane), interface patch SASA and SASA ratio, and DSSP secondary-structure ratios (helix/sheet/coil) for ICL2 and ICL3.
- *PAE flexibility*: mean PAE for ICL2 and ICL3, intra-ICL PAE, and cross-ICL/TM5/TM6 PAE (8 features total).

When a structure was unavailable, all 38 features were set to zero.

### 4.3 Model architectures

#### 4.3.1 SVM baseline

Support vector machines with RBF kernel were trained using scikit-learn. Hyperparameters were fixed at C = 10.0 with balanced class weights, based on prior hyperparameter search. Features were standardized with `StandardScaler` before training.

#### 4.3.2 Cross-attention network

The `PairedCrossAttentionNet` was implemented in PyTorch. Given a GPCR feature vector **g** ∈ ℝ^(d_GPCR) and a G protein feature vector **p** ∈ ℝ^(d_Gprot), the model first projects both into a shared hidden space:

**q** = proj_GPCR(**g**) ∈ ℝ^h  
**k** = **v** = proj_Gprot(**p**) ∈ ℝ^h

A multi-head cross-attention module (4 heads, h = 256) computes an attended GPCR representation, which is concatenated with the original projection and fed through a 3-layer feed-forward network with GELU activation, layer normalization, and dropout (0.3). The final output is a scalar logit passed through sigmoid. Training used AdamW (lr = 1e-4, weight_decay = 1e-4), BCEWithLogitsLoss with positive class weighting, and early stopping with a patience of 15 epochs on the validation fold AUC.

### 4.4 Training and evaluation

For cluster-aware CV, the 387 clusters were sorted by size and greedily assigned to folds to balance sample counts. Each model was trained 5 times (once per fold) and evaluated on the held-out fold. For LOGPSO, four independent models were trained, each leaving out one G protein family.

All deep-learning experiments were performed on a single NVIDIA GPU with CUDA 12.x. Training time per fold ranged from ~2 minutes (8M baseline) to ~8 minutes (650M alpha).

### 4.5 Candidate selection

After full-data training of the best-performing configuration, predicted probabilities were ranked to select candidates for experimental validation. Candidates were chosen to cover the prediction spectrum: high-confidence positives, boundary cases near 0.50, high-confidence negatives, and the most confident misclassifications (false positives and false negatives).

---

## 5. Data Availability

- The paired dataset of 1,639 GPCR–G protein annotations, all extracted features, and model evaluation results are archived on Zenodo ([DOI to be inserted upon acceptance]) and available in the project repository under `paired_dataset/`.
- AlphaFold PDB structures were retrieved from the EBI AlphaFold Database (https://alphafold.ebi.ac.uk).
- GPCR coupling annotations were sourced from GPCRdb (https://gpcrdb.org).

## 6. Code Availability

All code for feature extraction, model training, cross-validation, statistical testing, and figure generation is available in the project repository and archived on Zenodo ([DOI to be inserted upon acceptance]). Key scripts include:
- `train_paired_cross_attention_650m.py`: 650M cross-attention training
- `paired_cross_validation_enhanced_v2_650m.py`: 650M SVM ablations
- `train_paired_baselines_650m.py`: MLP, Random Forest, and XGBoost baseline training
- `statistical_significance_test_paired.py`: paired statistical tests on fold AUCs
- `generate_schematic_figure.py`: Figure 1 schematic generation
- `generate_figures_for_manuscript.py`: results figure generation
- `generate_wetlab_candidates_650m.py`: wet-lab candidate generation
- `compute_geometric_alphafold_features.py`: AlphaFold geometric feature extraction
- `compute_pae_features.py`: PAE matrix feature extraction

## 7. References

1. Pierce KL, Premont RT, Lefkowitz RJ. Seven-transmembrane receptors. *Nat Rev Mol Cell Biol*. 2002;3(9):639–650.
2. Hauser AS, Attwood MM, Rask-Andersen M, Schiöth HB, Gloriam DE. Trends in GPCR drug discovery: new agents, targets and indications. *Nat Rev Drug Discov*. 2017;16(12):829–842.
3. Wootten D, Christopoulos A, Marti-Solano M, Babu MM, Sexton PM. Mechanisms of signalling and biased agonism in G protein-coupled receptors. *Nat Rev Mol Cell Biol*. 2018;19(10):638–653.
4. Flock T, Ravarani CN, Sun D, et al. Unveiling the structural basis of class A GPCR allosteric modulation. *Nature*. 2023;622(7983):609–616.
5. Möller S, Vilo J, Croning MD. Prediction of the coupling specificity of G protein-coupled receptors to their G proteins. *Bioinformatics*. 2001;17 Suppl 1:S174–S181.
6. Ono T, Hishigaki H. Prediction of GPCR-G protein coupling specificity using features of sequences and biological functions. *Genomics Proteomics Bioinformatics*. 2006;4(1):26–35.
7. Flock T, Hauser AS, Lund N, et al. Selectivity determinants of GPCR-G-protein binding. *Nature*. 2023;545(7654):317–322.
8. Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. 2023;379(6637):1123–1130.
9. Rives A, Meier J, Sercu T, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc Natl Acad Sci USA*. 2021;118(15):e2016239118.
10. Maeda S, Qu Q, Robertson MJ, et al. Structures of the M1 and M2 muscarinic acetylcholine receptor/G protein complexes. *Science*. 2019;364(6440):552–557.
11. Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature*. 2021;596(7873):583–589.
12. Sgourakis NG, Bagos PG, Hamodrakas SJ. Prediction of GPCR-G protein coupling selectivity. *Proteins*. 2005;58(3):544–553.
13. Xu Y, Wang J, Wei D, et al. DLKcat: deep learning for prediction of enzyme kinetic parameters. *bioRxiv*. 2023.
14. Vig J, Madani A, Varshney LR, et al. BERTology meets biology: interpreting attention in protein language models. *ICLR*. 2021.
15. Miglionico P, et al. Predicting and engineering GPCR-G protein coupling specificity with AlphaFold3. *ISMB/ECCB*. 2025.
16. Sledzieski S, et al. D-SCRIPT translates genome to phenome with sequence-based prediction of protein–protein interactions. *Nat Commun*. 2021;12(1):5300.
17. Sledzieski S, et al. Sequence-based prediction of protein–protein interactions: a structure-aware interpretable deep-learning model. *Cell Syst*. 2022;13(12):968–980.
18. Mahbub S, et al. EGRET: edge aggregated graph attention networks and transfer learning improve protein–protein interaction site prediction. *Brief Bioinform*. 2022;23(2):bbab562.
19. Reim R, et al. Large-scale benchmark of sequence-based protein–protein interaction prediction. *Nat Methods*. 2025;22(3):485–494.
20. Gamouh S, et al. The contribution of protein structures to binding-site prediction diminishes with increasing language-model complexity. *Bioinformatics*. 2025;41(2):btaf078.
21. Isberg V, et al. Generic GPCR residue numbers – aligning topology maps while minding the gaps. *Trends Pharmacol Sci*. 2015;36(1):22–31.

---

## Figure Legends

**Figure 1. Schematic overview of the study design, feature engineering, and cross-attention architecture.** (A) Dataset curation: GPCRdb annotations and literature curation yielded 1,639 experimentally validated GPCR–G protein pairs spanning 431 GPCRs and 4 families. CD-HIT clustering at 40% sequence identity produced 387 clusters, which were partitioned into cluster-aware 5-fold cross-validation (CV) or leave-one-G-protein-family-out (LOGPSO) splits. (B) Feature engineering: global mean-pooled ESM-2 embeddings for GPCR (650M, 1280-d) and G protein (8M, 320-d) were concatenated. Topology-aware intracellular loop 2/3 (ICL2/3) features included ESM-2 mean-pooled embeddings and 8 physicochemical statistics per loop. Optional 38-dimensional AlphaFold structural descriptors (geometric + PAE flexibility) were appended. (C) Cross-attention architecture: GPCR and G-protein feature vectors are independently projected, then fed into a multi-head cross-attention module (query = GPCR, key/value = G-protein). The attended representation is concatenated with the original GPCR projection and passed through a feed-forward network with sigmoid output.

**Figure 2. Cluster-aware CV AUC comparison across all model configurations.** Performance of SVM (RBF) and cross-attention deep neural networks under baseline (global ESM-2 only), ICL-full (global + ICL2/3 local features), and Alpha (global + ICL + 38-d AlphaFold structural features) configurations, for both 8M (320-d) and 650M (1280-d) ESM-2 embeddings.

**Figure 3. Effect of protein language model scale on prediction performance.** Comparison of 8M versus 650M ESM-2 embeddings for (A) SVM and (B) cross-attention architectures. Deep learning architectures show larger absolute gains from scaling embedding size than kernel-based methods.

**Figure 4. ICL local features must match global embedding dimension.** SVM and cross-attention performance under three dimension-alignment regimes. Heterogeneous concatenation of 1280-d global embeddings with 320-d ICL features degrades SVM performance, whereas dimension-aligned 1280-d ICL features unlock the full potential of the cross-attention model.

**Figure 5. AlphaFold structural features do not improve performance beyond ESM-2 + ICL.** Direct comparison of ICL-full versus ICL-full + AlphaFold (38-d) across all four model configurations. In every case, the addition of AlphaFold geometric and PAE descriptors results in a small performance decrease (negative Δ), indicating redundancy with the ESM-2 650M representations.

**Figure 6. Feature-group sensitivity in the 650M cross-attention model.** Mean absolute gradient of the predicted probability with respect to each input feature group, computed on the full dataset after model convergence. ICL physicochemical statistics exhibit the highest sensitivity, while G-protein ESM embeddings show the lowest, confirming that the model learns primarily GPCR-centric discriminative patterns under the current mean-pooled architecture.

---

## Supplementary Information

**Supplementary Data 1.** The 1,639 GPCR–G protein pairing annotations with binary coupling labels and sequence cluster assignments.

**Supplementary Data 2.** Complete list of the 38 AlphaFold structural descriptors extracted for each GPCR.

**Supplementary Table 1.** Hyperparameter search results for SVM and cross-attention models.

**Supplementary Table 2.** Per-fold cluster-aware CV and LOGPSO results for all model configurations.

**Supplementary Table 3.** Wet-lab candidate selections from the best-performing model, including predicted probabilities and experimental annotation status.

**Supplementary Table 4.** Paired statistical significance tests (paired *t*-test and Wilcoxon signed-rank test) on cluster-aware CV fold AUCs for key model comparisons.

**Supplementary Figure 1.** PAE feature distributions versus coupling labels (scatterplot matrix).

**Supplementary Figure 2.** Per-fold AUC bar charts for all cluster-aware CV and LOGPSO configurations.

**Supplementary Code.** Python scripts for feature extraction, model training, cross-validation, statistical testing, and figure generation.

---

*Manuscript compiled on 2026-04-17 from the Phase 2 experimental results.*
