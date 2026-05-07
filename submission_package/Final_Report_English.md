# Final Research Report (English Version)

**Title:** Topology-Aware GPCR–G Protein Coupling Prediction via ESM-2 Embeddings and Curated Transmembrane Annotations

---

## Part 1: Background & Motivation

### 1.1 Biological Background

G protein-coupled receptors (GPCRs) are the largest family of membrane proteins in the human genome and account for roughly 34% of approved drug targets. They activate downstream signaling by coupling to heterotrimeric G proteins (four major families: Gq, Gi, Gs, and G12/13). Predicting which G-protein family a given GPCR couples to is therefore a central problem in drug design and signal-transduction research.

Structural biology has established that the G-protein binding interface of Class-A GPCRs lies on the cytoplasmic side, where the cytoplasmic ends of **TM5 and TM6** together with **ICL2** form the core cavity; **ICL3** further strengthens this interaction in certain receptors (Rasmussen et al., *Nature* 2011; subsequent cryo-EM structures).

### 1.2 Two Core Limitations of Existing Methods

| Limitation                                          | Manifestation                                                                                           | Consequence                                                                                            |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Single-protein classification degradation** | Most methods treat the task as "classify the GPCR" using a fixed G-protein template (e.g., human GNAQ). | The model never learns pairwise interaction rules; it collapses into a GPCR-only classifier.           |
| **Lack of topology guidance**                 | Whole-sequence mean-pooled embeddings are used without distinguishing binding vs. non-binding regions.  | Model attention drifts to non-functional regions (N-tail/C-tail) and misses the true ICL2/3 interface. |

### 1.3 Our Three Solutions

1. **Genuine pairwise task construction** — every sample is a *(GPCR, G-protein family)* pair. The same GPCR appears with different families, forcing the model to use G-protein-side information.
2. **Topology-aware feature engineering** — we extract precise ICL2/3 boundaries from curated UniProt transmembrane annotations and inject this structural prior.
3. **Strict homology-controlled evaluation** — a three-tier cross-validation scheme (Random → Cluster-aware → LOGPSO) to eliminate sequence leakage.

---

## Part 2: Data & Methods (Brief)

### 2.1 Dataset

- **Source:** GPCRdb/IUPHAR coupling annotations + a local seed of 51 Gq-specific pairs.
- **Final usable pairs:** 1,639
- **Unique GPCRs:** 448 (clustered into 387 homology groups at 30% identity)
- **Positives:** 417 (25.4%) | **Negatives:** 1,222 (74.6%)

### 2.2 Features

- **ESM-2 embeddings:** Main experiments use `esm2_t6_8M_UR50D` (320-d). We also ran a parallel study with `esm2_t33_650M_UR50D` (1,280-d) to test model-scale effects.
- **UniProt topology:** 7-TM helices are used to pinpoint ICL2 and ICL3 boundaries (98.8% success rate).
- **ICL2/3 descriptors:** 8 physicochemical statistics per loop (length, hydrophobicity, net charge, aromatic ratio, etc.).
- **V2 enhancements (new):**
  - **Family-level Gq positive-rate prior** — how often a given GPCR family couples to Gq in the training data.
  - **ICL2/3 homology similarity** — cosine similarity of ICL stats to the nearest human ortholog.

### 2.3 Three Evaluation Tiers

| Tier                       | Splitting rule                    | Interpretation                                |
| -------------------------- | --------------------------------- | --------------------------------------------- |
| **Random CV**        | Standard 5-fold stratified        | Optimistic upper bound                        |
| **Cluster-aware CV** | By homology cluster (CD-HIT ~30%) | Realistic generalization to unseen receptors  |
| **LOGPSO**           | Leave-one-G-protein-family-out    | Extreme stress-test for cross-family transfer |

---

## Part 3: Core Experimental Results

### 3.1 Feature Ablation (Main Table)

**V1 (original features, 8M ESM-2):**

| Config      | Dim   | Random CV AUC  | Cluster CV AUC           | LOGPSO AUC |
| ----------- | ----- | -------------- | ------------------------ | ---------- |
| Baseline    | 640   | 0.815 ± 0.031 | 0.797 ± 0.022           | 0.638      |
| + ICL stats | 656   | 0.832 ± 0.017 | 0.810 ± 0.030           | 0.628      |
| + ICL full  | 1,296 | 0.839 ± 0.014 | **0.830 ± 0.041** | 0.609      |
| ICL only    | 656   | 0.598 ± 0.025 | 0.683 ± 0.035           | 0.551      |

**V2 (with family prior + ICL similarity, 8M ESM-2):**

| Config         | Dim   | Random CV AUC            | Cluster CV AUC           | LOGPSO AUC |
| -------------- | ----- | ------------------------ | ------------------------ | ---------- |
| Baseline       | 640   | 0.815 ± 0.031           | 0.798 ± 0.014           | 0.638      |
| + ICL stats V2 | 658   | **0.848 ± 0.020** | **0.836 ± 0.019** | 0.631      |
| + ICL full V2  | 1,298 | 0.848 ± 0.015           | **0.842 ± 0.005** | 0.606      |

**V2 (with family prior + ICL similarity, 650M ESM-2):**

| Config         | Dim   | Random CV AUC  | Cluster CV AUC           | LOGPSO AUC |
| -------------- | ----- | -------------- | ------------------------ | ---------- |
| Baseline       | 1,600 | 0.820 ± 0.033 | 0.819 ± 0.017           | 0.606      |
| + ICL stats V2 | 1,618 | 0.830 ± 0.032 | **0.833 ± 0.014** | 0.601      |
| + ICL full V2  | 4,178 | 0.822 ± 0.028 | 0.832 ± 0.019           | 0.593      |

**Key findings:**

- Topology-aware features raised Cluster CV AUC from **0.797 to 0.830** (V1, +0.033).
- Adding the family prior and ICL similarity pushed the 8M model to **0.842** (+0.045 overall) and the 650M model to **0.833** (+0.014).
- **Non-monotonic model-scale effect:** The 650M baseline is already strong (0.819), so the *relative* gain from explicit priors is smaller than for the 8M model. This suggests larger embeddings already encode some family/species information implicitly.
- The ICL-only model achieves only 0.598 in Random CV, confirming that **local topology and global sequence context are complementary, not interchangeable.**

### 3.2 LOGPSO: A Protocol–Representation Mismatch

All models containing a fixed G-protein vector collapse under LOGPSO (AUC ~0.60 and precision/recall/F1 ≈ 0). This is **not a biological bottleneck** but a structural incompatibility: when one entire G-protein family is held out, every test sample carries the *same unseen constant block*, causing the RBF kernel to assign nearly identical distances. The ICL-only model avoids this and yields non-zero metrics, confirming the diagnosis.

### 3.3 Interpretability

- **Permutation importance:** ICL2 aromatic ratio and ICL3 length rank above any single ESM-2 dimension, aligning with structural biology.
- **SHAP regional attention:** After adding topology features, model weight shifts from non-binding regions (N-tail/C-tail) toward the known binding interface (ICL2/ICL3), consistent with the hypothesis that structural annotations steer attention biologically.

---

## Part 4: Independent Validation — OPN4 Test Set

We predicted whether 9 opsin sequences can activate **human GNAQ**, with strict leakage checks and label-reliability grading.

### 4.1 Leakage Analysis

- **NP_150598.1** (human melanopsin iso1): Jaccard = 1.000 to Q9UHM6 → **LEAKED**
- **NP_001025186.1** (human melanopsin iso2): Jaccard = 0.967 to Q9UHM6 → **LEAKED**
- The remaining 7 non-human opsins show max similarity 0.07–0.17 → **clean**

### 4.2 V1 Results (original features)

Overall accuracy: 44.4%; clean accuracy: **28.6%** (2/7). Honey-bee and butterfly opsins suffer systematic false positives (probabilities 0.55–0.74).

### 4.3 V2 Results — 8M vs 650M

**V2 (8M):** overall 77.8%; clean **71.4%**

- False positives drop sharply (bee/butterfly probs fall to 0.31–0.44).
- **Trade-off:** the 14.3% family prior is so strong that it pushes two distant positives below threshold (fish melanopsin and spider kumopsin1).

**V2 (650M):** overall 88.9%; clean **85.7%**

- Fish melanopsin probability: **0.852** (correct)
- Spider kumopsin1 probability: **0.544** (correct)
- One new false negative: squid opsin (0.519).

**Critical insight:** The richer 1,280-d embeddings of the 650M model **resist the suppressive effect of the family prior** on distant positives, yielding a dramatic cross-species improvement.

### 4.4 Label Reliability Issue

Five of the nine test labels are graded "D — unreliable" for the human-GNAQ framework. The bee and butterfly opsins were labeled "No" based on the inference that "insects use insect Gq," which **does not logically imply** they cannot activate human GNAQ. This means a portion of the "errors" in V1 may actually reflect a flawed benchmark rather than poor model generalization.

---

## Part 5: Three Core Conclusions

1. **Pairwise framework eliminates single-protein degradation.** SHAP confirms all 320 G-protein ESM-2 dimensions contribute, proving the model learns interaction rules, not just GPCR classification.
2. **Topology annotations steer pretrained embeddings toward biologically sensible decision boundaries.** UniProt-curated 7-TM boundaries outperform heuristic methods (98.8% vs. 1.2% success), and model attention shifts from non-functional tails to the ICL2/3 interface.
3. **Cross-family generalization demands new interaction representations.** LOGPSO exposes a *protocol–representation mismatch* for concatenated fixed vectors. Future work should move toward residue-level cross-attention or AlphaFold-derived interface contacts.

---

## Part 6: Limitations & Future Directions

| Limitation                                       | Impact | Mitigation / Next step                                                                    |
| ------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------- |
| No 3D structural features                        | High   | Integrate AlphaFold interface-contact predictions                                         |
| LOGPSO protocol–representation mismatch         | Medium | Adopt attention-based co-embedding                                                        |
| Binary family labels (no subtypes)               | Medium | Move toward subtype-level prediction                                                      |
| No wet-lab validation                            | High   | Site-directed mutagenesis + BRET/Co-IP                                                    |
| Cross-species benchmark unreliable               | High   | Label unverified samples as**UNKNOWN** and remove them from quantitative evaluation |
| Family prior can suppress distant positives (8M) | Medium | Use larger pretrained models (650M) or expand training-set species coverage               |

---

## One-Sentence Summary for Closing

**By injecting UniProt-curated transmembrane topology into ESM-2 embeddings and introducing family-level coupling priors plus ICL2/3 homology similarity, we raised Cluster CV AUC from 0.797 to 0.842 (8M) and 0.833 (650M), with interpretability analyses confirming attention shifts toward the true binding interface. On the independent OPN4 test, clean-sample accuracy jumped from 28.6% to 71.4% (8M) and 85.7% (650M), yet the unreliable labeling of distant species in the human-GNAQ framework remains a critical bottleneck for cross-species evaluation.**
