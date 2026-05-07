# PPT Outline — 5-Minute Oral Presentation

**Title:** Topology-Aware GPCR–G Protein Coupling Prediction via ESM-2 and Curated Transmembrane Annotations

---

## Slide 1: Title

- **Title:** Topology-Aware GPCR–G Protein Coupling Prediction via ESM-2 Embeddings and Curated Transmembrane Annotations
- **Subtitle:** Integrating family-level coupling priors and ICL2/3 homology similarity

---

## Slide 2: Background — Why GPCR–G Protein Coupling Matters

- **Bullet points:**
  - GPCRs = largest membrane-protein family; ~34% of approved drug targets
  - Coupling specificity (Gq / Gi / Gs / G12/13) determines downstream signaling
  - Structural biology: ICL2 and ICL3 are the primary docking interfaces (*Nature* 2015; *Science* 2022)
- **Visual suggestion:** A simple cartoon of GPCR with 7-TM helices + G protein; highlight ICL2/ICL3 in red
- *(30 seconds)*

---

## Slide 3: Two Core Limitations of Existing Methods

- **Left column: Problem 1 — Single-protein classification degradation**
  - Most methods = "classify the GPCR" with a fixed G-protein template
  - Result: model collapses into a GPCR-only classifier; never learns pairwise rules
- **Right column: Problem 2 — Lack of topology guidance**
  - Whole-sequence mean-pooled embeddings blur binding vs. non-binding regions
  - Attention drifts to N-tail/C-tail; misses the true ICL2/3 interface
- **Visual suggestion:** Side-by-side comparison diagrams
- *(30 seconds)*

---

## Slide 4: Our Three Solutions

- **1. Pairwise task construction**
  - Every sample = (GPCR, G-protein family) pair
  - Same GPCR appears with different families → model must use G-protein-side info
- **2. Topology-aware feature engineering**
  - Curated UniProt 7-TM annotations → precise ICL2/3 boundaries (98.8% success)
  - 16 physicochemical stats per loop injected as local features
- **3. Strict homology-controlled evaluation**
  - Three-tier scheme: Random CV → Cluster-aware CV → LOGPSO
- *(40 seconds)*

---

## Slide 5: Main Results — Cross-Validation AUC

- **Table (simplified):**

| Config                                | 8M Cluster CV   | 650M Cluster CV |
| ------------------------------------- | --------------- | --------------- |
| Baseline                              | 0.797           | 0.819           |
| + ICL stats + family prior + sim (V2) | **0.842** | **0.833** |

- **Key talking points:**
  - Topology + family prior pushes 8M model up by **+0.045**
  - 650M baseline is already stronger; relative gain is smaller (+0.014), suggesting larger embeddings encode some family info implicitly
  - ICL-only model = 0.598, proving global and local features are **complementary**
- *(50 seconds)*

---

## Slide 6: The LOGPSO Stress Test — A Protocol–Representation Mismatch

- **Bullet points:**
  - LOGPSO = leave-one-G-protein-family-out; AUC collapses to ~0.60
  - **Diagnosis:** not a biological bottleneck, but a structural mismatch
  - When testing on the held-out family, every sample carries the *same unseen constant vector*
  - RBF kernel distances become identical → model cannot separate positives from negatives
  - ICL-only model avoids this and yields non-zero metrics, confirming the diagnosis
- **Take-away:** concatenated fixed-vector representations fail at cross-family transfer; need attention-based co-embedding
- *(30 seconds)*

---

## Slide 7: Independent Validation — OPN4 Test Set

- **Setup:** Predict "can this opsin activate human GNAQ?" for 9 sequences
- **Leakage check:** 2 human isoforms leaked; 7 non-human opsins are clean (max sim 0.07–0.17)
- **Results table (clean samples only):**

| Model                  | Clean Accuracy        |
| ---------------------- | --------------------- |
| V1 (original features) | **28.6%** (2/7) |
| V2 (8M ESM-2)          | **71.4%** (5/7) |
| V2 (650M ESM-2)        | **85.7%** (6/7) |

- **650M breakthroughs:**
  - Fish melanopsin prob: 0.852 ✓
  - Spider kumopsin1 prob: 0.544 ✓
- *(50 seconds)*

---

## Slide 8: Critical Insight — Label Reliability Problem

- **Bullet points:**
  - 5 of 9 test labels are graded "D — unreliable" in the human-GNAQ framework
  - Bee/butterfly opsins labeled "No" because "insects use insect Gq"
  - This does **not** logically imply they cannot activate human GNAQ
  - **Implication:** some "errors" in V1 may actually be benchmark flaws, not model flaws
  - Pragmatic fix: label unverified samples as UNKNOWN and exclude from quantitative evaluation
- *(30 seconds)*

---

## Slide 9: Three Core Conclusions

- **1.** Pairwise framework works — SHAP confirms all G-protein dimensions contribute; the model learns interaction rules, not just GPCR classification
- **2.** Topology annotations steer embeddings — model attention shifts from non-functional tails to the ICL2/3 interface, matching structural biology
- **3.** Cross-family generalization needs new representations — LOGPSO exposes the limit of concatenated fixed vectors; residue-level cross-attention or AlphaFold contacts are the next step
- *(30 seconds)*

---

## Slide 10: Limitations & Future Directions

- **Limitations:**
  - No 3D structural features yet
  - Cross-species benchmark labels are unreliable for distant species
  - Binary family labels lack subtype resolution
- **Future work:**
  - Integrate AlphaFold interface-contact predictions
  - Expand training set to invertebrate GPCRs + test even larger PLMs (ESM-2 3B, ESM-3)
  - Wet-lab validation: site-directed mutagenesis + BRET/Co-IP
- *(25 seconds)*


## Speaking Tips for Key Slides

**Slide 5 (Main CV Results):**
Spend extra time on the **comparison between 8M and 650M**. Say something like: *"Notice that the larger 650M model already has a strong baseline at 0.819, so the explicit family prior gives a smaller relative gain. This tells us that bigger embeddings already encode some family information implicitly — but they still benefit from topology-aware features."*

**Slide 7 (OPN4 Test):**
This is your "wow" moment. Emphasize the **650M breakthrough**: *"With the 650M model, we correctly classified the fish melanopsin with 85% probability and the spider kumopsin1 with 54% — both were false negatives in the smaller 8M model."*

**Slide 8 (Label Reliability):**
This shows scientific maturity. Say: *"Importantly, we discovered that many 'wrong' predictions in our first model weren't actually model errors — the benchmark labels themselves are unreliable for distant species in the human-GNAQ framework."*

**Slide 11 (Closing):**
Read the one-sentence summary slowly and clearly. Then pause, smile, and open for questions.
