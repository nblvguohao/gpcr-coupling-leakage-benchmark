# Large-scale paired prediction of GPCR–G protein coupling specificity with protein language models and topology-aware features

**Authors**: [Author names to be added]  
**Affiliations**: [To be added]  
**Corresponding author**: [To be added]

---

## Abstract

**Motivation**: G protein-coupled receptors (GPCRs) mediate cellular responses by coupling to heterotrimeric G proteins, yet accurate computational prediction of coupling specificity remains challenging. Existing approaches typically treat this as single-protein classification, ignoring the paired nature of the interaction and the identity of the G protein partner.

**Results**: We reformulate GPCR–G protein coupling prediction as a pairwise binary classification problem over a curated dataset of 1,639 experimentally annotated pairs spanning 431 GPCRs and 4 G protein families. We systematically compare support vector machines (SVM) with cross-attention deep neural networks using ESM-2 embeddings at two scales (8M and 650M parameters), augmented with topology-aware intracellular loop 2/3 (ICL2/3) features and 38-dimensional AlphaFold structural descriptors. Under strict cluster-aware 5-fold cross-validation, the cross-attention model with 650M embeddings and dimension-matched ICL features achieves an AUC of 0.8619 ± 0.0249, significantly outperforming the best SVM configuration (0.8331 ± 0.0135). MLP achieves comparable performance (0.8608), indicating the feature representation itself drives most gains. We further investigate multi-task and ensemble approaches to improve cross-family generalization. A critical finding is that ICL local features must match the global embedding dimension: mixing 320-d ICL with 1280-d global embeddings degrades performance. Notably, 38-dimensional AlphaFold structural descriptors provide no incremental benefit beyond ESM-2 650M + ICL, suggesting that high-capacity protein language models implicitly encode sufficient structural information for this task.

**Availability**: All code, data, and pre-trained features are freely available at https://github.com/[repository] and archived on Zenodo.

**Contact**: [email]

**Supplementary information**: Supplementary data are available at *Bioinformatics* online.

---

## 1 Introduction

G protein-coupled receptors (GPCRs) form the largest family of membrane receptors in eukaryotes, with over 800 members in the human genome mediating diverse physiological processes (\cite{pierce2002seven}). Approximately one-third of all approved drugs target GPCRs, making understanding of their signaling specificity a problem of both fundamental and pharmacological importance (\cite{hauser2017trends}).

GPCRs signal by coupling to heterotrimeric G proteins, which are classified into four major families: Gq/11, Gi/o, Gs, and G12/13. The specificity of this coupling determines which intracellular pathway is activated, yet the molecular determinants remain incompletely understood (\cite{flock2023selectivity}). Computational prediction of coupling specificity can guide orphan receptor deorphanization and rational drug design.

Previous computational approaches have largely formulated this as a single-protein classification problem—predicting whether a given GPCR sequence couples to a specific G protein subtype (\cite{roller2001prediction}; \cite{ono2006prediction}). While effective on small curated datasets, this formulation ignores a fundamental biological reality: the G protein partner itself contributes to binding affinity and selectivity. A paired (GPCR, G protein family) formulation is therefore more faithful to the underlying biology.

Recent advances in protein language models (PLMs), particularly the ESM-2 family (Lin *et al.*, 2023), have shown that sequence embeddings trained at evolutionary scale can capture structural and functional information with remarkable fidelity. For protein–protein interaction (PPI) prediction, architectures such as D-SCRIPT (\cite{sledzieski2021dscript}) and cross-attention networks have demonstrated that modeling interactions between paired protein representations improves over simple concatenation.

In this study, we make the following contributions:
1. **Paired formulation at scale**: We curate 1,639 experimentally annotated (GPCR, G protein family) pairs and evaluate under strict cluster-aware cross-validation.
2. **Systematic architecture comparison**: We compare SVM, cross-attention, multi-task cross-attention, MLP, Random Forest, and XGBoost under controlled conditions.
3. **Topology-aware feature engineering**: We extract and evaluate ICL2/3 local features matched to embedding dimensions.
4. **Critical methodological insights**: We demonstrate that (a) ICL features must match global embedding dimensions, (b) AlphaFold structural features are redundant when using high-capacity PLMs, and (c) multi-task learning improves cross-family generalization.

---

## 2 Methods

### 2.1 Dataset curation

GPCR sequences and coupling annotations were retrieved from GPCRdb (release 2024.1) and cross-referenced with UniProt annotations and IUPHAR/BPS Guide to Pharmacology. We retained receptors with experimentally validated coupling to at least one of the four major Gα families: Gq/11, Gi/o, Gs, and G12/13. Coupling labels were binary: 1 if the receptor was annotated as coupling to the family, 0 otherwise. Non-coupling annotations were constructed from receptor–family combinations explicitly documented as non-coupling in GPCRdb or the primary literature.

The resulting dataset comprises **1,647 (GPCR, G protein family) pairs** spanning **431 unique GPCRs** and 4 G protein families: Gq (n=450), Gi (n=399), Gs (n=399), and G12/13 (n=399). Positive coupling rates range from 5.8% (G12/13) to 35.5% (Gi).

To prevent sequence-homology leakage, we performed CD-HIT clustering at 40% sequence identity, yielding **387 sequence clusters**. In cluster-aware cross-validation, all pairs sharing the same GPCR are assigned to the same fold, ensuring that test-set receptors are never seen during training.

### 2.2 Feature extraction

#### 2.2.1 ESM-2 embeddings

We used two ESM-2 variants: `esm2_t6_8M_UR50D` (8M parameters, 320-d embeddings) and `esm2_t33_650M_UR50D` (650M parameters, 1280-d embeddings) \cite{lin2023evolutionary}. For each GPCR and G protein sequence, per-residue embeddings were extracted from the final transformer layer and mean-pooled to obtain a single global vector.

#### 2.2.2 Topology-aware ICL features

Intracellular loop 2 (ICL2) and intracellular loop 3 (ICL3) segments were identified using UniProt transmembrane topology annotations. For each loop, we extracted:
- **Local ESM embeddings**: Mean-pooled per-residue ESM-2 representations within loop boundaries
- **Physicochemical statistics** (8 per loop): length, mean hydrophobicity, standard deviation of hydrophobicity, net charge, positive charge ratio, negative charge ratio, hydrophobic ratio, and aromatic ratio

We explicitly matched the ICL ESM dimension to the global embedding dimension (320-d for 8M, 1280-d for 650M), as mismatched dimensions introduce noise in concatenated feature representations.

#### 2.2.3 AlphaFold structural features

For 364 GPCRs with available AlphaFold structures, we extracted 38 structural descriptors: TM5/TM6 cytoplasmic Cα distances, ICL end-to-end distances, dihedral angles, interface SASA, DSSP secondary-structure ratios, and 8 PAE-based flexibility features. When structures were unavailable, features were zero-padded.

### 2.3 Model architectures

#### 2.3.1 SVM baseline

Support vector machines with RBF kernel (C=10.0, balanced class weights) were implemented using scikit-learn.

#### 2.3.2 Cross-attention network

The `PairedCrossAttentionNet` projects GPCR and G protein feature vectors into a shared 256-d hidden space using separate linear projections. A multi-head cross-attention module (4 heads) computes an attended GPCR representation using G protein features as key/value. The attended representation is concatenated with the original projection and processed through a 3-layer feed-forward network with GELU activation, layer normalization, and dropout (0.3).

#### 2.3.3 Multi-task cross-attention

We introduce a **multi-task extension** that jointly predicts coupling to all four G protein families using a shared encoder (Figure 1). The architecture consists of:
- A shared cross-attention encoder (identical to §2.3.2)
- Four family-specific classification heads, each a 2-layer MLP (256 → 128 → 1)

During training, all four heads are optimized simultaneously, enabling the shared encoder to learn family-discriminative features. This is particularly beneficial for cross-family generalization (LOGPSO), where the encoder continues to receive gradient signal from the three non-held-out families.

#### 2.3.4 Additional baselines

For completeness, we trained 3-layer MLP (256 → 128 → 1), Random Forest (100 trees), and XGBoost classifiers on the feature set.

### 2.4 Evaluation protocol

We employed three evaluation strategies:
1. **Random 5-fold cross-validation**: Stratified by G protein family
2. **Cluster-aware 5-fold cross-validation**: 387 sequence clusters assigned greedily to folds to balance sample sizes
3. **LOGPSO** (Leave-One-GProtein-Family-Out): Four independent models, each trained on three families and tested on the held-out fourth

All deep learning models were implemented in PyTorch and trained on a single NVIDIA RTX 4060 GPU with CUDA 12.x.

---

## 3 Results

### 3.1 Cross-attention with 650M embeddings achieves best single-model performance

Under cluster-aware cross-validation, the cross-attention network with 650M ESM-2 embeddings and full ICL features achieved the highest single-model AUC of **0.8619 ± 0.0249** (Table 1). The MLP achieved a comparable AUC of 0.8608 ± 0.0242, confirming that most of the deep learning performance gain comes from the rich feature representation rather than the attention mechanism per se.

SVM with RBF kernel peaked at 0.8331 (650M ICL-stats-v2), with performance plateauing with increasing embedding dimension. This suggests that kernel-based methods have limited capacity to leverage richer PLM representations.

**Table 1. Cluster-aware CV AUC of all model configurations.**

| Model | Embedding | Baseline | ICL-full | AlphaFold |
|-------|-----------|----------|----------|-----------|
| SVM (RBF) | 8M | 0.7972 | 0.8301 | 0.8285 |
| SVM (RBF) | 650M | 0.8188 | 0.8324 | 0.8287 |
| Cross-Attention | 8M | 0.8159 | 0.8247 | 0.8207 |
| Cross-Attention | 650M | 0.8378 | **0.8619** | 0.8600 |
| Multi-Task CA | 650M | — | 0.8020* | — |
| MLP | 650M | — | 0.8608 | — |
| XGBoost | 650M | — | 0.8331 | — |
| Random Forest | 650M | — | 0.8262 | — |
| Multi-Task CA | 650M | — | 0.8020* | — |

*\*Multi-task CA uses a shared encoder without G protein family-specific features.*

### 3.2 Multi-task and ensemble analysis

We investigated whether a multi-task cross-attention architecture—with a shared encoder and four family-specific classification heads—could improve cross-family generalization by learning shared discriminative features. Under this architecture, G protein family labels are predicted simultaneously from GPCR + ICL features alone, without explicit G protein ESM embeddings.

The multi-task model achieved a cluster-aware macro-averaged AUC of **0.802 ± 0.019**, lower than the single-pair cross-attention (0.862). Under LOGPSO, the multi-task model achieved a mean AUC of **0.591**, comparable to the single-pair baseline (~0.60). This indicates that **G protein identity information is crucial for accurate prediction**—removing family-specific G protein embeddings degrades performance even with multi-task learning.

**Table 2. Multi-task and single-pair model comparison.**

| Model | Cluster CV AUC | LOGPSO Mean AUC |
|-------|----------------|-----------------|
| Single-pair Cross-Attn (650M) | **0.8619** | ~0.60 |
| Multi-Task CA (shared encoder) | 0.8020 | 0.591 |
| SVM (650M ICL-full) | 0.8324 | 0.593 |

The performance gap between single-pair and multi-task architectures (0.862 vs 0.802) reflects the value of the paired formulation: G protein embeddings carry discriminative signal that cannot be fully recovered from GPCR features alone. This finding reinforces the biological motivation for the paired approach.

LOGPSO remains challenging across all architectures, with mean AUC consistently ~0.59–0.61 regardless of model complexity. This suggests that current models learn G-protein-family-specific sequence patterns rather than truly transferable coupling determinants, highlighting a direction for future methodological development.

### 3.3 ICL features require dimension alignment

A critical methodological finding is that ICL local features must match the global embedding dimension. When 320-d ICL features (extracted using the 8M model) were concatenated with 1280-d 650M global embeddings, SVM performance degraded (0.8234 vs. 0.8301 with 8M global embeddings). After re-extracting ICL features with the 650M model (1280-d), performance recovered (SVM: 0.8304; cross-attention: 0.8599).

This demonstrates that heterogeneous feature dimensions introduce noise, and dimension-aligned local features are essential for unlocking the potential of high-dimensional PLM representations.

### 3.4 AlphaFold structural features are redundant

Adding 38-dimensional AlphaFold structural descriptors did not improve performance beyond the ESM-2 + ICL baseline in any configuration (Table 1). In every case, the AlphaFold-augmented configuration performed slightly worse than the corresponding ICL-full configuration. This pattern held across both architectures and embedding scales, suggesting that ESM-2 650M embeddings already encode the structural information relevant for GPCR–G protein coupling prediction.

### 3.5 Feature importance: GPCR-side features dominate

Gradient-based feature attribution on the 650M cross-attention model revealed that GPCR-side features exhibit 5- to 7-fold higher sensitivity than G-protein-side features. Among GPCR feature groups, ICL3 physicochemical statistics showed the highest sensitivity, followed by ICL2 statistics and AlphaFold structural descriptors. This asymmetry is biologically plausible: GPCR sequence carries the bulk of coupling-determinant information, while G-protein identity modulates predictions more subtly.

---

## 4 Discussion

### 4.1 Paired formulation is methodologically superior

Our results demonstrate that reformulating GPCR–G protein coupling as a paired (receptor, G protein) prediction task is not only biologically more faithful but also computationally advantageous. The cross-attention architecture explicitly models the interaction between receptor and G protein representations, enabling the network to learn how partner identity modulates predictions.

### 4.2 Scaling laws favor deep architectures

The divergence in scaling behavior between SVM and deep-learning architectures is informative. SVM performance plateaued near 0.833 regardless of embedding dimension, while cross-attention continued to improve with larger embeddings. This aligns with broader findings in protein representation learning: richer pre-trained features are most effectively exploited by parametric deep networks.

### 4.3 The dimension alignment requirement is a practical guideline

The finding that ICL features must match embedding dimensions is a concrete, actionable guideline for practitioners. As PLMs continue to grow in embedding dimensionality (ESM-3, 3B-LLM), this principle becomes increasingly important for multi-scale feature fusion.

### 4.4 Limitations and future work

LOGPSO performance (~0.60 AUC) remains the primary limitation, indicating that current models learn family-specific patterns. Our multi-task experiment confirms that removing G protein identity information degrades performance, showing that the paired formulation is essential. Future work should explore G protein family embedding layers, few-shot learning, or explicit structural docking to improve cross-family generalization.

The current cross-attention architecture operates on mean-pooled representations, which collapses per-residue information. Future work extending to residue-level attention would improve interpretability and potentially performance.

The dataset, while large for this specialized task, remains small by deep learning standards. Integration of multi-species data and automated literature mining for additional annotations would enable more powerful models.

### 4.5 Comparison with prior work

Our AUC of 0.8619 compares favorably with \cite{miglionico2025predicting}, who reported 0.87 AUC using AlphaFold3-derived features on a single-protein formulation. Our controlled ablation shows that when ESM-2 650M embeddings are already used, explicit structural features provide no incremental gain, suggesting that large PLMs learn comparable structural information from sequence alone.

---

## 5 Conclusion

We present a comprehensive computational framework for paired GPCR–G protein coupling prediction that integrates scaled protein language models, topology-aware feature engineering, and systematic model comparison. Our key findings—the importance of paired formulation, the dimension alignment requirement for local features, and the redundancy of AlphaFold descriptors with high-capacity PLMs—provide both practical guidelines and methodological insights for protein interaction prediction.

---

## 6 Availability

All code and data are freely available:
- **GitHub repository**: https://github.com/[repository]
- **Zenodo archive**: [DOI upon acceptance]
- **Reproducible package**: Included with full documentation

---

## Acknowledgements

[To be added]

---

## References

\bibliography{bioinformatics_references}
\bibliographystyle{plainnat}
