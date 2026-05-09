You are an expert Scientific Illustrator for top-tier journals (Nature, Cell, Briefings in Bioinformatics).
Your task is to generate a professional "Graphical Abstract / Framework Figure" based on the research below.

**Abstract:**
We reformulate GPCR-G protein coupling prediction as a pairwise binary classification problem
over 1,647 experimentally annotated (GPCR, G protein family) pairs spanning 431 GPCRs across
four G protein families (Gq/11, Gi/o, Gs, G12/13). Using ESM-2 protein language model embeddings
(up to 650M parameters, 1280-dimensional) combined with topology-aware intracellular loop (ICL2/3)
features, a cross-attention neural network achieves AUC of 0.862 under strict cluster-aware
cross-validation. Two key methodological findings emerge: (1) ICL local features must dimension-match
the global ESM embedding, and (2) AlphaFold structural descriptors provide no incremental benefit
beyond ESM-2 650M, demonstrating that large protein language models implicitly encode structural
information. A calibration analysis (Brier score 0.008) confirms the model produces reliable
probability estimates for experimental candidate prioritization.

**Methodology Pipeline (sequential flow):**

[INPUT] GPCR sequences + G protein sequences
  |
  v
[FEATURE EXTRACTION] Three parallel branches:
  Branch A: Global ESM-2 650M embeddings (1280-d mean-pooled from final transformer layer)
  Branch B: Topology-aware ICL2/3 features — UniProt transmembrane annotations →
           extract loop boundaries → mean-pool local ESM within loops + 8 physicochemical stats
           KEY RULE: ICL dimension MUST match global embedding dimension (1280-d → 1280-d)
  Branch C (OPTIONAL): 38 AlphaFold structural descriptors (TM distances, SASA, PAE, DSSP)
           FINDING: This branch provides NO gain beyond ESM-2 650M + ICL
  |
  v
[FEATURE FUSION] GPCR features (global ESM + ICL2 + ICL3) concatenated with
                    G protein features (global ESM only)
  |
  v
[CROSS-ATTENTION NETWORK]
  - Linear projection → shared 256-d space (separate for GPCR and G protein)
  - Multi-head cross-attention (4 heads): GPCR = Query, G protein = Key/Value
  - Attended GPCR representation concatenated with original projection
  - 3-layer FFN (GELU + LayerNorm + Dropout 0.3)
  - Sigmoid output → P(coupling) ∈ [0,1]
  |
  v
[EVALUATION] Three strategies:
  1. Random 5-fold CV
  2. Cluster-aware 5-fold CV (387 CD-HIT clusters at 40% identity)
  3. Leave-one-G-protein-family-out (LOGPSO) — cross-family generalization

**Visual Style Requirements:**
1. Style: Flat vector illustration, clean lines, academic aesthetic — similar to figures in
   DeepMind (AlphaFold) or Briefings in Bioinformatics papers.
2. Layout: Top-to-Bottom flow with three horizontal panels:
   - Panel A: Dataset construction (GPCRdb → 1,647 pairs → 4 families → CD-HIT clustering → CV splits)
   - Panel B: Feature engineering (three parallel columns: ESM-2, ICL topology, AlphaFold → fusion)
   - Panel C: Cross-attention architecture (GPCR/Query, G-protein/Key-Value, attention, FFN, output)
3. Color Palette: Pastel professional tones, white background.
   - GPCR-related: soft blue
   - G protein-related: soft orange/coral
   - ICL/topology: soft green
   - AlphaFold (optional/dashed): soft grey
   - Neural network components: soft purple/indigo
4. Text Rendering: MUST include legible text labels for key components:
   "GPCR ESM-2 650M", "G protein ESM-2 650M", "ICL2/3 Features", "Cross-Attention",
   "Q (GPCR)", "K,V (G protein)", "FFN", "P(coupling)", "AUC=0.862", "Brier=0.008"
5. Negative Constraints: NO photorealistic images, NO 3D rendering, NO messy sketches,
   NO text smaller than readable size, NO dark backgrounds.

**Core Novelty to Highlight:**
- Paired formulation (GPCR + G protein as dual inputs, not single-protein classification)
- Dimension alignment rule (ICL local features must match global embedding dim)
- AlphaFold features → dashed/optional branch → X mark indicating "no gain"
- Cross-attention mechanism (Query/Key/Value architecture)
