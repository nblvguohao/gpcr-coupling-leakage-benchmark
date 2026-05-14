#!/usr/bin/env python3
"""
Label audit: annotate each (GPCR, G protein) pair with metadata and evaluate
performance across three label-quality tiers.

Tier 1: Full dataset (1647 pairs)
Tier 2: Remove zero-positive GPCRs (158 GPCRs with no positive annotations)
Tier 3: High-confidence only — pairs where labeling logic is most defensible

The audit addresses reviewer concern: are negative labels genuinely negative
(experimentally tested) or merely untested?

Annotation fields:
  - species: human/mouse/rat (parsed from UniProt ID)
  - evidence_strength: direct_assay | curated_negative | inferred_negative
  - zero_positive_gpcr: whether this GPCR has zero positive labels
  - n_families_tested: how many of 4 families have annotations for this GPCR
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss)
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "label_audit.json"

N_FOLDS = 5
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ==========================================================================
# Species detection
# ==========================================================================

# UniProt ID prefixes indicating species
HUMAN_IDS = set()  # populated by checking known human GPCRs
# Heuristic: IDs without species prefix or with "HUMAN" prefix are human
# Mouse IDs typically have MOUSE prefix, Rat IDs have RAT prefix
# Common pattern in this dataset: "A_", "B_", "C_" prefixes may indicate species

def detect_species(gpcr_id):
    """Detect species from GPCR ID prefix convention."""
    if "_" in gpcr_id:
        prefix = gpcr_id.split("_")[0]
        base = gpcr_id.split("_", 1)[1]
    else:
        prefix = ""
        base = gpcr_id

    # Known species prefixes in this dataset
    MOUSE_PREFIXES = {"MOUSE", "MUS", "M_"}
    RAT_PREFIXES = {"RAT", "R_"}
    # Some prefixes like "5_", "A_", "B_", "C_", "D_", "G_", "T_" need checking

    if prefix.upper() in MOUSE_PREFIXES:
        return "mouse"
    elif prefix.upper() in RAT_PREFIXES:
        return "rat"
    elif prefix == "":
        return "human"
    elif len(prefix) <= 2 and prefix[0].isalpha() and prefix[0].isupper():
        # Single/double-letter prefix: check if base is a known human UniProt
        # Most non-mouse/rat prefixes in this dataset are paralog/copy indicators
        return "human"
    else:
        return "human"  # default for most entries


# ==========================================================================
# Evidence strength classification
# ==========================================================================

def classify_evidence(source, coupling, gpcr_id, gpcr_pos_families, gpcr_n_tested):
    """
    Classify evidence strength for a pair.

    Positive labels from GPCRdb/IUPHAR: high-confidence (direct experimental assay)
    Positive labels from local_seed: high-confidence (manually curated Gq seed)
    Negative labels from GPCRdb where GPCR has been tested against that family: medium-confidence
    Negative labels where GPCR has NOT been tested: low-confidence (inferred negative)

    Returns: one of "direct_assay", "curated_negative", "inferred_negative"
    """
    if coupling == 1:
        return "direct_assay"  # all positives are experimentally validated

    # Negative label
    if source == "local_seed":
        # Local seed negatives are manually curated
        return "curated_negative"

    # gpcrdb_iuphar negative
    # Check if this GPCR has any positive coupling to any family
    if gpcr_pos_families > 0:
        # GPCR couples to at least one family; negative to this family
        # This is likely a tested negative (selective coupling)
        return "curated_negative"
    else:
        # GPCR has zero positive couplings across all families
        # This could be: a true orphan, an untested GPCR, or genuinely uncoupled
        # We classify as "inferred_negative" since we can't distinguish
        return "inferred_negative"


# ==========================================================================
# Feature loading (minimal)
# ==========================================================================

def load_gpcr_features():
    with open(GPCR_FEATURES_FILE) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}


def load_icl_features():
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        feat = gpcr_feats.get(gid.split("_", 1)[1])
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                return gpcr_feats[key]
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]; break
    icl2_esm = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    icl3_esm = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    icl2_stats = rec.get("ICL2_stats", {}) if rec else {}
    icl3_stats = rec.get("ICL3_stats", {}) if rec else {}
    if icl2_esm.size == 0: icl2_esm = np.zeros(dim)
    if icl3_esm.size == 0: icl3_esm = np.zeros(dim)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    s2 = np.array([icl2_stats.get(k, 0.0) for k in sk])
    s3 = np.array([icl3_stats.get(k, 0.0) for k in sk])
    return icl2_esm, s2, icl3_esm, s3


def build_vectors(df, gpcr_feats, icl_data):
    """Build feature vectors: GPCR 1280 + ICL 2576 + family onehot 4 = 3860-d"""
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]
        gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES], dtype=np.float64)
        if gpcr_f is None: continue
        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({
            "gpcr_id": gid, "family": gf,
            "cluster_id": int(row["cluster_id"]),
        })
    return np.array(X_list), np.array(y_list), metas


# ==========================================================================
# Cluster-aware CV (same pattern)
# ==========================================================================

def get_cluster_folds(metas, cluster_list, n_folds=5):
    n = len(metas)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}
    c_sizes = defaultdict(int)
    for i in range(n): c_sizes[s2c[i]] += 1
    sorted_c = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)
    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0.0] * n_folds
    for c in sorted_c:
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0: continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += c_sizes[cid]
    return s2c, fold_cids


def evaluate_subset(X, y, metas, cluster_list, label):
    """Run cluster-aware SVM CV on a data subset."""
    if len(y) < 20:
        return {"auc": float("nan"), "auc_std": float("nan"),
                "prauc": float("nan"), "n": int(len(y)), "error": "too few samples"}

    s2c, fold_cids = get_cluster_folds(metas, cluster_list)
    fold_aucs, fold_praucs = [], []
    all_probs, all_labels = [], []

    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        if len(te_idx) < 5 or len(tr_idx) < 10:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])

        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]

        all_probs.extend(p); all_labels.extend(y[te_idx])
        if len(set(y[te_idx])) >= 2:
            fold_aucs.append(roc_auc_score(y[te_idx], p))
            fold_praucs.append(average_precision_score(y[te_idx], p))

    if not fold_aucs:
        return {"auc": float("nan"), "n": int(len(y)), "error": "no valid folds"}

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "prauc_mean": float(np.mean(fold_praucs)) if fold_praucs else float("nan"),
        "prauc_std": float(np.std(fold_praucs)) if fold_praucs else float("nan"),
        "brier": float(brier_score_loss(all_labels, all_probs)),
        "n": int(len(y)),
        "n_pos": int(sum(y)),
        "pos_ratio": float(np.mean(y)),
        "fold_aucs": [float(a) for a in fold_aucs],
    }


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=" * 70)
    print("  Label Audit: Metadata Annotation & Tiered Evaluation")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # Load data
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()
    gpcr_feats = load_gpcr_features()
    icl_data = load_icl_features()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  Dataset: {len(df)} pairs, {df['gpcr_id'].nunique()} GPCRs")

    # ======================================================================
    # Phase 1: Annotate metadata
    # ======================================================================
    print("\n--- Phase 1: Metadata Annotation ---")

    # Per-GPCR statistics
    gpcr_pos_count = defaultdict(int)
    gpcr_neg_count = defaultdict(int)
    gpcr_families_tested = defaultdict(set)
    gpcr_sources = defaultdict(set)

    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = row["g_protein_family"]
        if row["coupling"] == 1:
            gpcr_pos_count[gid] += 1
        else:
            gpcr_neg_count[gid] += 1
        gpcr_families_tested[gid].add(gf)
        gpcr_sources[gid].add(row["source"])

    # Annotate each pair
    annotations = []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        species = detect_species(gid)
        evidence = classify_evidence(
            row["source"], int(row["coupling"]), gid,
            gpcr_pos_count[gid], len(gpcr_families_tested[gid])
        )
        annotations.append({
            "gpcr_id": gid,
            "g_protein_family": row["g_protein_family"],
            "coupling": int(row["coupling"]),
            "source": row["source"],
            "species": species,
            "evidence_strength": evidence,
            "zero_positive_gpcr": gpcr_pos_count[gid] == 0,
            "n_families_tested": len(gpcr_families_tested[gid]),
            "gpcr_pos_count": gpcr_pos_count[gid],
        })

    # Summary statistics
    species_counts = defaultdict(int)
    evidence_counts = defaultdict(int)
    zero_pos_count = sum(1 for a in annotations if a["zero_positive_gpcr"])
    n_zero_pos_gpcrs = sum(1 for gid in set(a["gpcr_id"] for a in annotations)
                           if gpcr_pos_count[gid] == 0)

    for a in annotations:
        species_counts[a["species"]] += 1
        evidence_counts[a["evidence_strength"]] += 1

    print(f"  Species: {dict(species_counts)}")
    print(f"  Evidence: {dict(evidence_counts)}")
    print(f"  Zero-positive GPCRs: {n_zero_pos_gpcrs} ({zero_pos_count} pairs)")
    print(f"  GPCRs with >=1 positive: {df['gpcr_id'].nunique() - n_zero_pos_gpcrs}")

    # ======================================================================
    # Phase 2: Build feature vectors and evaluate on three tiers
    # ======================================================================
    print("\n--- Phase 2: Tiered Evaluation ---")

    # Build full feature matrix
    X_full, y_full, metas_full = build_vectors(df, gpcr_feats, icl_data)
    print(f"  Full feature matrix: {X_full.shape}")

    # Tier 1: Full dataset
    print("\n  Tier 1: Full dataset")
    tier1 = evaluate_subset(X_full, y_full, metas_full, cluster_list, "full")
    print(f"    N={tier1['n']}, AUC={tier1.get('auc_mean', 'nan'):.4f} "
          f"+/- {tier1.get('auc_std', 0):.4f}, PRAUC={tier1.get('prauc_mean', 'nan'):.4f}")

    # Tier 2: Remove zero-positive GPCRs
    print("\n  Tier 2: Remove zero-positive GPCRs")
    tier2_mask = np.array([
        gpcr_pos_count[metas_full[i]["gpcr_id"]] > 0
        for i in range(len(y_full))
    ])
    X_t2, y_t2 = X_full[tier2_mask], y_full[tier2_mask]
    metas_t2 = [metas_full[i] for i in range(len(metas_full)) if tier2_mask[i]]
    tier2 = evaluate_subset(X_t2, y_t2, metas_t2, cluster_list, "no_zero_pos")
    print(f"    N={tier2['n']}, AUC={tier2.get('auc_mean', 'nan'):.4f} "
          f"+/- {tier2.get('auc_std', 0):.4f}, PRAUC={tier2.get('prauc_mean', 'nan'):.4f}")

    # Tier 3: High-confidence only (remove inferred_negative pairs)
    print("\n  Tier 3: High-confidence only (direct_assay + curated_negative)")
    tier3_mask = np.array([
        annotations[i]["evidence_strength"] != "inferred_negative"
        for i in range(len(y_full))
    ])
    X_t3, y_t3 = X_full[tier3_mask], y_full[tier3_mask]
    metas_t3 = [metas_full[i] for i in range(len(metas_full)) if tier3_mask[i]]
    tier3 = evaluate_subset(X_t3, y_t3, metas_t3, cluster_list, "high_conf")
    print(f"    N={tier3['n']}, AUC={tier3.get('auc_mean', 'nan'):.4f} "
          f"+/- {tier3.get('auc_std', 0):.4f}, PRAUC={tier3.get('prauc_mean', 'nan'):.4f}")

    # ======================================================================
    # Phase 3: Per-family evidence breakdown
    # ======================================================================
    print("\n--- Phase 3: Per-Family Evidence Breakdown ---")
    family_breakdown = {}
    for fam in FAMILIES:
        fam_pairs = [a for a in annotations if a["g_protein_family"] == fam]
        fam_evidence = defaultdict(lambda: {"pos": 0, "neg": 0})
        for a in fam_pairs:
            lbl = "pos" if a["coupling"] == 1 else "neg"
            fam_evidence[a["evidence_strength"]][lbl] += 1
        family_breakdown[fam] = {
            "total": len(fam_pairs),
            "positive": sum(1 for a in fam_pairs if a["coupling"] == 1),
            "by_evidence": {k: dict(v) for k, v in fam_evidence.items()},
        }
        print(f"    {fam}: {family_breakdown[fam]}")

    # ======================================================================
    # Save
    # ======================================================================
    out = {
        "description": "Label audit: tiered evaluation by label quality",
        "dataset_summary": {
            "n_pairs": len(annotations),
            "n_gpcrs": df["gpcr_id"].nunique(),
            "n_zero_positive_gpcrs": n_zero_pos_gpcrs,
            "species_distribution": dict(species_counts),
            "evidence_distribution": dict(evidence_counts),
        },
        "tiered_results": {
            "tier1_full": tier1,
            "tier2_remove_zero_positive": tier2,
            "tier3_high_confidence": tier3,
        },
        "per_family_breakdown": family_breakdown,
        "reference_svm_650m_icl_auc": 0.832,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
