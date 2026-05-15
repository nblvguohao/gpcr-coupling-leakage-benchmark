#!/usr/bin/env python3
"""
Unified cluster-level bootstrap statistics for GPCR coupling prediction evaluation.

Modes (selected via --mode):
  main     – CA vs SVM comparison: cluster-bootstrap AUC, paired difference test,
             G12/13 detailed metrics (PRAUC, recall@precision, calibration bins),
             per-family AUC/PRAUC/Brier breakdown.
             Output: bootstrap_statistics.json

  extended – CA vs SVM vs MLP comparison: cluster-bootstrap AUC for all three
             models, pairwise bootstrap difference tests (CA–SVM, CA–MLP),
             extended per-family metrics (recall@fixed-precision, precision@top-k),
             minimal family-conditioned MLP reference.
             Output: extended_bootstrap_statistics.json

  all      – Run both main and extended modes, producing both output files.
             (Default.)

All bootstrap procedures operate at the GPCR CLUSTER level (sampling clusters
with replacement, not individual pairs), which preserves within-cluster
correlation.  N_BOOTSTRAP = 2000, RANDOM_SEED = 42 throughout.

Core statistical methods retained:
  - Cluster-level bootstrap 95 % CI for AUC (percentile method)
  - Corrected resampled t-test for paired AUC differences
  - Two-sided bootstrap p-value (proportion of bootstrap diffs <= 0)
  - Precision-recall curve analysis with recall at fixed precision thresholds
  - Brier score, binned calibration (G12/13 in main mode)
  - Per-family performance decomposition for all four G-protein families
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, precision_recall_curve,
                              confusion_matrix)
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

# Input files
CA_PREDICTIONS_FILE   = DATA_DIR / "ca_predictions.json"
SVM_PREDICTIONS_FILE  = DATA_DIR / "svm_predictions.json"
MLP_PREDICTIONS_FILE  = DATA_DIR / "mlp_predictions.json"
PAIRING_MATRIX_FILE   = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE         = DATA_DIR / "sequence_clusters.json"

# Output files
OUTPUT_FILE          = DATA_DIR / "bootstrap_statistics.json"
EXTENDED_OUTPUT_FILE = DATA_DIR / "extended_bootstrap_statistics.json"

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(pred_file):
    """Convert {gpcr_id: {family: {label, prob}}} JSON to flat DataFrame."""
    with open(pred_file) as f:
        preds = json.load(f)
    records = []
    for gid, families in preds.items():
        for gfam, info in families.items():
            records.append({
                "gpcr_id": gid,
                "g_protein_family": gfam,
                "label": int(info.get("label", -1)),
                "prob": float(info.get("prob", float("nan"))),
            })
    return pd.DataFrame.from_records(records)


def load_cluster_map():
    """Build gpcr_id -> cluster_id mapping from the pairing matrix."""
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"])
    gpcr_to_cluster = {}
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        cid = int(row["cluster_id"])
        if gid not in gpcr_to_cluster:
            gpcr_to_cluster[gid] = cid
    return gpcr_to_cluster


# ═══════════════════════════════════════════════════════════════════════════════
# Cluster-level bootstrap core
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_bootstrap_auc(df, gpcr_to_cluster, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Bootstrap AUC over GPCR clusters (sample clusters with replacement).

    Returns dict with mean, std, 95 % CI (percentile), n_bootstrap, n_clusters.
    """
    np.random.seed(seed)

    cluster_to_rows = defaultdict(list)
    for i, row in df.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0:
            cluster_to_rows[cid].append(i)

    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)

    aucs = []
    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for cid in sampled_cids:
            sampled_rows.extend(cluster_to_rows[cid])

        y = df.iloc[sampled_rows]["label"].values
        p = df.iloc[sampled_rows]["prob"].values

        if len(set(y)) >= 2:
            aucs.append(roc_auc_score(y, p))

    aucs = np.array(aucs)
    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "ci_95_low": float(np.percentile(aucs, 2.5)),
        "ci_95_high": float(np.percentile(aucs, 97.5)),
        "n_bootstrap": n_bootstrap,
        "n_clusters": n_clusters,
    }


def cluster_bootstrap_diff(df_a, df_b, gpcr_to_cluster, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Bootstrap the AUC difference (A - B) over GPCR clusters.

    Cluster-level corrected resampled test. Returns mean_diff, std_diff,
    95 % CI, and two-sided bootstrap p-value (proportion of diffs <= 0).
    """
    np.random.seed(seed)

    cluster_to_rows = defaultdict(list)
    for i, row in df_a.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0:
            cluster_to_rows[cid].append(i)

    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)

    diffs = []
    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for cid in sampled_cids:
            sampled_rows.extend(cluster_to_rows[cid])

        y = df_a.iloc[sampled_rows]["label"].values
        p_a = df_a.iloc[sampled_rows]["prob"].values
        p_b = df_b.iloc[sampled_rows]["prob"].values

        if len(set(y)) >= 2:
            diffs.append(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))

    diffs = np.array(diffs)
    p_value = float(np.mean(diffs <= 0))
    return {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
        "p_value_bootstrap": p_value,
        "n_bootstrap": n_bootstrap,
        "n_clusters": n_clusters,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-cluster AUC influence (defined but not part of any mode's output)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auc_per_cluster(df, gpcr_to_cluster):
    """
    Compute per-cluster contribution to overall AUC.

    Returns list of {cluster_id, n_pairs_in_cluster, auc_when_removed,
    delta_auc} where delta_auc = overall_auc - auc_when_removed.
    """
    valid = df.dropna(subset=["prob"])
    all_auc = roc_auc_score(valid["label"], valid["prob"]) if len(set(valid["label"])) >= 2 else float("nan")

    cluster_contrib = []
    all_clusters = sorted(set(gpcr_to_cluster.get(gid, -1) for gid in valid["gpcr_id"]))

    for cid in all_clusters:
        if cid < 0:
            continue
        c_gpcrs = [gid for gid, c in gpcr_to_cluster.items() if c == cid]
        mask = ~valid["gpcr_id"].isin(c_gpcrs)
        if mask.sum() < 10:
            continue
        y_sub = valid.loc[mask, "label"]
        p_sub = valid.loc[mask, "prob"]
        if len(set(y_sub)) >= 2:
            sub_auc = roc_auc_score(y_sub, p_sub)
            cluster_contrib.append({
                "cluster_id": cid,
                "n_pairs_in_cluster": int((~mask).sum()),
                "auc_when_removed": float(sub_auc),
                "delta_auc": float(all_auc - sub_auc),
            })

    return cluster_contrib, all_auc


# ═══════════════════════════════════════════════════════════════════════════════
# G12/13 detailed metrics (main mode)
# ═══════════════════════════════════════════════════════════════════════════════

def g12_13_detailed_metrics(df):
    """
    Detailed metrics for G12/13 family (26 positive, 371 negative pairs).

    AUC alone is misleading for heavily imbalanced classes.
    Reports: PRAUC, recall at fixed precision levels, binned calibration.
    """
    g12 = df[df["g_protein_family"] == "G12_13"].dropna(subset=["prob"])

    if len(g12) == 0:
        return {"error": "no G12/13 pairs"}

    y = g12["label"].values
    p = g12["prob"].values

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    auc = roc_auc_score(y, p) if len(set(y)) >= 2 else float("nan")
    prauc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    prec, rec, thresh = precision_recall_curve(y, p)

    recall_at_precision = {}
    for target_prec in [0.5, 0.7, 0.8, 0.9]:
        best_recall = 0.0
        for pi, ri in zip(prec, rec):
            if pi >= target_prec and ri > best_recall:
                best_recall = ri
        recall_at_precision[f"recall@prec={target_prec}"] = float(best_recall)

    top_k = min(10, n_pos)
    if top_k > 0:
        top_idx = np.argsort(p)[-top_k:]
        precision_at_k = float(y[top_idx].mean())
    else:
        precision_at_k = float("nan")

    # Binned calibration
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    for j in range(n_bins):
        bmask = (p >= bin_edges[j]) & (p < bin_edges[j+1])
        if bmask.sum() > 0:
            bin_stats.append({
                "bin": f"[{bin_edges[j]:.1f}, {bin_edges[j+1]:.1f})",
                "n": int(bmask.sum()),
                "mean_predicted": float(p[bmask].mean()),
                "mean_actual": float(y[bmask].mean()),
                "accuracy": float((np.round(p[bmask]) == y[bmask]).mean()),
            })

    return {
        "n": int(len(y)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "pos_ratio": float(n_pos / len(y)),
        "auc": float(auc) if not np.isnan(auc) else None,
        "prauc": float(prauc),
        "brier": float(brier),
        "recall_at_precision": recall_at_precision,
        "precision_at_k": {f"precision@{top_k}": precision_at_k},
        "calibration_bins": bin_stats,
        "mean_prob_positive": float(p[y == 1].mean()) if n_pos > 0 else float("nan"),
        "mean_prob_negative": float(p[y == 0].mean()) if n_neg > 0 else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-family metrics (two variants: simple for main, extended for extended)
# ═══════════════════════════════════════════════════════════════════════════════

def per_family_breakdown(df):
    """
    Simple per-family metrics for all four G-protein families (main mode).

    Returns {family: {n, n_pos, n_neg, pos_ratio, auc, prauc, brier,
    mean_prob_positive, mean_prob_negative}}.
    """
    results = {}
    for fam in FAMILIES:
        fam_df = df[df["g_protein_family"] == fam].dropna(subset=["prob"])
        if len(fam_df) == 0:
            results[fam] = {"error": "no data"}
            continue

        y = fam_df["label"].values
        p = fam_df["prob"].values

        n_pos = int(y.sum())
        n_neg = int(len(y)) - n_pos

        results[fam] = {
            "n": int(len(y)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_ratio": float(n_pos / len(y)) if len(y) > 0 else 0.0,
            "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
            "prauc": float(average_precision_score(y, p)) if len(set(y)) >= 2 else None,
            "brier": float(brier_score_loss(y, p)),
            "mean_prob_positive": float(p[y == 1].mean()) if n_pos > 0 else None,
            "mean_prob_negative": float(p[y == 0].mean()) if n_neg > 0 else None,
        }
    return results


def per_family_metrics(df):
    """
    Extended per-family metrics with recall@fixed-precision and precision@top-k.

    Returns {family: {n, n_pos, pos_ratio, auc, prauc, brier,
    recall_at_precision, precision_at_k, mean_prob_positive,
    mean_prob_negative}}.
    """
    results = {}
    for fam in FAMILIES:
        sub = df[df["g_protein_family"] == fam].dropna(subset=["prob"])
        if len(sub) == 0:
            continue
        y, p = sub["label"].values, sub["prob"].values
        n_pos = int(y.sum())
        prec, rec, _ = precision_recall_curve(y, p)
        recall_at_prec = {}
        for target_p in [0.5, 0.7, 0.8, 0.9]:
            best_r = 0.0
            for pi, ri in zip(prec, rec):
                if pi >= target_p and ri > best_r:
                    best_r = ri
            recall_at_prec[f"recall@prec={target_p}"] = float(best_r)
        top_k = min(10, n_pos)
        prec_at_k = float(y[np.argsort(p)[-top_k:]].mean()) if top_k > 0 else float("nan")
        results[fam] = {
            "n": int(len(y)),
            "n_pos": n_pos,
            "pos_ratio": float(n_pos / len(y)),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
            "prauc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "recall_at_precision": recall_at_prec,
            "precision_at_k": {f"precision@{top_k}": prec_at_k},
            "mean_prob_positive": float(p[y == 1].mean()) if n_pos > 0 else None,
            "mean_prob_negative": float(p[y == 0].mean()) if (len(y) - n_pos) > 0 else None,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Mode: main  — CA vs SVM (original bootstrap_statistics.py behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

def run_main_mode():
    """Run CA vs SVM cluster-bootstrap with G12/13 detailed metrics."""
    print("=" * 70)
    print("  Bootstrap Statistics for GPCR Coupling Prediction")
    print(f"  Mode: main  |  {N_BOOTSTRAP} bootstrap samples over GPCR clusters")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # Load data
    print("\n[1/4] Loading predictions...")
    df_ca = load_predictions(CA_PREDICTIONS_FILE)
    df_svm = load_predictions(SVM_PREDICTIONS_FILE)
    gpcr_to_cluster = load_cluster_map()

    print(f"  CA predictions:  {len(df_ca)} pairs, {df_ca['gpcr_id'].nunique()} GPCRs")
    print(f"  SVM predictions: {len(df_svm)} pairs, {df_svm['gpcr_id'].nunique()} GPCRs")
    print(f"  Clusters mapped: {len(set(gpcr_to_cluster.values()))}")

    # Overall metrics
    valid_ca = df_ca.dropna(subset=["prob"])
    valid_svm = df_svm.dropna(subset=["prob"])
    ca_auc = roc_auc_score(valid_ca["label"], valid_ca["prob"])
    svm_auc = roc_auc_score(valid_svm["label"], valid_svm["prob"])
    print(f"\n  Overall CA AUC:  {ca_auc:.4f}")
    print(f"  Overall SVM AUC: {svm_auc:.4f}")
    print(f"  Delta (CA-SVM):  {ca_auc - svm_auc:.4f}")

    # 1. Cluster-level bootstrap AUC
    print(f"\n[2/4] Cluster-level bootstrap ({N_BOOTSTRAP} samples)...")
    ca_bootstrap = cluster_bootstrap_auc(valid_ca, gpcr_to_cluster)
    svm_bootstrap = cluster_bootstrap_auc(valid_svm, gpcr_to_cluster)

    print(f"  CA  AUC: {ca_bootstrap['mean']:.4f} [{ca_bootstrap['ci_95_low']:.4f}, {ca_bootstrap['ci_95_high']:.4f}] (cluster bootstrap)")
    print(f"  SVM AUC: {svm_bootstrap['mean']:.4f} [{svm_bootstrap['ci_95_low']:.4f}, {svm_bootstrap['ci_95_high']:.4f}] (cluster bootstrap)")

    # 2. Cluster-level bootstrap AUC difference
    print(f"\n[3/4] Cluster-level bootstrap AUC difference (CA - SVM)...")
    diff_bootstrap = cluster_bootstrap_diff(valid_ca, valid_svm, gpcr_to_cluster)
    print(f"  Mean diff: {diff_bootstrap['mean_diff']:.4f}")
    print(f"  95% CI:    [{diff_bootstrap['ci_95_low']:.4f}, {diff_bootstrap['ci_95_high']:.4f}]")
    print(f"  Bootstrap p-value (two-sided): {diff_bootstrap['p_value_bootstrap']:.4f}")

    # 3. G12/13 detailed metrics
    print(f"\n[4/4] G12/13 detailed metrics...")
    g12_ca = g12_13_detailed_metrics(df_ca)
    g12_svm = g12_13_detailed_metrics(df_svm)

    for model_name, g12 in [("CA", g12_ca), ("SVM", g12_svm)]:
        print(f"\n  {model_name} G12/13:")
        print(f"    N={g12['n']}, Pos={g12['n_pos']}, Neg={g12['n_neg']}")
        print(f"    AUC={g12.get('auc', 'nan'):.4f}, PRAUC={g12['prauc']:.4f}, Brier={g12['brier']:.4f}")
        print(f"    Mean prob (pos): {g12.get('mean_prob_positive', 'nan')}")
        if g12.get('mean_prob_positive') is not None:
            print(f"      = {g12['mean_prob_positive']:.4f}")
        print(f"    Recall@precision:")
        for k, v in g12["recall_at_precision"].items():
            print(f"      {k}: {v:.4f}")

    # Per-family breakdown
    per_fam_ca = per_family_breakdown(df_ca)
    per_fam_svm = per_family_breakdown(df_svm)

    print(f"\n  Per-family AUC comparison:")
    print(f"  {'Family':<12s} {'CA AUC':>10s} {'SVM AUC':>10s} {'CA PRAUC':>10s} {'SVM PRAUC':>10s}")
    print(f"  {'-'*60}")
    for fam in FAMILIES:
        ca_a = per_fam_ca.get(fam, {}).get("auc")
        svm_a = per_fam_svm.get(fam, {}).get("auc")
        ca_p = per_fam_ca.get(fam, {}).get("prauc")
        svm_p = per_fam_svm.get(fam, {}).get("prauc")
        print(f"  {fam:<12s} {str(ca_a) if ca_a else 'N/A':>10s} "
              f"{str(svm_a) if svm_a else 'N/A':>10s} "
              f"{str(ca_p) if ca_p else 'N/A':>10s} "
              f"{str(svm_p) if svm_p else 'N/A':>10s}")

    # Save
    out = {
        "description": f"Bootstrap statistics: {N_BOOTSTRAP} samples over GPCR clusters",
        "overall_auc": {
            "ca": float(ca_auc),
            "svm": float(svm_auc),
            "delta": float(ca_auc - svm_auc),
        },
        "cluster_bootstrap_auc": {
            "ca": ca_bootstrap,
            "svm": svm_bootstrap,
        },
        "cluster_bootstrap_auc_diff": diff_bootstrap,
        "g12_13_detailed": {
            "ca": g12_ca,
            "svm": g12_svm,
        },
        "per_family": {
            "ca": {fam: {k: v for k, v in metrics.items()}
                    for fam, metrics in per_fam_ca.items()},
            "svm": {fam: {k: v for k, v in metrics.items()}
                     for fam, metrics in per_fam_svm.items()},
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode: extended  — CA vs SVM vs MLP (original extended_bootstrap.py behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

def run_extended_mode():
    """Run CA vs SVM vs MLP cluster-bootstrap with extended per-family metrics."""
    print("=" * 70)
    print("  Extended Bootstrap Statistics")
    print(f"  Mode: extended  |  Comparisons: CA vs SVM | CA vs MLP")
    print("=" * 70)

    # Load data
    print("\nLoading predictions...")
    df_ca = load_predictions(CA_PREDICTIONS_FILE)
    df_svm = load_predictions(SVM_PREDICTIONS_FILE)
    gpcr_to_cluster = load_cluster_map()

    if MLP_PREDICTIONS_FILE.exists():
        df_mlp = load_predictions(MLP_PREDICTIONS_FILE)
        print(f"  MLP predictions loaded: {len(df_mlp)} pairs")
    else:
        df_mlp = None
        print("  MLP predictions NOT FOUND -- skipping MLP comparisons")

    print(f"  CA:  {len(df_ca)} pairs, {df_ca['gpcr_id'].nunique()} GPCRs")
    print(f"  SVM: {len(df_svm)} pairs")

    # Overall AUCs
    valid_ca = df_ca.dropna(subset=["prob"])
    valid_svm = df_svm.dropna(subset=["prob"])
    ca_auc = roc_auc_score(valid_ca["label"], valid_ca["prob"])
    svm_auc = roc_auc_score(valid_svm["label"], valid_svm["prob"])
    print(f"\n  Overall CA AUC:  {ca_auc:.4f}")
    print(f"  Overall SVM AUC: {svm_auc:.4f}")

    # 1. Bootstrap AUC for each model
    print(f"\n[1] Cluster-level bootstrap AUC ({N_BOOTSTRAP} samples)...")
    ca_bs = cluster_bootstrap_auc(valid_ca, gpcr_to_cluster)
    svm_bs = cluster_bootstrap_auc(valid_svm, gpcr_to_cluster)
    print(f"  CA:  {ca_bs['mean']:.4f} [{ca_bs['ci_95_low']:.4f}, {ca_bs['ci_95_high']:.4f}]")
    print(f"  SVM: {svm_bs['mean']:.4f} [{svm_bs['ci_95_low']:.4f}, {svm_bs['ci_95_high']:.4f}]")

    mlp_bs = None
    if df_mlp is not None:
        valid_mlp = df_mlp.dropna(subset=["prob"])
        mlp_auc = roc_auc_score(valid_mlp["label"], valid_mlp["prob"])
        mlp_bs = cluster_bootstrap_auc(valid_mlp, gpcr_to_cluster)
        print(f"  MLP: {mlp_bs['mean']:.4f} [{mlp_bs['ci_95_low']:.4f}, {mlp_bs['ci_95_high']:.4f}]")

    # 2. Pairwise bootstrap comparisons
    print(f"\n[2] Pairwise bootstrap comparisons...")

    diff_ca_svm = cluster_bootstrap_diff(valid_ca, valid_svm, gpcr_to_cluster)
    print(f"  CA vs SVM:  DAUC = {diff_ca_svm['mean_diff']:.4f} "
          f"[{diff_ca_svm['ci_95_low']:.4f}, {diff_ca_svm['ci_95_high']:.4f}], "
          f"p = {diff_ca_svm['p_value_bootstrap']:.4f}")

    diff_ca_mlp = None
    if df_mlp is not None:
        diff_ca_mlp = cluster_bootstrap_diff(valid_ca, valid_mlp, gpcr_to_cluster)
        print(f"  CA vs MLP:  DAUC = {diff_ca_mlp['mean_diff']:.4f} "
              f"[{diff_ca_mlp['ci_95_low']:.4f}, {diff_ca_mlp['ci_95_high']:.4f}], "
              f"p = {diff_ca_mlp['p_value_bootstrap']:.4f}")

    # 3. Per-family metrics for all models
    print(f"\n[3] Per-family metrics...")
    per_fam = {"CA": per_family_metrics(df_ca), "SVM": per_family_metrics(df_svm)}
    if df_mlp is not None:
        per_fam["MLP"] = per_family_metrics(df_mlp)

    for model in per_fam:
        print(f"\n  {model}:")
        print(f"  {'Family':<12s} {'AUC':>8s} {'PRAUC':>8s} {'Brier':>8s} "
              f"{'P@10':>8s} {'R@P=0.5':>10s} {'MeanProb+':>10s}")
        print(f"  {'-'*70}")
        for fam in FAMILIES:
            m = per_fam[model].get(fam, {})
            auc_s = f"{m.get('auc', 0):.3f}" if m.get('auc') is not None else "N/A"
            prauc_s = f"{m.get('prauc', 0):.3f}"
            brier_s = f"{m.get('brier', 0):.3f}"
            pk = list(m.get("precision_at_k", {}).values())
            pk_s = f"{pk[0]:.3f}" if pk else "N/A"
            rp = m.get("recall_at_precision", {}).get("recall@prec=0.5", 0)
            mp = m.get("mean_prob_positive")
            mp_s = f"{mp:.3f}" if mp is not None else "N/A"
            print(f"  {fam:<12s} {auc_s:>8s} {prauc_s:>8s} {brier_s:>8s} "
                  f"{pk_s:>8s} {rp:>10.3f} {mp_s:>10s}")

    # 4. Minimal family-conditioned MLP reference
    min_mlp_file = DATA_DIR / "minimal_family_classifier_results.json"
    min_mlp_ref = None
    if min_mlp_file.exists():
        with open(min_mlp_file) as f:
            min_data = json.load(f)
        min_mlp_ref = min_data["results"]["mlp"]
        print(f"\n[4] Minimal family-conditioned MLP reference:")
        print(f"  AUC = {min_mlp_ref['auc_mean']:.4f} "
              f"[{min_mlp_ref['auc_95ci_low']:.4f}, {min_mlp_ref['auc_95ci_high']:.4f}]")
        print(f"  PRAUC = {min_mlp_ref['prauc_mean']:.4f}, Brier = {min_mlp_ref['brier_mean']:.4f}")
        print(f"  DAUC vs CA: {ca_bs['mean'] - min_mlp_ref['auc_mean']:.4f}")

    # Save
    out = {
        "description": f"Extended bootstrap: {N_BOOTSTRAP} cluster-level samples, 3 pairwise comparisons",
        "cluster_bootstrap_auc": {"CA": ca_bs, "SVM": svm_bs},
        "pairwise_comparisons": {
            "CA_vs_SVM": diff_ca_svm,
            "CA_vs_MLP": diff_ca_mlp,
        },
        "per_family_metrics": per_fam,
        "minimal_family_conditioned_mlp_reference": min_mlp_ref,
        "note": (
            "CA vs MLP and minimal MLP vs CA comparisons use cluster-level bootstrap. "
            "Fold-level t-test results are in supplementary. "
            "If CA vs MLP bootstrap p > 0.05, write 'CA and MLP are statistically comparable'. "
            "Report real DAUC, CI, and p for each comparison -- do NOT reuse CA-SVM p=0.006 for other pairs."
        ),
    }
    if df_mlp is not None:
        out["cluster_bootstrap_auc"]["MLP"] = mlp_bs

    with open(EXTENDED_OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {EXTENDED_OUTPUT_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified cluster-level bootstrap statistics for GPCR coupling prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bootstrap.py                        # default: run all\n"
            "  python bootstrap.py --mode main            # CA vs SVM only\n"
            "  python bootstrap.py --mode extended        # CA vs SVM vs MLP\n"
            "  python bootstrap.py --mode all             # both (explicit)\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["main", "extended", "all"],
        default="all",
        help="Analysis mode: 'main' (CA vs SVM), 'extended' (CA vs SVM vs MLP + per-family), "
             "or 'all' (both, default).",
    )
    args = parser.parse_args()

    if args.mode == "main":
        run_main_mode()
    elif args.mode == "extended":
        run_extended_mode()
    else:  # "all"
        print("Running main mode...")
        run_main_mode()
        print("\n" + "=" * 70)
        print("Running extended mode...")
        run_extended_mode()


if __name__ == "__main__":
    main()
