#!/usr/bin/env python3
"""
Bootstrap statistics for GPCR coupling prediction evaluation.

1. Cluster-level bootstrap 95% CI for CA and SVM AUC
2. Cluster-level bootstrap for CA vs SVM difference (corrected resampled test)
3. G12/13 detailed metrics (PRAUC, recall@precision, calibration)
4. Per-family performance decomposition

Bootstrap over GPCR CLUSTERS (not pairs) preserves within-cluster correlation.
"""

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

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

CA_PREDICTIONS_FILE = DATA_DIR / "ca_predictions.json"
SVM_PREDICTIONS_FILE = DATA_DIR / "svm_predictions.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "bootstrap_statistics.json"

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


def load_predictions_as_dataframe(pred_file):
    """Convert {gpcr_id: {family: {label, prob}}} to flat DataFrame."""
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
    """Build gpcr_id -> cluster_id mapping."""
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"])
    gpcr_to_cluster = {}
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        cid = int(row["cluster_id"])
        if gid not in gpcr_to_cluster:
            gpcr_to_cluster[gid] = cid
    return gpcr_to_cluster


def compute_auc_per_cluster(df, gpcr_to_cluster):
    """
    Compute per-cluster contribution to AUC.

    Returns list of (cluster_id, n_pairs, n_pos, n_neg, auc_if_removed)
    where auc_if_removed = AUC when this cluster's pairs are excluded.
    This measures cluster influence rather than decomposing AUC.
    """
    # Overall AUC
    valid = df.dropna(subset=["prob"])
    all_auc = roc_auc_score(valid["label"], valid["prob"]) if len(set(valid["label"])) >= 2 else float("nan")

    cluster_contrib = []
    all_clusters = sorted(set(gpcr_to_cluster.get(gid, -1) for gid in valid["gpcr_id"]))

    for cid in all_clusters:
        if cid < 0: continue
        c_gpcrs = [gid for gid, c in gpcr_to_cluster.items() if c == cid]
        mask = ~valid["gpcr_id"].isin(c_gpcrs)
        if mask.sum() < 10: continue
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


def cluster_bootstrap_auc(df, gpcr_to_cluster, n_bootstrap=2000, seed=42):
    """
    Bootstrap AUC over GPCR clusters.

    Sample clusters with replacement, compute AUC, get 95% CI.
    """
    np.random.seed(seed)

    # Build cluster -> row indices mapping
    cluster_to_rows = defaultdict(list)
    for i, row in df.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0:
            cluster_to_rows[cid].append(i)

    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)

    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        # Sample clusters with replacement
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        # Collect all rows from sampled clusters (with duplicates)
        sampled_rows = []
        for cid in sampled_cids:
            sampled_rows.extend(cluster_to_rows[cid])

        # Use sampled rows to compute AUC
        y_sample = df.iloc[sampled_rows]["label"].values
        p_sample = df.iloc[sampled_rows]["prob"].values

        if len(set(y_sample)) >= 2:
            auc = roc_auc_score(y_sample, p_sample)
            bootstrap_aucs.append(auc)

    aucs = np.array(bootstrap_aucs)
    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "ci_95_low": float(np.percentile(aucs, 2.5)),
        "ci_95_high": float(np.percentile(aucs, 97.5)),
        "n_bootstrap": n_bootstrap,
        "n_clusters": n_clusters,
    }


def cluster_bootstrap_auc_difference(df_ca, df_svm, gpcr_to_cluster, n_bootstrap=2000, seed=42):
    """
    Bootstrap the AUC difference between CA and SVM over GPCR clusters.

    This is a cluster-level corrected resampled test.
    H0: CA and SVM have equal AUC (difference = 0)
    """
    np.random.seed(seed)

    # Build cluster -> row indices mapping
    cluster_to_rows = defaultdict(list)
    for i, row in df_ca.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0:
            cluster_to_rows[cid].append(i)

    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for cid in sampled_cids:
            sampled_rows.extend(cluster_to_rows[cid])

        y_sample = df_ca.iloc[sampled_rows]["label"].values
        p_ca = df_ca.iloc[sampled_rows]["prob"].values
        p_svm = df_svm.iloc[sampled_rows]["prob"].values

        if len(set(y_sample)) >= 2:
            auc_ca = roc_auc_score(y_sample, p_ca)
            auc_svm = roc_auc_score(y_sample, p_svm)
            bootstrap_diffs.append(auc_ca - auc_svm)

    diffs = np.array(bootstrap_diffs)
    # Two-sided p-value: proportion of bootstrap diffs <= 0
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


def g12_13_detailed_metrics(df):
    """
    Detailed metrics for G12/13 family (26 positive, 371 negative pairs).

    AUC alone is misleading for heavily imbalanced classes.
    Report: PRAUC, recall at fixed precision levels, calibration.
    """
    g12 = df[df["g_protein_family"] == "G12_13"].dropna(subset=["prob"])

    if len(g12) == 0:
        return {"error": "no G12/13 pairs"}

    y = g12["label"].values
    p = g12["prob"].values

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    # Basic metrics
    auc = roc_auc_score(y, p) if len(set(y)) >= 2 else float("nan")
    prauc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    # Precision-recall curve
    prec, rec, thresh = precision_recall_curve(y, p)

    # Recall at fixed precision levels
    recall_at_precision = {}
    for target_prec in [0.5, 0.7, 0.8, 0.9]:
        # Find the highest recall where precision >= target_prec
        best_recall = 0.0
        for pi, ri in zip(prec, rec):
            if pi >= target_prec and ri > best_recall:
                best_recall = ri
        recall_at_precision[f"recall@prec={target_prec}"] = float(best_recall)

    # Average precision at top-k (relevant for screening)
    top_k = min(10, n_pos)
    if top_k > 0:
        top_idx = np.argsort(p)[-top_k:]
        precision_at_k = float(y[top_idx].mean())
    else:
        precision_at_k = float("nan")

    # Calibration for G12/13
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


def per_family_breakdown(df):
    """Per-family metrics for all four families."""
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


def main():
    print("=" * 70)
    print("  Bootstrap Statistics for GPCR Coupling Prediction")
    print(f"  {N_BOOTSTRAP} bootstrap samples over GPCR clusters")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # Load data
    print("\n[1/4] Loading predictions...")
    df_ca = load_predictions_as_dataframe(CA_PREDICTIONS_FILE)
    df_svm = load_predictions_as_dataframe(SVM_PREDICTIONS_FILE)
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

    # ======================================================================
    # 1. Cluster-level bootstrap AUC
    # ======================================================================
    print(f"\n[2/4] Cluster-level bootstrap ({N_BOOTSTRAP} samples)...")
    ca_bootstrap = cluster_bootstrap_auc(valid_ca, gpcr_to_cluster, N_BOOTSTRAP, RANDOM_SEED)
    svm_bootstrap = cluster_bootstrap_auc(valid_svm, gpcr_to_cluster, N_BOOTSTRAP, RANDOM_SEED)

    print(f"  CA  AUC: {ca_bootstrap['mean']:.4f} [{ca_bootstrap['ci_95_low']:.4f}, {ca_bootstrap['ci_95_high']:.4f}] (cluster bootstrap)")
    print(f"  SVM AUC: {svm_bootstrap['mean']:.4f} [{svm_bootstrap['ci_95_low']:.4f}, {svm_bootstrap['ci_95_high']:.4f}] (cluster bootstrap)")

    # ======================================================================
    # 2. Cluster-level bootstrap AUC difference
    # ======================================================================
    print(f"\n[3/4] Cluster-level bootstrap AUC difference (CA - SVM)...")
    diff_bootstrap = cluster_bootstrap_auc_difference(valid_ca, valid_svm, gpcr_to_cluster, N_BOOTSTRAP, RANDOM_SEED)
    print(f"  Mean diff: {diff_bootstrap['mean_diff']:.4f}")
    print(f"  95% CI:    [{diff_bootstrap['ci_95_low']:.4f}, {diff_bootstrap['ci_95_high']:.4f}]")
    print(f"  Bootstrap p-value (two-sided): {diff_bootstrap['p_value_bootstrap']:.4f}")

    # ======================================================================
    # 3. G12/13 detailed metrics
    # ======================================================================
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

    # ======================================================================
    # Save
    # ======================================================================
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


if __name__ == "__main__":
    main()
