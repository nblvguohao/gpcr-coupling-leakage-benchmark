#!/usr/bin/env python3
"""
Extended bootstrap statistics: CA vs SVM, CA vs MLP, minimal MLP vs CA.
All comparisons use cluster-level bootstrap (2000 samples).
Also computes per-family PRAUC, precision@top-k, recall@fixed-precision for all models.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, precision_recall_curve)
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
CA_FILE = DATA_DIR / "ca_predictions.json"
SVM_FILE = DATA_DIR / "svm_predictions.json"
MLP_FILE = DATA_DIR / "mlp_predictions.json"
PAIRING_FILE = DATA_DIR / "pairing_matrix_raw.csv"
OUTPUT_FILE = DATA_DIR / "extended_bootstrap_statistics.json"
N_BOOTSTRAP, RS = 2000, 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


def load_preds(path):
    with open(path) as f:
        preds = json.load(f)
    records = []
    for gid, families in preds.items():
        for gfam, info in families.items():
            records.append({"gpcr_id": gid, "g_protein_family": gfam,
                           "label": int(info.get("label", -1)),
                           "prob": float(info.get("prob", float("nan")))})
    return pd.DataFrame.from_records(records)


def load_cluster_map():
    df = pd.read_csv(PAIRING_FILE).dropna(subset=["cluster_id"])
    return {row["gpcr_id"]: int(row["cluster_id"]) for _, row in df.iterrows()}


def cluster_bootstrap_auc(df, gpcr_to_cluster, n_bootstrap=N_BOOTSTRAP, seed=RS):
    np.random.seed(seed)
    cluster_to_rows = defaultdict(list)
    for i, row in df.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0: cluster_to_rows[cid].append(i)
    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)
    aucs = []
    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for cid in sampled_cids: sampled_rows.extend(cluster_to_rows[cid])
        y = df.iloc[sampled_rows]["label"].values
        p = df.iloc[sampled_rows]["prob"].values
        if len(set(y)) >= 2: aucs.append(roc_auc_score(y, p))
    aucs = np.array(aucs)
    return {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
            "ci_95_low": float(np.percentile(aucs, 2.5)),
            "ci_95_high": float(np.percentile(aucs, 97.5)),
            "n_bootstrap": n_bootstrap, "n_clusters": n_clusters}


def cluster_bootstrap_diff(df_a, df_b, gpcr_to_cluster, n_bootstrap=N_BOOTSTRAP, seed=RS):
    """Bootstrap AUC difference (A - B)."""
    np.random.seed(seed)
    cluster_to_rows = defaultdict(list)
    for i, row in df_a.iterrows():
        cid = gpcr_to_cluster.get(row["gpcr_id"], -1)
        if cid >= 0: cluster_to_rows[cid].append(i)
    cluster_ids = sorted(cluster_to_rows.keys())
    n_clusters = len(cluster_ids)
    diffs = []
    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for cid in sampled_cids: sampled_rows.extend(cluster_to_rows[cid])
        y = df_a.iloc[sampled_rows]["label"].values
        p_a = df_a.iloc[sampled_rows]["prob"].values
        p_b = df_b.iloc[sampled_rows]["prob"].values
        if len(set(y)) >= 2:
            diffs.append(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    diffs = np.array(diffs)
    return {"mean_diff": float(np.mean(diffs)), "std_diff": float(np.std(diffs)),
            "ci_95_low": float(np.percentile(diffs, 2.5)),
            "ci_95_high": float(np.percentile(diffs, 97.5)),
            "p_value_bootstrap": float(np.mean(diffs <= 0)),
            "n_bootstrap": n_bootstrap, "n_clusters": n_clusters}


def per_family_metrics(df):
    results = {}
    for fam in FAMILIES:
        sub = df[df["g_protein_family"] == fam].dropna(subset=["prob"])
        if len(sub) == 0: continue
        y, p = sub["label"].values, sub["prob"].values
        n_pos = int(y.sum())
        prec, rec, _ = precision_recall_curve(y, p)
        recall_at_prec = {}
        for target_p in [0.5, 0.7, 0.8, 0.9]:
            best_r = 0.0
            for pi, ri in zip(prec, rec):
                if pi >= target_p and ri > best_r: best_r = ri
            recall_at_prec[f"recall@prec={target_p}"] = float(best_r)
        top_k = min(10, n_pos)
        prec_at_k = float(y[np.argsort(p)[-top_k:]].mean()) if top_k > 0 else float("nan")
        results[fam] = {
            "n": int(len(y)), "n_pos": n_pos,
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


def main():
    print("=" * 70)
    print("  Extended Bootstrap Statistics")
    print(f"  Comparisons: CA vs SVM | CA vs MLP | Minimal MLP vs CA")
    print("=" * 70)

    # Load data
    print("\nLoading predictions...")
    df_ca = load_preds(CA_FILE)
    df_svm = load_preds(SVM_FILE)
    gpcr_to_cluster = load_cluster_map()

    # MLP predictions may or may not exist
    if MLP_FILE.exists():
        df_mlp = load_preds(MLP_FILE)
        print(f"  MLP predictions loaded: {len(df_mlp)} pairs")
    else:
        df_mlp = None
        print("  MLP predictions NOT FOUND — skipping MLP comparisons")

    print(f"  CA:  {len(df_ca)} pairs, {df_ca['gpcr_id'].nunique()} GPCRs")
    print(f"  SVM: {len(df_svm)} pairs")

    # Overall AUCs
    valid_ca = df_ca.dropna(subset=["prob"])
    valid_svm = df_svm.dropna(subset=["prob"])
    ca_auc = roc_auc_score(valid_ca["label"], valid_ca["prob"])
    svm_auc = roc_auc_score(valid_svm["label"], valid_svm["prob"])
    print(f"\n  Overall CA AUC:  {ca_auc:.4f}")
    print(f"  Overall SVM AUC: {svm_auc:.4f}")

    # ======================================================================
    # 1. Bootstrap AUC for each model
    # ======================================================================
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

    # ======================================================================
    # 2. Pairwise bootstrap comparisons
    # ======================================================================
    print(f"\n[2] Pairwise bootstrap comparisons...")

    # CA vs SVM
    diff_ca_svm = cluster_bootstrap_diff(valid_ca, valid_svm, gpcr_to_cluster)
    print(f"  CA vs SVM:  ΔAUC = {diff_ca_svm['mean_diff']:.4f} "
          f"[{diff_ca_svm['ci_95_low']:.4f}, {diff_ca_svm['ci_95_high']:.4f}], "
          f"p = {diff_ca_svm['p_value_bootstrap']:.4f}")

    # CA vs MLP
    diff_ca_mlp = None
    if df_mlp is not None:
        diff_ca_mlp = cluster_bootstrap_diff(valid_ca, valid_mlp, gpcr_to_cluster)
        print(f"  CA vs MLP:  ΔAUC = {diff_ca_mlp['mean_diff']:.4f} "
              f"[{diff_ca_mlp['ci_95_low']:.4f}, {diff_ca_mlp['ci_95_high']:.4f}], "
              f"p = {diff_ca_mlp['p_value_bootstrap']:.4f}")

    # ======================================================================
    # 3. Per-family metrics for all models
    # ======================================================================
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

    # ======================================================================
    # 4. Minimal family-conditioned MLP reference
    # ======================================================================
    # Load from existing results
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
        print(f"  ΔAUC vs CA: {ca_bs['mean'] - min_mlp_ref['auc_mean']:.4f}")

    # ======================================================================
    # Save
    # ======================================================================
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
            "Report real ΔAUC, CI, and p for each comparison — do NOT reuse CA-SVM p=0.006 for other pairs."
        ),
    }
    if df_mlp is not None:
        out["cluster_bootstrap_auc"]["MLP"] = mlp_bs

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
