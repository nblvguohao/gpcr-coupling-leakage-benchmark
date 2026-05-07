#!/usr/bin/env python3
"""
Paired cross-validation for GPCR-G protein coupling prediction.

Supports three evaluation protocols:
- Random CV: 5-fold stratified cross-validation
- Cluster-aware CV: Homology-cluster-based splitting (no sequence leakage)
- LOGPSO: Leave-one-G-protein-family-out

Usage:
    python src/paired_cross_validation.py \
        --pairing data/pairing_matrix_raw.csv \
        --clusters data/sequence_clusters.json \
        --gpcr-features data/gpcr_esm_features.json \
        --g-protein-features data/g_protein_esm_features.json \
        --icl-features data/icl_features.json \
        --output results/cv_results.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

GPROT_FAMILY_MAP = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}

STAT_KEYS = [
    "length", "mean_hydro", "std_hydro", "net_charge",
    "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio",
]


def load_gpcr_features(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    feats = {}
    for k, v in raw.items():
        arr = np.array(v)
        feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr
    return feats


def load_gprotein_features(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    feats = {}
    for subtype, info in raw.items():
        family = GPROT_FAMILY_MAP.get(subtype, subtype)
        arr = np.array(info["mean_pooling"])
        feats[subtype] = arr
        if family not in feats:
            feats[family] = arr
    return feats


def load_icl_features(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_gpcr_feat(gpcr_feats: dict, gid: str):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid:
        parts = gid.split("_", 1)
        if len(parts[0]) <= 2:
            feat = gpcr_feats.get(parts[1])
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                feat = gpcr_feats[key]
                break
    return feat


def resolve_icl_rec(icl_data: dict, gid: str):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break
    return rec


def get_icl_vector(icl_data: dict, gid: str, gpcr_feat_dim: int, mode: str = "stats"):
    rec = resolve_icl_rec(icl_data, gid)
    if rec is None:
        n_stats = len(STAT_KEYS)
        if mode == "stats":
            return np.zeros(n_stats * 2)
        elif mode == "full":
            return np.zeros(n_stats * 2 + gpcr_feat_dim * 2)
        else:
            return np.zeros(0)

    icl2_esm = np.array(rec.get("ICL2_esm", []))
    icl3_esm = np.array(rec.get("ICL3_esm", []))
    icl2_stats = rec.get("ICL2_stats", {})
    icl3_stats = rec.get("ICL3_stats", {})

    if icl2_esm.size == 0:
        icl2_esm = np.zeros(gpcr_feat_dim)
    if icl3_esm.size == 0:
        icl3_esm = np.zeros(gpcr_feat_dim)

    s2 = np.array([icl2_stats.get(k, 0.0) for k in STAT_KEYS])
    s3 = np.array([icl3_stats.get(k, 0.0) for k in STAT_KEYS])

    if mode == "stats":
        return np.concatenate([s2, s3])
    elif mode == "full":
        return np.concatenate([icl2_esm, s2, icl3_esm, s3])
    return np.zeros(0)


def build_vectors(df: pd.DataFrame, gpcr_feats: dict, gprot_feats: dict,
                  icl_data: dict, icl_mode: str = "none"):
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)

    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = resolve_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())

        if gpcr_feat is None:
            missing[f"gpcr_missing_{gid}"] += 1
            continue
        if gprot_feat is None:
            missing[f"gprot_missing_{gfam}"] += 1
            continue

        vec_parts = [gpcr_feat, gprot_feat]
        if icl_mode != "none" and icl_data:
            icl_vec = get_icl_vector(icl_data, gid, len(gpcr_feat), mode=icl_mode)
            vec_parts.append(icl_vec)

        X_list.append(np.concatenate(vec_parts))
        y_list.append(int(row["coupling"]))
        meta.append({
            "gpcr_id": gid,
            "g_protein_family": gfam,
            "cluster_id": int(row["cluster_id"]),
        })

    if missing:
        print(f"[WARN] Missing features: {dict(missing)}")
    if not X_list:
        return np.empty((0, 0)), np.array(y_list), meta
    return np.vstack(X_list), np.array(y_list), meta


def evaluate_svm(X_train, y_train, X_test, y_test,
                 kernel="rbf", C=10.0, class_weight="balanced"):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    svm = SVC(kernel=kernel, C=C, class_weight=class_weight,
              probability=True, random_state=42)
    svm.fit(Xtr, y_train)
    y_proba = svm.predict_proba(Xte)[:, 1]
    y_pred = svm.predict(Xte)

    metrics = {
        "auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) >= 2 else float("nan"),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, svm, scaler


def experiment_random_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs, fold_accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_aucs.append(m["auc"])
        fold_accs.append(m["accuracy"])
    return {
        "auc_mean": round(float(np.nanmean(fold_aucs)), 4),
        "auc_std": round(float(np.nanstd(fold_aucs)), 4),
        "acc_mean": round(float(np.nanmean(fold_accs)), 4),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


def experiment_cluster_cv(X, y, meta, cluster_list, n_folds=5):
    n = len(y)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}

    cluster_sizes = defaultdict(int)
    cluster_pos = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1
        cluster_pos[cid] += int(y[i])

    n_clusters = len(cluster_list)
    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(n_clusters), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]

    fold_aucs = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        if len(set(y[test_idx])) < 2:
            fold_aucs.append(float("nan"))
            continue
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_aucs.append(m["auc"])

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


def experiment_logpso(X, y, meta):
    families = sorted({m["g_protein_family"] for m in meta})
    if len(families) <= 1:
        return {}

    results = {}
    for test_fam in families:
        train_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] != test_fam]
        test_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] == test_fam]
        if len(train_idx) == 0 or len(test_idx) == 0 or len(set(y[test_idx])) < 2:
            continue
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        results[test_fam] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{k: round(float(v), 4) for k, v in m.items()},
        }
    return results


def run_all_experiments(X, y, meta, cluster_list, config_name: str):
    print(f"\n--- {config_name} ---")
    res_random = experiment_random_cv(X, y)
    res_cluster = experiment_cluster_cv(X, y, meta, cluster_list)
    res_logpso = experiment_logpso(X, y, meta)
    print(f"  Random CV    AUC={res_random['auc_mean']:.4f}±{res_random['auc_std']:.4f}")
    print(f"  Cluster CV   AUC={res_cluster['auc_mean']:.4f}±{res_cluster['auc_std']:.4f}")
    if res_logpso:
        avg_logpso = np.mean([r["auc"] for r in res_logpso.values()])
        print(f"  LOGPSO avg   AUC={avg_logpso:.4f}")
    return {
        "random_cv": res_random,
        "cluster_cv": res_cluster,
        "logpso_cv": res_logpso,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", required=True, help="CSV pairing matrix")
    parser.add_argument("--clusters", required=True, help="JSON sequence clusters")
    parser.add_argument("--gpcr-features", required=True, help="JSON GPCR ESM features")
    parser.add_argument("--g-protein-features", required=True, help="JSON G-protein ESM features")
    parser.add_argument("--icl-features", default="", help="Optional JSON ICL features")
    parser.add_argument("--output", required=True, help="Output JSON for CV results")
    args = parser.parse_args()

    df = pd.read_csv(args.pairing)
    with open(args.clusters) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    gpcr_feats = load_gpcr_features(args.gpcr_features)
    gprot_feats = load_gprotein_features(args.g_protein_features)
    icl_data = load_icl_features(args.icl_features) if args.icl_features else {}

    print(f"[INFO] Pairing matrix: {len(df)} rows")
    print(f"[INFO] Clusters: {clusters_data['n_clusters']}")
    print(f"[INFO] GPCR features: {len(gpcr_feats)}, G-protein features: {len(gprot_feats)}")

    all_results = {}

    # Baseline: global ESM only
    X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, icl_mode="none")
    print(f"[INFO] Baseline samples: {len(y)} (pos={int(y.sum())}, neg={len(y)-int(y.sum())}), dim={X.shape[1]}")
    if len(y) > 0:
        all_results["baseline"] = run_all_experiments(X, y, meta, cluster_list, "Baseline (Global ESM)")

    # ICL stats
    if icl_data:
        X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, icl_mode="stats")
        print(f"[INFO] ICL-stats samples: {len(y)}, dim={X.shape[1]}")
        if len(y) > 0:
            all_results["icl_stats"] = run_all_experiments(X, y, meta, cluster_list, "ICL stats")

        # ICL full
        X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, icl_mode="full")
        print(f"[INFO] ICL-full samples: {len(y)}, dim={X.shape[1]}")
        if len(y) > 0:
            all_results["icl_full"] = run_all_experiments(X, y, meta, cluster_list, "ICL full")

    all_results["n_samples"] = int(len(y))
    all_results["n_positive"] = int(y.sum())
    all_results["n_negative"] = int(len(y) - y.sum())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
