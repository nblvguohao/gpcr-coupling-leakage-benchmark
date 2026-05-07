#!/usr/bin/env python3
"""
Permutation importance for ICL2/3 statistical features.

Trains an SVM with full features (global ESM + G-protein + ICL2/3 stats + ICL2/3 local ESM)
and computes sklearn permutation_importance specifically on the ICL statistical dimensions.

Usage:
    python src/permutation_importance.py \
        --pairing data/pairing_matrix_raw.csv \
        --gpcr-features data/gpcr_esm_features.json \
        --g-protein-features data/g_protein_esm_features.json \
        --icl-features data/icl_features.json \
        --output results/permutation_importance.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
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


def load_features(gpcr_path: str, gprot_path: str, icl_path: str):
    with open(gpcr_path) as f:
        gpcr_raw = json.load(f)
    with open(gprot_path) as f:
        gprot_raw = json.load(f)

    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    gprot_feats = {}
    for subtype, info in gprot_raw.items():
        family = GPROT_FAMILY_MAP.get(subtype, subtype)
        arr = np.array(info["mean_pooling"])
        gprot_feats[subtype] = arr
        if family not in gprot_feats:
            gprot_feats[family] = arr

    icl_data = {}
    if Path(icl_path).exists():
        with open(icl_path) as f:
            icl_data = json.load(f)
    return gpcr_feats, gprot_feats, icl_data


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


def build_vectors(df: pd.DataFrame, gpcr_feats: dict, gprot_feats: dict, icl_data: dict):
    X_list, y_list = [], []
    gpcr_dim = None
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = resolve_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
            continue
        if gpcr_dim is None:
            gpcr_dim = len(gpcr_feat)

        vec_parts = [gpcr_feat, gprot_feat]
        rec = resolve_icl_rec(icl_data, gid)
        if rec:
            icl2 = np.array(rec.get("ICL2_esm", []))
            icl3 = np.array(rec.get("ICL3_esm", []))
            if icl2.size == 0:
                icl2 = np.zeros(gpcr_dim)
            if icl3.size == 0:
                icl3 = np.zeros(gpcr_dim)
            s2 = np.array([rec.get("ICL2_stats", {}).get(k, 0.0) for k in STAT_KEYS])
            s3 = np.array([rec.get("ICL3_stats", {}).get(k, 0.0) for k in STAT_KEYS])
            vec_parts.extend([icl2, s2, icl3, s3])

        X_list.append(np.concatenate(vec_parts))
        y_list.append(int(row["coupling"]))

    if not X_list:
        return np.empty((0, 0)), np.array(y_list), gpcr_dim or 320
    return np.vstack(X_list), np.array(y_list), gpcr_dim or 320


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", required=True)
    parser.add_argument("--gpcr-features", required=True)
    parser.add_argument("--g-protein-features", required=True)
    parser.add_argument("--icl-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-repeats", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.pairing)
    gpcr_feats, gprot_feats, icl_data = load_features(
        args.gpcr_features, args.g_protein_features, args.icl_features
    )
    X, y, gpcr_dim = build_vectors(df, gpcr_feats, gprot_feats, icl_data)
    gprot_dim = gpcr_dim
    n_stats = len(STAT_KEYS)

    print(f"[INFO] Samples: {len(y)} (pos={int(y.sum())}, neg={len(y)-int(y.sum())})")
    print(f"[INFO] Feature dim: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
              probability=True, random_state=42)
    svm.fit(X_scaled, y)
    print("[INFO] SVM trained")

    # Permutation importance on a subset for speed
    rng = np.random.RandomState(42)
    n_subset = min(50, len(y))
    subset_idx = rng.choice(len(y), size=n_subset, replace=False)
    result = permutation_importance(
        svm, X_scaled[subset_idx], y[subset_idx],
        n_repeats=args.n_repeats, random_state=42, scoring="roc_auc", n_jobs=1
    )

    importances = result.importances_mean
    n_total = X.shape[1]

    # Locate ICL stat regions
    # Layout: [gpcr_dim][gprot_dim][icl2_esm][icl2_stats][icl3_esm][icl3_stats]
    icl2_stats_start = gpcr_dim + gprot_dim + gpcr_dim
    icl2_stats_end = icl2_stats_start + n_stats
    icl3_stats_start = icl2_stats_end + gpcr_dim
    icl3_stats_end = icl3_stats_start + n_stats

    icl2_imp = importances[icl2_stats_start:icl2_stats_end]
    icl3_imp = importances[icl3_stats_start:icl3_stats_end]

    print("\n--- ICL2 statistical features permutation importance (AUC drop) ---")
    icl2_results = {}
    for rank, d in enumerate(np.argsort(icl2_imp)[::-1], 1):
        name = STAT_KEYS[d]
        icl2_results[name] = round(float(icl2_imp[d]), 6)
        print(f"  {rank}. {name:20s}  {icl2_imp[d]:.6f}")

    print("\n--- ICL3 statistical features permutation importance (AUC drop) ---")
    icl3_results = {}
    for rank, d in enumerate(np.argsort(icl3_imp)[::-1], 1):
        name = STAT_KEYS[d]
        icl3_results[name] = round(float(icl3_imp[d]), 6)
        print(f"  {rank}. {name:20s}  {icl3_imp[d]:.6f}")

    # Region averages for comparison
    regions = {
        "GPCR_global": importances[:gpcr_dim].mean(),
        "G_protein": importances[gpcr_dim:gpcr_dim + gprot_dim].mean(),
        "ICL2_esm": importances[gpcr_dim + gprot_dim:gpcr_dim + gprot_dim + gpcr_dim].mean(),
        "ICL2_stats": icl2_imp.mean(),
        "ICL3_esm": importances[icl3_stats_start - gpcr_dim:icl3_stats_start].mean(),
        "ICL3_stats": icl3_imp.mean(),
    }
    print("\n--- Region mean permutation importance ---")
    for name, val in regions.items():
        print(f"  {name:20s}  {val:.6f}")

    results = {
        "n_samples": int(len(y)),
        "n_subset": int(n_subset),
        "n_repeats": args.n_repeats,
        "icl2_stats": icl2_results,
        "icl3_stats": icl3_results,
        "region_means": {k: round(float(v), 6) for k, v in regions.items()},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
