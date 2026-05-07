#!/usr/bin/env python3
"""
SHAP attribution analysis for paired GPCR-G protein features.

Uses Permutation SHAP on a stratified subset to estimate global
feature importance. Splits importance into GPCR, G-protein, and
optional ICL regions.

Usage:
    python src/shap_attribution.py \
        --pairing data/pairing_matrix_raw.csv \
        --gpcr-features data/gpcr_esm_features.json \
        --g-protein-features data/g_protein_esm_features.json \
        --icl-features data/icl_features.json \
        --output-dir results/shap
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap
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
    if icl_path and Path(icl_path).exists():
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
    parser.add_argument("--icl-features", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-shap", type=int, default=50, help="Max samples for SHAP computation")
    args = parser.parse_args()

    df = pd.read_csv(args.pairing)
    gpcr_feats, gprot_feats, icl_data = load_features(
        args.gpcr_features, args.g_protein_features, args.icl_features
    )
    X, y, gpcr_dim = build_vectors(df, gpcr_feats, gprot_feats, icl_data)
    gprot_dim = gpcr_dim  # Same dimension

    print(f"[INFO] Samples: {len(y)} (pos={int(y.sum())}, neg={len(y)-int(y.sum())})")
    print(f"[INFO] Feature dim: {X.shape[1]} (GPCR={gpcr_dim}, G-prot={gprot_dim})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
              probability=True, random_state=42)
    svm.fit(X_scaled, y)
    print("[INFO] SVM trained")

    # Stratified subset for SHAP
    n_shap = min(args.n_shap, len(y))
    rng = np.random.RandomState(42)
    pos_idx = rng.permutation(np.where(y == 1)[0])[:n_shap // 2]
    neg_idx = rng.permutation(np.where(y == 0)[0])[:n_shap - len(pos_idx)]
    shap_idx = np.concatenate([pos_idx, neg_idx])
    X_shap = X_scaled[shap_idx]
    print(f"[INFO] Computing Permutation SHAP on n={len(shap_idx)} ...")

    masker = shap.maskers.Independent(X_scaled)
    explainer = shap.explainers.Permutation(svm.predict_proba, masker)
    shap_values = explainer(X_shap, max_evals=1300)
    sv_class1 = shap_values.values[:, :, 1]
    mean_abs_sv = np.abs(sv_class1).mean(axis=0)

    # Region summaries
    regions = {
        "GPCR_global": (0, gpcr_dim),
        "G_protein": (gpcr_dim, gpcr_dim + gprot_dim),
    }
    offset = gpcr_dim + gprot_dim
    if X.shape[1] > offset:
        regions["ICL2_esm"] = (offset, offset + gpcr_dim)
        offset += gpcr_dim
        regions["ICL2_stats"] = (offset, offset + len(STAT_KEYS))
        offset += len(STAT_KEYS)
        regions["ICL3_esm"] = (offset, offset + gpcr_dim)
        offset += gpcr_dim
        regions["ICL3_stats"] = (offset, offset + len(STAT_KEYS))

    print("\n--- SHAP importance by region ---")
    region_results = {}
    for name, (s, e) in regions.items():
        if e <= mean_abs_sv.shape[0]:
            imp = mean_abs_sv[s:e]
            region_results[name] = {
                "mean_abs_shap": round(float(imp.mean()), 6),
                "max_abs_shap": round(float(imp.max()), 6),
                "sum_abs_shap": round(float(imp.sum()), 6),
            }
            print(f"  {name:20s} mean={imp.mean():.6f}  max={imp.max():.6f}  sum={imp.sum():.6f}")

    # Top individual dimensions
    top_k = 20
    top_dims = np.argsort(mean_abs_sv)[::-1][:top_k]
    print(f"\n--- Top {top_k} dimensions ---")
    for rank, d in enumerate(top_dims, 1):
        print(f"  {rank:2d}. dim {d:4d} | SHAP={mean_abs_sv[d]:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_samples": int(len(y)),
        "n_shap_subset": int(len(shap_idx)),
        "feature_dim": int(X.shape[1]),
        "gpcr_dim": int(gpcr_dim),
        "gprot_dim": int(gprot_dim),
        "regions": region_results,
        "top_dims": [int(d) for d in top_dims],
        "top_shap": [round(float(mean_abs_sv[d]), 6) for d in top_dims],
    }
    with open(output_dir / "shap_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(output_dir / "mean_abs_shap.npy", mean_abs_sv)
    np.save(output_dir / "shap_values_class1.npy", sv_class1)
    print(f"\n[OK] Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
