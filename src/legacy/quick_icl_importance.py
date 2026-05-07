#!/usr/bin/env python3
"""
快速计算 ICL-stats 模型中 ICL2/3 统计特征的全局排列重要性。
使用 sklearn.permutation_importance（比 SHAP PermutationExplainer 快 100 倍以上）。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

BASE = Path(__file__).parent
PAIRING_FILE = BASE / "paired_dataset" / "pairing_matrix_raw.csv"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = BASE / "paired_dataset" / "g_protein_esm_features.json"
ICL_FEATURES_FILE = BASE / "paired_dataset" / "icl_features.json"

STAT_KEYS = [
    "length", "mean_hydro", "std_hydro", "net_charge",
    "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio",
]


def load_features():
    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    with open(G_PROTEIN_FEATURES_FILE) as f:
        gprot_raw = json.load(f)

    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    family_map = {
        "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
        "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
    }
    gprot_feats = {}
    for subtype, info in gprot_raw.items():
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])

    icl_data = {}
    if ICL_FEATURES_FILE.exists():
        with open(ICL_FEATURES_FILE) as f:
            icl_data = json.load(f)
    return gpcr_feats, gprot_feats, icl_data


def get_gpcr_feat(gpcr_feats, gid):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        base = gid.split("_", 1)[1]
        feat = gpcr_feats.get(base)
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                feat = gpcr_feats[key]
                break
    return feat


def get_icl_stats(icl_data, gid):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break
    icl2 = rec.get("ICL2_stats", {}) if rec else {}
    icl3 = rec.get("ICL3_stats", {}) if rec else {}
    vec = []
    for k in STAT_KEYS:
        vec.append(icl2.get(k, 0.0))
    for k in STAT_KEYS:
        vec.append(icl3.get(k, 0.0))
    return np.array(vec)


def main():
    df = pd.read_csv(PAIRING_FILE)
    gpcr_feats, gprot_feats, icl_data = load_features()

    X_list, y_list = [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
            continue
        icl_vec = get_icl_stats(icl_data, gid)
        X_list.append(np.concatenate([gpcr_feat, gprot_feat, icl_vec]))
        y_list.append(int(row["coupling"]))

    X = np.array(X_list)
    y = np.array(y_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_scaled, y)

    # 排列重要性（仅针对 ICL 维度 640-655）
    rng = np.random.RandomState(42)
    n_subset = min(50, len(y))
    subset_idx = rng.choice(len(y), size=n_subset, replace=False)
    result = permutation_importance(
        svm, X_scaled[subset_idx], y[subset_idx],
        n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=1
    )

    importances = result.importances_mean
    icl2_imp = importances[640:648]
    icl3_imp = importances[648:656]

    print("--- ICL2 统计特征排列重要性 (AUC drop) ---")
    for rank, d in enumerate(np.argsort(icl2_imp)[::-1], 1):
        print(f"  {rank}. {STAT_KEYS[d]:20s}  {icl2_imp[d]:.6f}")

    print("\n--- ICL3 统计特征排列重要性 (AUC drop) ---")
    for rank, d in enumerate(np.argsort(icl3_imp)[::-1], 1):
        print(f"  {rank}. {STAT_KEYS[d]:20s}  {icl3_imp[d]:.6f}")

    print(f"\n--- 对比: GPCR全局维度平均重要性 ---")
    print(f"  GPCR dims (0-319) mean: {importances[:320].mean():.6f}")
    print(f"  G-prot dims (320-639) mean: {importances[320:640].mean():.6f}")
    print(f"  ICL2_stats mean: {icl2_imp.mean():.6f}")
    print(f"  ICL3_stats mean: {icl3_imp.mean():.6f}")


if __name__ == "__main__":
    main()
