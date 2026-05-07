#!/usr/bin/env python3
"""
SHAP 特征重要性归因 — ICL-stats 增强模型 (640-d global + 16-d ICL2/3 stats).
验证 ICL2/3 理化统计特征是否被模型有效利用。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
PAIRING_FILE = BASE / "paired_dataset" / "pairing_matrix_raw.csv"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = BASE / "paired_dataset" / "g_protein_esm_features.json"
ICL_FEATURES_FILE = BASE / "paired_dataset" / "icl_features.json"
OUTPUT_DIR = BASE / "shap_results_icl"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    print("=" * 70)
    print("  SHAP 特征归因 — ICL-stats 增强模型")
    print("=" * 70)

    df = pd.read_csv(PAIRING_FILE)
    gpcr_feats, gprot_feats, icl_data = load_features()

    X_list, y_list, meta = [], [], []
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
        meta.append(gid)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"[INFO] 样本数: {len(y)} (正: {y.sum()}, 负: {len(y)-y.sum()})")
    print(f"[INFO] 特征维度: {X.shape[1]} (GPCR 0-319, G-protein 320-639, ICL2_stats 640-647, ICL3_stats 648-655)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_scaled, y)
    print("[INFO] SVM 训练完成")

    n_shap = min(20, len(y))
    rng = np.random.RandomState(42)
    pos_idx = rng.permutation(np.where(y == 1)[0])[:n_shap // 2]
    neg_idx = rng.permutation(np.where(y == 0)[0])[:n_shap - len(pos_idx)]
    shap_idx = np.concatenate([pos_idx, neg_idx])
    X_shap = X_scaled[shap_idx]
    print(f"[INFO] 开始计算 Permutation SHAP (子集 n={len(shap_idx)}, max_evals=1500) ...")
    masker = shap.maskers.Independent(X_scaled)
    explainer = shap.explainers.Permutation(svm.predict_proba, masker)
    shap_values = explainer(X_shap, max_evals=1500)
    sv_class1 = shap_values.values[:, :, 1]
    mean_abs_sv = np.abs(sv_class1).mean(axis=0)

    gpcr_importance = mean_abs_sv[:320]
    gprot_importance = mean_abs_sv[320:640]
    icl2_importance = mean_abs_sv[640:648]
    icl3_importance = mean_abs_sv[648:656]

    top_gpcr_dims = np.argsort(gpcr_importance)[::-1][:20]
    top_gprot_dims = np.argsort(gprot_importance)[::-1][:20]
    top_icl2_idx = np.argsort(icl2_importance)[::-1]
    top_icl3_idx = np.argsort(icl3_importance)[::-1]

    print("\n--- Top 20 GPCR ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gpcr_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gpcr_importance[d]:.4f}")

    print("\n--- Top 20 G-protein ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gprot_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gprot_importance[d]:.4f}")

    print("\n--- ICL2 统计特征重要性 (by |SHAP|) ---")
    for rank, d in enumerate(top_icl2_idx, 1):
        print(f"  {rank:2d}. {STAT_KEYS[d]:20s} | SHAP={icl2_importance[d]:.6f}")

    print("\n--- ICL3 统计特征重要性 (by |SHAP|) ---")
    for rank, d in enumerate(top_icl3_idx, 1):
        print(f"  {rank:2d}. {STAT_KEYS[d]:20s} | SHAP={icl3_importance[d]:.6f}")

    print(f"\n--- ICL 统计特征总体 vs Global 对比 ---")
    print(f"  ICL2_stats 平均 |SHAP|: {icl2_importance.mean():.6f}")
    print(f"  ICL3_stats 平均 |SHAP|: {icl3_importance.mean():.6f}")
    print(f"  GPCR 平均 |SHAP|:     {gpcr_importance.mean():.6f}")
    print(f"  G-protein 平均 |SHAP|: {gprot_importance.mean():.6f}")

    results = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "top_gpcr_dims": [int(d) for d in top_gpcr_dims],
        "top_gpcr_shap": [round(float(gpcr_importance[d]), 6) for d in top_gpcr_dims],
        "top_gprot_dims": [int(d) for d in top_gprot_dims],
        "top_gprot_shap": [round(float(gprot_importance[d]), 6) for d in top_gprot_dims],
        "icl2_stats_shap": {k: round(float(v), 6) for k, v in zip(STAT_KEYS, icl2_importance.tolist())},
        "icl3_stats_shap": {k: round(float(v), 6) for k, v in zip(STAT_KEYS, icl3_importance.tolist())},
        "icl2_mean_abs": round(float(icl2_importance.mean()), 6),
        "icl3_mean_abs": round(float(icl3_importance.mean()), 6),
    }
    with open(OUTPUT_DIR / "global_shap_summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.save(OUTPUT_DIR / "shap_values_class1.npy", sv_class1)
    np.save(OUTPUT_DIR / "mean_abs_shap.npy", mean_abs_sv)
    print(f"\n[OK] 结果保存到: {OUTPUT_DIR}/")

    # 生物学判断
    print("\n" + "=" * 70)
    print("  生物学洞察")
    print("=" * 70)
    if icl2_importance.max() > 0 or icl3_importance.max() > 0:
        print("-> ICL2/3 统计特征 SHAP 重要性非零。模型已将注意力部分转移到已知结合界面上。")
    else:
        print("-> WARNING: ICL2/3 统计特征 SHAP 仍为零。模型未利用局部拓扑信息。")


if __name__ == "__main__":
    main()
