#!/usr/bin/env python3
"""
SHAP 特征重要性归因 — 配对数据集 (315 样本, 4 G-protein families)
验证 G 蛋白维度是否有非零 SHAP 重要性。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
PAIRING_FILE = BASE / "paired_dataset" / "pairing_matrix_raw.csv"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = BASE / "paired_dataset" / "g_protein_esm_features.json"
OUTPUT_DIR = BASE / "shap_results_paired"
OUTPUT_DIR.mkdir(exist_ok=True)


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
        "GNAQ": "Gq",
        "GNAI1": "Gi",
        "GNAI2": "Gi",
        "GNAI3": "Gi",
        "GNAS": "Gs",
        "GNA12": "G12_13",
        "GNA13": "G12_13",
    }
    gprot_feats = {}
    for subtype, info in gprot_raw.items():
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])
    return gpcr_feats, gprot_feats


def main():
    print("=" * 70)
    print("  SHAP 特征归因 — 配对数据集")
    print("=" * 70)

    df = pd.read_csv(PAIRING_FILE)
    gpcr_feats, gprot_feats = load_features()

    X_list, y_list = [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = gpcr_feats.get(gid)
        if gpcr_feat is None:
            for key in gpcr_feats:
                if "_" in key and key.split("_", 1)[1] == gid:
                    gpcr_feat = gpcr_feats[key]
                    break
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gpcr_feat is None or gprot_feat is None:
            continue
        X_list.append(np.concatenate([gpcr_feat, gprot_feat]))
        y_list.append(int(row["coupling"]))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"[INFO] 样本数: {len(y)} (正: {y.sum()}, 负: {len(y)-y.sum()})")
    print(f"[INFO] 特征维度: {X.shape[1]} (GPCR 0-319, G-protein 320-639)")

    # Train SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_scaled, y)
    print("[INFO] SVM 训练完成")

    # SHAP (使用 stratified 子集加速)
    n_shap = min(50, len(y))
    rng = np.random.RandomState(42)
    pos_idx = rng.permutation(np.where(y == 1)[0])[:n_shap // 2]
    neg_idx = rng.permutation(np.where(y == 0)[0])[:n_shap - len(pos_idx)]
    shap_idx = np.concatenate([pos_idx, neg_idx])
    X_shap = X_scaled[shap_idx]
    print(f"[INFO] 开始计算 Permutation SHAP (子集 n={len(shap_idx)}) ...")
    masker = shap.maskers.Independent(X_scaled)
    explainer = shap.explainers.Permutation(svm.predict_proba, masker)
    shap_values = explainer(X_shap, max_evals=1300)
    sv_class1 = shap_values.values[:, :, 1]
    mean_abs_sv = np.abs(sv_class1).mean(axis=0)

    gpcr_importance = mean_abs_sv[:320]
    gprot_importance = mean_abs_sv[320:]

    top_gpcr_dims = np.argsort(gpcr_importance)[::-1][:20]
    top_gprot_dims = np.argsort(gprot_importance)[::-1][:20]

    print("\n--- Top 20 GPCR ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gpcr_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gpcr_importance[d]:.4f}")

    print("\n--- Top 20 G-protein ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gprot_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gprot_importance[d]:.4f}")

    print(f"\n--- G-protein 维度统计 ---")
    print(f"  平均 |SHAP|: {gprot_importance.mean():.6f}")
    print(f"  最大 |SHAP|: {gprot_importance.max():.6f}")
    print(f"  >0 的比例:   {(gprot_importance > 0).sum()} / 320")

    results = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "gprot_mean_abs_shap": round(float(gprot_importance.mean()), 6),
        "gprot_max_abs_shap": round(float(gprot_importance.max()), 6),
        "gprot_nonzero_count": int((gprot_importance > 0).sum()),
        "top_gpcr_dims": [int(d) for d in top_gpcr_dims],
        "top_gpcr_shap": [round(float(gpcr_importance[d]), 6) for d in top_gpcr_dims],
        "top_gprot_dims": [int(d) for d in top_gprot_dims],
        "top_gprot_shap": [round(float(gprot_importance[d]), 6) for d in top_gprot_dims],
    }
    with open(OUTPUT_DIR / "global_shap_summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.save(OUTPUT_DIR / "shap_values_class1.npy", sv_class1)
    np.save(OUTPUT_DIR / "mean_abs_shap.npy", mean_abs_sv)
    print(f"\n[OK] 结果保存到: {OUTPUT_DIR}/")

    print("\n" + "=" * 70)
    print("  生物学洞察")
    print("=" * 70)
    if results["gprot_nonzero_count"] > 0:
        print("-> G-protein 维度 SHAP 重要性 非零。模型确实在学习 G 蛋白信息。")
    else:
        print("-> WARNING: G-protein 维度 SHAP 仍为零。模型退化为单方分类。")


if __name__ == "__main__":
    main()
