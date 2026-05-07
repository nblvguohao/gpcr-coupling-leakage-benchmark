#!/usr/bin/env python3
"""
配对模型交叉验证 (Strategy C Phase 1 Week 3)。

核心逻辑:
- 输入: paired_dataset/pairing_matrix_raw.csv + sequence_clusters.json
- 特征: GPCR ESM-2 + G protein ESM-2 (+ 可选结构特征)
- CV 策略:
  1) Random CV (基线)
  2) Cluster-aware CV (按 GPCR cluster 拆分, 同一 GPCR 所有配对必须在同 fold)
  3) LOGPSO (Leave-One-G-Protein-Subtype-Out)
- 模型: SVM-RBF / SVM-Linear (Grid Search), 当前样本量 <150 暂不启用 Cross-Attention
- 输出: paired_dataset/paired_cv_results.json
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
    recall_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "paired_cv_results.json"


def load_features():
    """加载 GPCR 和 G 蛋白的 ESM-2 mean-pooling 特征。"""
    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    with open(G_PROTEIN_FEATURES_FILE) as f:
        gprot_raw = json.load(f)

    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    gprot_feats = {}
    for subtype, info in gprot_raw.items():
        family_map = {
            "GNAQ": "Gq",
            "GNAI1": "Gi",
            "GNAI2": "Gi",
            "GNAI3": "Gi",
            "GNAS": "Gs",
            "GNA12": "G12_13",
            "GNA13": "G12_13",
        }
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])

    return gpcr_feats, gprot_feats


def build_paired_vectors(df: pd.DataFrame, gpcr_feats: dict, gprot_feats: dict):
    """
    为 pairing matrix 的每一行构造特征向量。
    若某 GPCR 或 G 蛋白特征缺失，返回 None 并打印警告。
    """
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        # 处理 prefix ID: 如果 gpcr_feats 中没有完整 ID，尝试去掉 prefix
        gpcr_feat = gpcr_feats.get(gid)
        if gpcr_feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
            base = gid.split("_", 1)[1]
            gpcr_feat = gpcr_feats.get(base)
        if gpcr_feat is None:
            for key in gpcr_feats:
                if "_" in key and key.split("_", 1)[1] == gid:
                    gpcr_feat = gpcr_feats[key]
                    break

        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            # 尝试大小写变体
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())

        if gpcr_feat is None:
            missing[f"gpcr_missing_{gid}"] += 1
            continue
        if gprot_feat is None:
            missing[f"gprot_missing_{gfam}"] += 1
            continue

        vec = np.concatenate([gpcr_feat, gprot_feat])
        X_list.append(vec)
        y_list.append(int(row["coupling"]))
        meta.append({
            "gpcr_id": gid,
            "g_protein_family": gfam,
            "cluster_id": int(row["cluster_id"]),
        })

    if missing:
        print(f"[WARN] 缺失特征统计 (去重后): {dict(missing)}")
    return np.array(X_list), np.array(y_list), meta


def evaluate_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=10.0, class_weight="balanced"):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    svm = SVC(kernel=kernel, C=C, class_weight=class_weight, probability=True, random_state=42)
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
    configs = [
        ("SVM-RBF C=10 balanced", "rbf", 10.0, "balanced"),
        ("SVM-RBF C=1 balanced", "rbf", 1.0, "balanced"),
        ("SVM-Linear C=1 balanced", "linear", 1.0, "balanced"),
    ]
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, kernel, C, cw in configs:
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx], kernel=kernel, C=C, class_weight=cw)
            fold_aucs.append(m["auc"])
        results[name] = {
            "auc_mean": round(float(np.nanmean(fold_aucs)), 4),
            "auc_std": round(float(np.nanstd(fold_aucs)), 4),
            "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        }
    return results


def experiment_cluster_cv(X, y, meta, cluster_list, n_folds=5):
    """Cluster-aware CV: 按 GPCR cluster 拆分，同一 GPCR 所有配对必须在同一 fold。"""
    # 建立 meta index -> cluster_id
    n = len(y)
    sample_to_cluster = {}
    for i in range(n):
        sample_to_cluster[i] = meta[i]["cluster_id"]

    # cluster 列表已经是 cluster_id 顺序
    n_clusters = len(cluster_list)
    # 计算每个 cluster 的样本数
    cluster_sizes = defaultdict(int)
    cluster_pos = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1
        cluster_pos[cid] += int(y[i])

    # 贪心分配 cluster 到 fold (平衡正负和总样本数)
    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    fold_pos = [0] * n_folds
    sorted_cids = sorted(range(n_clusters), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        # 分配到总样本数最小的 fold
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]
        fold_pos[target] += cluster_pos[cid]

    configs = [
        ("SVM-RBF C=10 balanced", "rbf", 10.0, "balanced"),
        ("SVM-RBF C=1 balanced", "rbf", 1.0, "balanced"),
        ("SVM-Linear C=1 balanced", "linear", 1.0, "balanced"),
    ]
    results = {}
    for name, kernel, C, cw in configs:
        fold_aucs = []
        for f in range(n_folds):
            test_clusters = set(fold_clusters[f])
            test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
            train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
            if len(set(y[test_idx])) < 2:
                print(f"    [WARN] Fold {f+1} 测试集单一标签，跳过 AUC")
                fold_aucs.append(float("nan"))
                continue
            m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx], kernel=kernel, C=C, class_weight=cw)
            fold_aucs.append(m["auc"])
        valid = [a for a in fold_aucs if not np.isnan(a)]
        results[name] = {
            "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
            "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
            "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        }
    return results


def experiment_logpso(X, y, meta):
    """Leave-One-G-Protein-Subtype-Out: 留出一个 G protein subtype 做测试。"""
    # 确定存在的 G protein family
    families = sorted({m["g_protein_family"] for m in meta})
    if len(families) <= 1:
        print("[WARN] 仅发现 1 个 G protein family，无法执行 LOGPSO。")
        return {}

    results = {}
    for test_fam in families:
        train_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] != test_fam]
        test_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] == test_fam]
        if len(train_idx) == 0 or len(test_idx) == 0 or len(set(y[test_idx])) < 2:
            print(f"    [WARN] LOGPSO {test_fam} 无法评估 (测试集单一标签或为空)")
            continue
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                                kernel="rbf", C=10.0, class_weight="balanced")
        results[test_fam] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{k: round(float(v), 4) for k, v in m.items()},
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm-only", action="store_true", help="仅运行 SVM 基线")
    args = parser.parse_args()

    print("=" * 70)
    print("  配对模型交叉验证 (Paired Cross-Validation)")
    print("=" * 70)

    # 加载数据
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    print(f"[INFO] 加载配对矩阵: {len(df)} 行")

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]
    print(f"[INFO] 加载聚类结果: {clusters_data['n_clusters']} 个簇")

    gpcr_feats, gprot_feats = load_features()
    print(f"[INFO] GPCR 特征数: {len(gpcr_feats)}, G蛋白特征数: {len(gprot_feats)}")

    X, y, meta = build_paired_vectors(df, gpcr_feats, gprot_feats)
    print(f"[INFO] 有效样本: {len(y)} (正: {int(y.sum())}, 负: {len(y)-int(y.sum())})")
    print(f"[INFO] 特征维度: {X.shape[1]}")

    if len(y) == 0:
        print("[ERROR] 没有有效样本，无法继续。")
        return

    # 实验 1: Random CV
    res_random = experiment_random_cv(X, y)
    print("\n--- Random CV (SVM) ---")
    for name, r in res_random.items():
        print(f"  {name:35s} AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    # 实验 2: Cluster-aware CV
    res_cluster = experiment_cluster_cv(X, y, meta, cluster_list)
    print("\n--- Cluster-aware CV (SVM) ---")
    for name, r in res_cluster.items():
        print(f"  {name:35s} AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    # 实验 3: LOGPSO
    res_logpso = experiment_logpso(X, y, meta)
    if res_logpso:
        print("\n--- LOGPSO CV (SVM-RBF C=10) ---")
        for fam, r in res_logpso.items():
            print(f"  留出 {fam:8s} 测试 AUC={r['auc']:.4f} (train={r['n_train']}, test={r['n_test']})")

    # 保存结果
    all_results = {
        "random_cv": res_random,
        "cluster_cv": res_cluster,
        "logpso_cv": res_logpso,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "feature_dim": int(X.shape[1]),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果保存: {OUTPUT_FILE}")

    # Go/No-Go checkpoint 提示
    print("\n" + "=" * 70)
    print("  Week 4 Go/No-Go Checkpoint (早期数据)")
    print("=" * 70)
    best_cluster_auc = res_cluster.get("SVM-RBF C=10 balanced", {}).get("auc_mean", 0.0)
    if best_cluster_auc >= 0.78:
        print(f"  [PASS] Cluster-aware AUC = {best_cluster_auc:.4f} >= 0.78")
    elif best_cluster_auc >= 0.75:
        print(f"  [WARN] Cluster-aware AUC = {best_cluster_auc:.4f} (0.75-0.78)")
    else:
        print(f"  [FAIL] Cluster-aware AUC = {best_cluster_auc:.4f} < 0.75")
        print("         建议: 检查标签噪声、G蛋白特征是否正确加载。")


if __name__ == "__main__":
    main()
