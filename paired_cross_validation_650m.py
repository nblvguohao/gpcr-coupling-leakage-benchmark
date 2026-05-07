#!/usr/bin/env python3
"""
650M ESM-2 增强版配对模型交叉验证。
使用 esm2_t33_650M_UR50D 1280-d 全局特征重新运行 baseline 与 icl_full。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "paired_cv_650m_results.json"


def load_features():
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
            "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
            "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
        }
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])

    return gpcr_feats, gprot_feats


def load_icl_features():
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


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


def get_icl_vector(icl_data, gid, gpcr_feat_dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break

    icl2_esm = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    icl3_esm = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    icl2_stats = rec.get("ICL2_stats", {}) if rec else {}
    icl3_stats = rec.get("ICL3_stats", {}) if rec else {}

    if icl2_esm.size == 0:
        icl2_esm = np.zeros(gpcr_feat_dim)
    if icl3_esm.size == 0:
        icl3_esm = np.zeros(gpcr_feat_dim)

    stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge", "pos_charge_ratio",
                 "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]
    icl2_stat_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_stat_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])

    return icl2_esm, icl2_stat_vec, icl3_esm, icl3_stat_vec


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="baseline"):
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
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

        base_vec = np.concatenate([gpcr_feat, gprot_feat])
        vec_parts = [base_vec]

        if mode == "icl_full":
            icl2_esm, icl2_stat, icl3_esm, icl3_stat = get_icl_vector(icl_data, gid, gpcr_feat_dim=1280)
            icl_vec = np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat])
            vec_parts.append(icl_vec)

        vec = np.concatenate(vec_parts)

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


def evaluate_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=10.0,
                 class_weight="balanced", tune_threshold=False,
                 threshold_val_size=0.15, random_state=42):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    svm = SVC(kernel=kernel, C=C, class_weight=class_weight,
              probability=True, random_state=random_state)
    svm.fit(Xtr, y_train)

    if tune_threshold:
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            Xtr, y_train, test_size=threshold_val_size,
            random_state=random_state, stratify=y_train)
        svm_val = SVC(kernel=kernel, C=C, class_weight=class_weight,
                      probability=True, random_state=random_state)
        svm_val.fit(X_tr_sub, y_tr_sub)
        y_val_proba = svm_val.predict_proba(X_val)[:, 1]
        thresholds = np.sort(np.unique(y_val_proba))
        if len(thresholds) == 0:
            thresholds = np.array([0.5])
        else:
            thresholds = np.concatenate([[0.0], thresholds, [1.0]])
        best_f1 = -1.0
        best_thresh = 0.5
        for thr in thresholds:
            y_val_pred = (y_val_proba >= thr).astype(int)
            if np.sum(y_val_pred) == 0 or np.sum(y_val_pred) == len(y_val_pred):
                continue
            f1 = f1_score(y_val, y_val_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thr
        y_proba = svm.predict_proba(Xte)[:, 1]
        y_pred = (y_proba >= best_thresh).astype(int)
    else:
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
    configs = [("SVM-RBF C=10 balanced", "rbf", 10.0, "balanced")]
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
    n = len(y)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}

    cluster_sizes = defaultdict(int)
    cluster_pos = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1
        cluster_pos[cid] += int(y[i])

    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]

    configs = [("SVM-RBF C=10 balanced", "rbf", 10.0, "balanced")]
    results = {}
    for name, kernel, C, cw in configs:
        fold_aucs = []
        for f in range(n_folds):
            test_clusters = set(fold_clusters[f])
            test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
            train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
            if len(set(y[test_idx])) < 2:
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
    families = sorted({m["g_protein_family"] for m in meta})
    if len(families) <= 1:
        return {}
    results = {}
    for test_fam in families:
        train_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] != test_fam]
        test_idx = [i for i, m in enumerate(meta) if m["g_protein_family"] == test_fam]
        if len(train_idx) == 0 or len(test_idx) == 0 or len(set(y[test_idx])) < 2:
            continue
        m, _, _ = evaluate_svm(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            kernel="rbf", C=10.0, class_weight="balanced", tune_threshold=True)
        m["used_threshold_tune"] = True
        results[test_fam] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{k: round(float(v), 4) for k, v in m.items()},
        }
    return results


def run_all_experiments(X, y, meta, cluster_list, label):
    print(f"\n{'='*70}")
    print(f"  实验组: {label}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"{'='*70}")
    rr = experiment_random_cv(X, y)
    rc = experiment_cluster_cv(X, y, meta, cluster_list)
    rl = experiment_logpso(X, y, meta)
    print(f"Random CV      AUC = {rr['SVM-RBF C=10 balanced']['auc_mean']:.4f} ± {rr['SVM-RBF C=10 balanced']['auc_std']:.4f}")
    print(f"Cluster-aware  AUC = {rc['SVM-RBF C=10 balanced']['auc_mean']:.4f} ± {rc['SVM-RBF C=10 balanced']['auc_std']:.4f}")
    if rl:
        logpso_mean = np.mean([v["auc"] for v in rl.values()])
        print(f"LOGPSO  mean   AUC = {logpso_mean:.4f}")
        for fam, r in rl.items():
            print(f"  leave-out {fam:8s} AUC={r['auc']:.4f}")
    return {"random_cv": rr, "cluster_cv": rc, "logpso_cv": rl}


def main():
    print("=" * 70)
    print("  650M ESM-2 配对交叉验证 (Baseline vs ICL-full)")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    print(f"[INFO] 配对矩阵: {len(df)} 行")

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]
    print(f"[INFO] 聚类: {clusters_data['n_clusters']} 簇")

    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    print(f"[INFO] GPCR 全局特征: {len(gpcr_feats)}, G蛋白: {len(gprot_feats)}, ICL: {len(icl_data)}")

    all_results = {}

    # Baseline
    X_base, y_base, meta_base = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="baseline")
    print(f"[INFO] Baseline 有效样本: {len(y_base)}")
    all_results["baseline"] = run_all_experiments(X_base, y_base, meta_base, cluster_list, "Baseline (1600-d global)")

    # ICL enhanced
    X_icl, y_icl, meta_icl = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="icl_full")
    print(f"[INFO] ICL-Full 有效样本: {len(y_icl)}")
    all_results["icl_full"] = run_all_experiments(X_icl, y_icl, meta_icl, cluster_list, "Global + ICL2/3 local (3536-d)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果保存: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("  650M vs 8M 快速对比摘要")
    print("=" * 70)
    print(f"{'策略':<25s} {'Random':>10s} {'Cluster':>10s} {'LOGPSO':>10s}")
    for key in ["baseline", "icl_full"]:
        rr = all_results[key]["random_cv"]["SVM-RBF C=10 balanced"]["auc_mean"]
        rc = all_results[key]["cluster_cv"]["SVM-RBF C=10 balanced"]["auc_mean"]
        rl = all_results[key]["logpso_cv"]
        rl_mean = round(float(np.mean([v["auc"] for v in rl.values()])), 4) if rl else float("nan")
        print(f"{key:<25s} {rr:>10.4f} {rc:>10.4f} {rl_mean:>10.4f}")


if __name__ == "__main__":
    main()
