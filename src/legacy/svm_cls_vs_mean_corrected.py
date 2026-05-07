#!/usr/bin/env python3
"""
控制变量比较：CLS token vs Mean pooling on SVM (使用真实 GNAQ P50148)
消除模型差异带来的混淆，公正评估两种特征聚合策略。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
CLS_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples_cls.json"
LABELS_FILE = BASE / "merged_dataset" / "extended_labels.json"
GNAQ_FEATURE_FILE = BASE / "server_sync" / "extended_data" / "features" / "gnaq_esm_features.json"
OUTPUT_FILE = BASE / "svm_cls_vs_mean_corrected.json"


def deduplicate_and_load(features_raw, labels, gnaq_mean, gnaq_cls):
    """去重并返回 mean / cls 两种特征矩阵。"""
    prefixed = set()
    for k in labels:
        if "_" in k and len(k.split("_")[0]) <= 2:
            base = k.split("_", 1)[1]
            if base in labels:
                prefixed.add(k)

    keep_ids = [k for k in labels if k not in prefixed]

    X_mean, X_cls, y_list = [], [], []
    for uid in keep_ids:
        feat = np.array(features_raw[uid])
        if feat.ndim == 2:
            X_mean.append(feat.mean(axis=0))
        else:
            X_mean.append(feat)
        y_list.append(labels[uid])

    # CLS token 特征
    with open(CLS_FEATURES_FILE) as f:
        cls_dict = json.load(f)
    for uid in keep_ids:
        X_cls.append(np.array(cls_dict[uid]))

    # 拼接真实 GNAQ
    gq_mean = np.array(gnaq_mean)
    gq_cls = np.array(gnaq_cls)

    X_mean_paired = np.concatenate([
        np.array(X_mean),
        np.tile(gq_mean, (len(X_mean), 1))
    ], axis=1)
    X_cls_paired = np.concatenate([
        np.array(X_cls),
        np.tile(gq_cls, (len(X_cls), 1))
    ], axis=1)

    y = np.array(y_list)
    return X_mean_paired, X_cls_paired, y, keep_ids


def evaluate_svm(X, y, kernel="rbf", C=10.0, class_weight="balanced"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        ytr, yte = y[train_idx], y[test_idx]

        svm = SVC(kernel=kernel, C=C, class_weight=class_weight,
                  probability=True, random_state=42)
        svm.fit(Xtr, ytr)

        proba = svm.predict_proba(Xte)[:, 1]
        pred = svm.predict(Xte)

        fold_aucs.append(roc_auc_score(yte, proba))
        fold_accs.append(accuracy_score(yte, pred))
        fold_f1s.append(f1_score(yte, pred, zero_division=0))

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "acc_mean": float(np.mean(fold_accs)),
        "f1_mean": float(np.mean(fold_f1s)),
        "fold_aucs": [round(x, 4) for x in fold_aucs],
    }


def main():
    print("=" * 65)
    print("  SVM 控制变量比较: CLS vs Mean (真实 GNAQ P50148)")
    print("=" * 65)

    # 加载数据
    with open(FEATURES_FILE) as f:
        features_raw = json.load(f)
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    with open(GNAQ_FEATURE_FILE) as f:
        gnaq_esm = json.load(f)

    X_mean, X_cls, y, ids = deduplicate_and_load(
        features_raw, labels, gnaq_esm["mean_pooling"], gnaq_esm["cls_token"]
    )
    print(f"[INFO] 去重后样本数: {len(y)} (正: {y.sum()}, 负: {len(y)-y.sum()})")

    results = {}

    # 1. Mean pooling
    print("\n--- Mean Pooling ---")
    r_mean = evaluate_svm(X_mean, y, kernel="rbf", C=10.0, class_weight="balanced")
    results["mean_pooling"] = r_mean
    print(f"AUC: {r_mean['auc_mean']:.4f} ± {r_mean['auc_std']:.4f}")
    print(f"Acc: {r_mean['acc_mean']:.4f} | F1: {r_mean['f1_mean']:.4f}")

    # 2. CLS token
    print("\n--- CLS Token ---")
    r_cls = evaluate_svm(X_cls, y, kernel="rbf", C=10.0, class_weight="balanced")
    results["cls_token"] = r_cls
    print(f"AUC: {r_cls['auc_mean']:.4f} ± {r_cls['auc_std']:.4f}")
    print(f"Acc: {r_cls['acc_mean']:.4f} | F1: {r_cls['f1_mean']:.4f}")

    # 汇总
    print("\n" + "=" * 65)
    print("  结果汇总")
    print("=" * 65)
    delta = r_mean["auc_mean"] - r_cls["auc_mean"]
    print(f"Mean Pooling AUC: {r_mean['auc_mean']:.4f}")
    print(f"CLS Token    AUC: {r_cls['auc_mean']:.4f}")
    print(f"Δ (Mean - CLS):   {delta:+.4f}")

    if abs(delta) < 0.01:
        print("结论: 两种聚合方式在 SVM 上无显著差异。")
    elif delta > 0:
        print("结论: Mean pooling 略优于 CLS token (在 SVM 上)。")
    else:
        print("结论: CLS token 略优于 Mean pooling (在 SVM 上)。")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
