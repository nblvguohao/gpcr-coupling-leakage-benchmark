#!/usr/bin/env python3
"""
SVM基线对比 - 100样本
使用ESM-2特征训练SVM作为深度学习模型的基线
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_results')

def load_features(feature_type='cls'):
    """加载ESM-2特征"""
    print(f"[INFO] 加载ESM-2特征 (类型: {feature_type})...")

    if feature_type == 'cls':
        feature_file = FEATURE_DIR / 'esm_features_100samples_cls.json'
    else:
        feature_file = FEATURE_DIR / 'esm_features_100samples.json'

    with open(feature_file, 'r') as f:
        features_dict = json.load(f)

    with open(DATA_DIR / 'extended_labels.json', 'r') as f:
        labels_dict = json.load(f)

    # 获取Gαq模板特征
    gq_id = [k for k, v in labels_dict.items() if v == 1][0]
    gq_feature = np.array(features_dict[gq_id])

    if feature_type != 'cls':
        gq_feature = gq_feature.mean(axis=0)

    X_list, y_list = [], []

    for uid, label in labels_dict.items():
        if uid in features_dict:
            feat = np.array(features_dict[uid])
            if feature_type != 'cls':
                feat = feat.mean(axis=0)

            # 拼接GPCR和G蛋白特征
            combined = np.concatenate([feat, gq_feature])
            X_list.append(combined)
            y_list.append(label)

    X = np.array(X_list)  # (100, 640) - 拼接后
    y = np.array(y_list)

    print(f"[OK] 加载完成: {len(y)} 个样本")
    print(f"    特征维度: {X.shape[1]}")
    print(f"    正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")

    return X, y

def train_svm_baseline(X, y, kernel='linear', C=1.0):
    """训练SVM基线模型"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    print(f"\n[INFO] SVM参数: kernel={kernel}, C={C}")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练SVM
        svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # 预测
        y_proba = svm.predict_proba(X_test_scaled)[:, 1]
        y_pred = svm.predict(X_test_scaled)

        # 计算指标
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        all_metrics.append(metrics)
        print(f"[Fold {fold+1}] AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # 汇总
    final_metrics = {
        'auc': {'mean': np.mean([m['auc'] for m in all_metrics]), 'std': np.std([m['auc'] for m in all_metrics])},
        'accuracy': {'mean': np.mean([m['accuracy'] for m in all_metrics]), 'std': np.std([m['accuracy'] for m in all_metrics])},
        'precision': {'mean': np.mean([m['precision'] for m in all_metrics]), 'std': np.std([m['precision'] for m in all_metrics])},
        'recall': {'mean': np.mean([m['recall'] for m in all_metrics]), 'std': np.std([m['recall'] for m in all_metrics])},
        'f1': {'mean': np.mean([m['f1'] for m in all_metrics]), 'std': np.std([m['f1'] for m in all_metrics])}
    }

    print("\n" + "=" * 60)
    print("5折交叉验证平均性能:")
    for metric, values in final_metrics.items():
        print(f"  {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

    return final_metrics, all_metrics

def main():
    print("=" * 70)
    print("SVM基线对比 - 100样本")
    print("=" * 70)

    # 测试不同特征类型
    results = {}

    # 1. CLS Token特征
    print("\n" + "=" * 70)
    print("测试1: CLS Token特征")
    print("=" * 70)
    X_cls, y = load_features('cls')

    for kernel in ['linear', 'rbf']:
        for C in [0.1, 1.0, 10.0]:
            print(f"\n--- Kernel: {kernel}, C: {C} ---")
            metrics, fold_results = train_svm_baseline(X_cls, y, kernel=kernel, C=C)
            results[f'cls_{kernel}_C{C}'] = {
                'params': {'feature': 'cls', 'kernel': kernel, 'C': C},
                'metrics': metrics,
                'fold_results': fold_results
            }

    # 2. Mean Pooling特征
    print("\n" + "=" * 70)
    print("测试2: Mean Pooling特征")
    print("=" * 70)
    X_mean, y = load_features('mean')

    for kernel in ['linear', 'rbf']:
        for C in [1.0]:
            print(f"\n--- Kernel: {kernel}, C: {C} ---")
            metrics, fold_results = train_svm_baseline(X_mean, y, kernel=kernel, C=C)
            results[f'mean_{kernel}_C{C}'] = {
                'params': {'feature': 'mean', 'kernel': kernel, 'C': C},
                'metrics': metrics,
                'fold_results': fold_results
            }

    # 保存结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    # 找出最佳配置
    best_config = max(results.items(), key=lambda x: x[1]['metrics']['auc']['mean'])
    print(f"\n最佳配置: {best_config[0]}")
    print(f"  AUC: {best_config[1]['metrics']['auc']['mean']:.4f} ± {best_config[1]['metrics']['auc']['std']:.4f}")

    # 保存所有结果
    with open(OUTPUT_DIR / 'svm_baseline_100samples.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR / 'svm_baseline_100samples.json'}")

    # 对比深度学习
    print("\n" + "=" * 70)
    print("方法对比")
    print("=" * 70)
    print(f"SVM最佳:     AUC = {best_config[1]['metrics']['auc']['mean']:.4f} ± {best_config[1]['metrics']['auc']['std']:.4f}")
    print(f"Cross-Attn:  AUC = 0.8489 ± 0.0580 (CLS token)")

if __name__ == "__main__":
    main()
