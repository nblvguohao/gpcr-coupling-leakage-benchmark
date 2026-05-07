#!/usr/bin/env python3
"""
消融实验
测试不同特征组合对性能的影响
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path('/data/lgh/GPCR/output/real_data/features')
DATA_DIR = Path('/data/lgh/GPCR/output/real_data')
OUTPUT_DIR = DATA_DIR / 'results'


def load_features(feature_type='esm'):
    """加载不同类型的特征"""
    with open(DATA_DIR / 'real_labels.json', 'r') as f:
        labels = json.load(f)

    if feature_type == 'esm':
        with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
            features = json.load(f)
        ids = list(features.keys())
        X = np.array([features[k] for k in ids])

    elif feature_type == 'phys':
        with open(FEATURES_DIR / 'phys_features.json', 'r') as f:
            features = json.load(f)
        ids = list(features.keys())
        X = np.array([features[k] for k in ids])

    elif feature_type == 'combined':
        with open(FEATURES_DIR / 'combined_features.json', 'r') as f:
            features = json.load(f)
        ids = list(features.keys())
        # combined_features是字典格式，需要合并所有特征
        X_list = []
        for k in ids:
            feat_dict = features[k]
            combined = []
            for key in sorted(feat_dict.keys()):
                val = feat_dict[key]
                if isinstance(val, list):
                    combined.extend(val)
                else:
                    combined.append(val)
            X_list.append(combined)
        X = np.array(X_list)

    elif feature_type == 'esm_phys':
        with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
            esm = json.load(f)
        with open(FEATURES_DIR / 'phys_features.json', 'r') as f:
            phys = json.load(f)
        ids = list(esm.keys())
        X = np.array([np.concatenate([esm[k], phys[k]]) for k in ids])

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    y = np.array([labels[k] for k in ids])
    return X, y


def cross_validate(X, y, model, n_splits=5):
    """执行交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        try:
            scores['auc'].append(roc_auc_score(y_test, y_prob))
        except:
            scores['auc'].append(0.5)

    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in scores.items()}


def main():
    print("="*70)
    print("消融实验 - 特征组合比较")
    print("="*70)

    # 定义特征组合
    feature_configs = [
        ('ESM-2 only', 'esm', 320),
        ('Physico only', 'phys', 29),
        ('ESM-2 + Physico', 'esm_phys', 349),
        ('All Combined', 'combined', 349)
    ]

    # 定义模型
    models = {
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}

    for feat_name, feat_type, feat_dim in feature_configs:
        print(f"\n{'='*70}")
        print(f"特征组合: {feat_name} (dim={feat_dim})")
        print(f"{'='*70}")

        X, y = load_features(feat_type)
        print(f"样本数: {len(y)}, 特征维度: {X.shape[1]}")

        results[feat_name] = {}

        for model_name, model in models.items():
            print(f"\n  模型: {model_name}")
            scores = cross_validate(X, y, model)
            results[feat_name][model_name] = scores

            print(f"    AUC: {scores['auc']['mean']:.4f} ± {scores['auc']['std']:.4f}")
            print(f"    Acc: {scores['accuracy']['mean']:.4f} ± {scores['accuracy']['std']:.4f}")
            print(f"    F1:  {scores['f1']['mean']:.4f} ± {scores['f1']['std']:.4f}")

    # 汇总表格
    print(f"\n{'='*70}")
    print("消融实验汇总 (AUC)")
    print(f"{'='*70}")
    print(f"{'特征组合':<20} {'SVM-Lin':<12} {'SVM-RBF':<12} {'RF':<12} {'LR':<12}")
    print("-"*70)

    for feat_name in results:
        row = f"{feat_name:<20}"
        for model_name in ['SVM (Linear)', 'SVM (RBF)', 'Random Forest', 'Logistic Regression']:
            auc = results[feat_name][model_name]['auc']['mean']
            row += f" {auc:<12.4f}"
        print(row)

    # 保存结果
    with open(OUTPUT_DIR / 'ablation_study.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存到: {OUTPUT_DIR / 'ablation_study.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
