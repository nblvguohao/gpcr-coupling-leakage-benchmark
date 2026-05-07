#!/usr/bin/env python3
"""
方法比较实验
与基线方法进行对比：Random Forest, SVM, XGBoost, Logistic Regression
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping...")

FEATURES_DIR = Path('/data/lgh/GPCR/output/real_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/output/real_data/results')

def load_data():
    """加载ESM特征和标签"""
    with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
        esm_features = json.load(f)
    with open('/data/lgh/GPCR/output/real_data/real_labels.json', 'r') as f:
        labels = json.load(f)

    ids = list(esm_features.keys())
    X = np.array([esm_features[k] for k in ids])
    y = np.array([labels[k] for k in ids])

    return X, y, ids

def evaluate_method(clf, X_train, y_train, X_val, y_val):
    """评估单个方法"""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_val, y_prob)
    except:
        auc = 0.5

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }

def cross_validation_all_methods(X, y, n_splits=5):
    """所有方法的交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 初始化方法
    methods = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    }

    if XGBOOST_AVAILABLE:
        methods['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    # 存储结果
    results = {name: [] for name in methods.keys()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        print("-" * 60)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, clf in methods.items():
            result = evaluate_method(clf, X_train, y_train, X_val, y_val)
            results[name].append(result)
            print(f"{name:20s}: Acc={result['accuracy']:.4f}, AUC={result['auc']:.4f}")

    return results

def main():
    print("="*70)
    print("方法比较实验")
    print("="*70)

    X, y, ids = load_data()
    print(f"\n数据集统计:")
    print(f"  总样本: {len(y)}")
    print(f"  正样本: {sum(y)}")
    print(f"  负样本: {len(y) - sum(y)}")
    print(f"  特征维度: {X.shape[1]}")

    print(f"\n进行5折交叉验证...")
    results = cross_validation_all_methods(X, y, n_splits=5)

    # 汇总结果
    print(f"\n{'='*70}")
    print("结果汇总 (5折交叉验证平均值 ± 标准差)")
    print(f"{'='*70}")
    print(f"{'方法':<20s} {'准确率':<12s} {'精确率':<12s} {'召回率':<12s} {'F1':<12s} {'AUC':<12s}")
    print("-"*70)

    summary = {}
    for name, fold_results in results.items():
        accs = [r['accuracy'] for r in fold_results]
        precs = [r['precision'] for r in fold_results]
        recs = [r['recall'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]

        summary[name] = {
            'accuracy': {'mean': np.mean(accs), 'std': np.std(accs)},
            'precision': {'mean': np.mean(precs), 'std': np.std(precs)},
            'recall': {'mean': np.mean(recs), 'std': np.std(recs)},
            'f1': {'mean': np.mean(f1s), 'std': np.std(f1s)},
            'auc': {'mean': np.mean(aucs), 'std': np.std(aucs)}
        }

        print(f"{name:<20s} {np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"{np.mean(precs):.4f}±{np.std(precs):.4f} "
              f"{np.mean(recs):.4f}±{np.std(recs):.4f} "
              f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} "
              f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    print("="*70)

    # 保存结果
    with open(OUTPUT_DIR / 'method_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n结果已保存到: {OUTPUT_DIR / 'method_comparison.json'}")

    # 找出最佳方法
    best_auc = max(summary.items(), key=lambda x: x[1]['auc']['mean'])
    print(f"\n最佳方法 (按AUC): {best_auc[0]} (AUC={best_auc[1]['auc']['mean']:.4f})")

if __name__ == "__main__":
    main()
