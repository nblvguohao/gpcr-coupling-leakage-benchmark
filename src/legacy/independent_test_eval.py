#!/usr/bin/env python3
"""
独立测试集评估
严格按照训练70% / 验证15% / 测试15%的划分
测试集仅在最后使用一次
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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

def evaluate_model(clf, X_test, y_test, model_name):
    """评估模型"""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} 测试结果:")
    print(f"  准确率:  {acc:.4f}")
    print(f"  精确率:  {prec:.4f}")
    print(f"  召回率:  {rec:.4f}")
    print(f"  F1分数:  {f1:.4f}")
    print(f"  AUC:     {auc:.4f}")
    print(f"  混淆矩阵:")
    print(f"    TN={cm[0,0]:2d} FP={cm[0,1]:2d}")
    print(f"    FN={cm[1,0]:2d} TP={cm[1,1]:2d}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

def main():
    print("="*70)
    print("独立测试集评估")
    print("="*70)
    print("\n数据划分策略:")
    print("  训练集: 70% (用于模型训练)")
    print("  验证集: 15% (用于超参数调优)")
    print("  测试集: 15% (严格独立，仅在最后使用)")
    print("="*70)

    X, y, ids = load_data()
    print(f"\n数据集统计:")
    print(f"  总样本: {len(y)}")
    print(f"  正样本: {sum(y)}")
    print(f"  负样本: {len(y) - sum(y)}")

    # 第一步：划分出测试集 (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 第二步：从剩余数据中划分训练集和验证集 (70% / 15% of total)
    # 剩余85%中，70%/85% ≈ 82.4% 作为训练集，15%/85% ≈ 17.6% 作为验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    )

    print(f"\n划分结果:")
    print(f"  训练集: {len(y_train)} 样本 ({len(y_train)/len(y)*100:.1f}%)")
    print(f"    - 正样本: {sum(y_train)}")
    print(f"    - 负样本: {len(y_train) - sum(y_train)}")
    print(f"  验证集: {len(y_val)} 样本 ({len(y_val)/len(y)*100:.1f}%)")
    print(f"    - 正样本: {sum(y_val)}")
    print(f"    - 负样本: {len(y_val) - sum(y_val)}")
    print(f"  测试集: {len(y_test)} 样本 ({len(y_test)/len(y)*100:.1f}%)")
    print(f"    - 正样本: {sum(y_test)}")
    print(f"    - 负样本: {len(y_test) - sum(y_test)}")

    # 在训练集上训练多个模型
    print(f"\n{'='*70}")
    print("在训练集上训练模型...")
    print(f"{'='*70}")

    models = {
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # 训练并在验证集上选择最佳模型
    best_val_auc = 0
    best_model_name = None
    best_model = None

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_acc = accuracy_score(y_val, clf.predict(X_val))

        print(f"{name:20s}: 验证集 Acc={val_acc:.4f}, AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_name = name
            best_model = clf

    print(f"\n最佳模型 (基于验证集): {best_model_name} (AUC={best_val_auc:.4f})")

    # 最后在独立测试集上评估所有模型
    print(f"\n{'='*70}")
    print("在独立测试集上的最终评估")
    print(f"{'='*70}")

    final_results = {}
    for name, clf in models.items():
        results = evaluate_model(clf, X_test, y_test, name)
        final_results[name] = results

    # 保存结果
    with open(OUTPUT_DIR / 'independent_test_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n结果已保存到: {OUTPUT_DIR / 'independent_test_results.json'}")
    print("="*70)

if __name__ == "__main__":
    main()
