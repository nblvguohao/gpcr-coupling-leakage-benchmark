#!/usr/bin/env python3
"""
统计显著性检验
比较不同方法的性能差异是否具有统计显著性
"""
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path('/data/lgh/GPCR/output/real_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/output/real_data/results')

def load_data():
    """加载特征和标签"""
    with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
        features = json.load(f)
    with open('/data/lgh/GPCR/output/real_data/real_labels.json', 'r') as f:
        labels = json.load(f)

    ids = list(features.keys())
    X = np.array([features[k] for k in ids])
    y = np.array([labels[k] for k in ids])

    return X, y, ids

def cross_validation_scores(X, y, model, n_splits=5):
    """执行交叉验证并返回每折的AUC分数"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    acc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        acc_scores.append(accuracy_score(y_test, y_pred))

    return np.array(auc_scores), np.array(acc_scores)

def paired_t_test(scores1, scores2, method1_name, method2_name):
    """执行配对t检验"""
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # 效应量 (Cohen's d)
    diff = scores1 - scores2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    return {
        'method1': method1_name,
        'method2': method2_name,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05),
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff, ddof=1))
    }

def wilcoxon_test(scores1, scores2, method1_name, method2_name):
    """执行Wilcoxon符号秩检验"""
    w_stat, p_value = stats.wilcoxon(scores1, scores2)

    return {
        'method1': method1_name,
        'method2': method2_name,
        'w_statistic': float(w_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni多重检验校正"""
    n = len(p_values)
    corrected_alpha = alpha / n
    significant = [p < corrected_alpha for p in p_values]

    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_comparisons': n,
        'significant': [bool(s) for s in significant]
    }

def main():
    print("="*70)
    print("统计显著性检验")
    print("="*70)

    X, y, ids = load_data()
    print(f"\n数据集: {len(y)} 个样本")
    print(f"  正样本: {sum(y)}, 负样本: {len(y) - sum(y)}")

    # 定义模型
    models = {
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # 收集每折的AUC分数
    print("\n执行5折交叉验证收集分数...")
    method_scores = {}

    for name, model in models.items():
        auc_scores, acc_scores = cross_validation_scores(X, y, model)
        method_scores[name] = {
            'auc': auc_scores,
            'accuracy': acc_scores
        }
        print(f"  {name}: AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}")

    # 执行配对检验
    print("\n" + "="*70)
    print("配对t检验结果 (AUC)")
    print("="*70)

    method_names = list(models.keys())
    t_test_results = []
    wilcoxon_results = []
    all_p_values = []

    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            m1, m2 = method_names[i], method_names[j]
            scores1 = method_scores[m1]['auc']
            scores2 = method_scores[m2]['auc']

            # t检验
            t_result = paired_t_test(scores1, scores2, m1, m2)
            t_test_results.append(t_result)
            all_p_values.append(t_result['p_value'])

            print(f"\n{m1} vs {m2}:")
            print(f"  t统计量: {t_result['t_statistic']:.4f}")
            print(f"  p值: {t_result['p_value']:.4f}")
            print(f"  Cohen's d: {t_result['cohens_d']:.4f}")
            print(f"  显著性: {'***' if t_result['p_value'] < 0.001 else '**' if t_result['p_value'] < 0.01 else '*' if t_result['p_value'] < 0.05 else 'ns'}")

            # Wilcoxon检验
            w_result = wilcoxon_test(scores1, scores2, m1, m2)
            wilcoxon_results.append(w_result)

    # Bonferroni校正
    print("\n" + "="*70)
    print("多重检验校正 (Bonferroni)")
    print("="*70)
    bonferroni = bonferroni_correction(all_p_values)
    print(f"比较次数: {bonferroni['n_comparisons']}")
    print(f"原始alpha: {bonferroni['original_alpha']}")
    print(f"校正后alpha: {bonferroni['corrected_alpha']:.4f}")

    # 保存结果
    results = {
        'method_scores': {
            name: {
                'auc_mean': float(scores['auc'].mean()),
                'auc_std': float(scores['auc'].std()),
                'auc_scores': scores['auc'].tolist(),
                'acc_mean': float(scores['accuracy'].mean()),
                'acc_std': float(scores['accuracy'].std())
            }
            for name, scores in method_scores.items()
        },
        'paired_t_test': t_test_results,
        'wilcoxon_test': wilcoxon_results,
        'bonferroni_correction': bonferroni,
        'sample_size': len(y)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'statistical_significance.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {OUTPUT_DIR / 'statistical_significance.json'}")
    print("="*70)

if __name__ == "__main__":
    main()
