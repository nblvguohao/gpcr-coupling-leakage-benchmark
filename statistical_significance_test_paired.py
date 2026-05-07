#!/usr/bin/env python3
"""
Paired statistical significance tests for cluster-aware CV fold AUCs.
Compares key model configurations using paired t-test and Wilcoxon signed-rank test.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_fold_aucs(d, *keys):
    """Safely navigate nested dict to extract fold_aucs list."""
    try:
        for k in keys:
            d = d[k]
        return np.array(d["fold_aucs"])
    except (KeyError, TypeError):
        return None


def paired_tests(a, b, label_a, label_b):
    """Run paired t-test and Wilcoxon signed-rank test."""
    # Remove NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean = a[mask]
    b_clean = b[mask]
    if len(a_clean) < 2:
        return {"n": int(len(a_clean)), "t_stat": None, "t_pvalue": None,
                "w_stat": None, "w_pvalue": None, "mean_diff": None}
    t_stat, t_p = stats.ttest_rel(a_clean, b_clean)
    w_stat, w_p = stats.wilcoxon(a_clean, b_clean, alternative="two-sided")
    return {
        "n": int(len(a_clean)),
        "t_stat": round(float(t_stat), 4),
        "t_pvalue": round(float(t_p), 4),
        "w_stat": round(float(w_stat), 4),
        "w_pvalue": round(float(w_p), 4),
        "mean_diff": round(float(np.mean(a_clean - b_clean)), 4),
    }


def main():
    print("=" * 70)
    print("  Paired Statistical Significance Tests (Cluster-aware CV)")
    print("=" * 70)

    svm_8m = load_json(DATA_DIR / "paired_cv_enhanced_results.json")
    svm_650m = load_json(DATA_DIR / "paired_cv_enhanced_v2_650m_results.json")
    dl_8m = load_json(DATA_DIR / "paired_cross_attention_results.json")
    dl_650m = load_json(DATA_DIR / "paired_cross_attention_650m_results.json")
    baselines = load_json(DATA_DIR / "paired_baselines_650m_results.json") if (DATA_DIR / "paired_baselines_650m_results.json").exists() else {}

    def get_fold_aucs(primary, fallback):
        return primary if primary is not None else fallback

    # Collect fold AUCs for key configurations
    configs = {
        "SVM 8M Baseline": get_fold_aucs(safe_fold_aucs(svm_8m, "baseline", "cluster_cv", "SVM-RBF C=10 balanced"), safe_fold_aucs(svm_8m, "baseline", "cluster_cv")),
        "SVM 8M ICL-full": get_fold_aucs(safe_fold_aucs(svm_8m, "icl_full", "cluster_cv", "SVM-RBF C=10 balanced"), safe_fold_aucs(svm_8m, "icl_full", "cluster_cv")),
        "SVM 650M Baseline": get_fold_aucs(safe_fold_aucs(svm_650m, "baseline", "cluster_cv", "SVM-RBF C=10 balanced"), safe_fold_aucs(svm_650m, "baseline", "cluster_cv")),
        "SVM 650M ICL-full": get_fold_aucs(safe_fold_aucs(svm_650m, "icl_full_v2", "cluster_cv", "SVM-RBF C=10 balanced"), safe_fold_aucs(svm_650m, "icl_full_v2", "cluster_cv")),
        "SVM 650M Alpha": get_fold_aucs(safe_fold_aucs(svm_650m, "alpha", "cluster_cv", "SVM-RBF C=10 balanced"), safe_fold_aucs(svm_650m, "alpha", "cluster_cv")),
        "Cross-Attn 8M Baseline": safe_fold_aucs(dl_8m, "baseline"),
        "Cross-Attn 8M ICL-full": safe_fold_aucs(dl_8m, "icl_full"),
        "Cross-Attn 650M Baseline": safe_fold_aucs(dl_650m, "baseline"),
        "Cross-Attn 650M ICL-full": safe_fold_aucs(dl_650m, "icl_full"),
        "Cross-Attn 650M Alpha": safe_fold_aucs(dl_650m, "alpha"),
    }

    if baselines:
        configs["MLP 650M ICL-full"] = np.array(baselines.get("mlp", {}).get("fold_aucs", []))
        configs["RF 650M ICL-full"] = np.array(baselines.get("random_forest", {}).get("fold_aucs", []))
        configs["XGBoost 650M ICL-full"] = np.array(baselines.get("xgboost", {}).get("fold_aucs", []))

    # Define comparisons of interest
    comparisons = [
        ("Cross-Attn 650M ICL-full", "MLP 650M ICL-full", "Cross-Attn vs MLP (650M ICL-full)"),
        ("Cross-Attn 650M ICL-full", "Cross-Attn 650M Baseline", "ICL-full vs Baseline (Cross-Attn 650M)"),
        ("Cross-Attn 650M ICL-full", "Cross-Attn 650M Alpha", "ICL-full vs Alpha (Cross-Attn 650M)"),
        ("Cross-Attn 650M Baseline", "Cross-Attn 8M Baseline", "650M vs 8M Baseline (Cross-Attn)"),
        ("Cross-Attn 650M ICL-full", "Cross-Attn 8M ICL-full", "650M vs 8M ICL-full (Cross-Attn)"),
        ("SVM 8M ICL-full", "SVM 8M Baseline", "ICL-full vs Baseline (SVM 8M)"),
        ("Cross-Attn 8M Baseline", "SVM 8M Baseline", "Cross-Attn vs SVM (8M Baseline)"),
        ("Cross-Attn 8M ICL-full", "SVM 8M ICL-full", "Cross-Attn vs SVM (8M ICL-full)"),
        ("MLP 650M ICL-full", "RF 650M ICL-full", "MLP vs RF (650M ICL-full)"),
        ("MLP 650M ICL-full", "XGBoost 650M ICL-full", "MLP vs XGBoost (650M ICL-full)"),
    ]

    records = []
    for a_key, b_key, desc in comparisons:
        a = configs.get(a_key)
        b = configs.get(b_key)
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            continue
        res = paired_tests(a, b, a_key, b_key)
        records.append({
            "Comparison": desc,
            "Model A": a_key,
            "Model B": b_key,
            **res,
        })
        print(f"{desc}: n={res['n']}, mean_diff={res['mean_diff']:.4f}, t_p={res['t_pvalue']:.4f}, w_p={res['w_pvalue']:.4f}")

    df = pd.DataFrame(records)
    out_json = DATA_DIR / "statistical_tests_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved statistical test results to {out_json}")

    # Generate a publication-quality table image
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.5)))
    ax.axis("off")
    ax.axis("tight")

    display_df = df[["Comparison", "n", "mean_diff", "t_pvalue", "w_pvalue"]].copy()
    display_df.columns = ["Comparison", "n", "Mean ΔAUC", "t-test p", "Wilcoxon p"]

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#2c3e50"] * len(display_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(display_df.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if j in [3, 4]:
                val = display_df.iloc[i - 1, j]
                if isinstance(val, float) and val < 0.05:
                    table[(i, j)].set_facecolor("#d5f5e3")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "table_statistical_tests.png", dpi=300, bbox_inches="tight")
    print(f"[OK] Saved statistical test table to {FIG_DIR / 'table_statistical_tests.png'}")


if __name__ == "__main__":
    main()
