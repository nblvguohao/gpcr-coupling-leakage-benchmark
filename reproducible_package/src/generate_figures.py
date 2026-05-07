#!/usr/bin/env python3
"""
Generate publication-quality figures from cross-validation results.

Usage:
    python src/generate_figures.py \
        --cv-results results/cv_results.json \
        --permutation results/permutation_importance.json \
        --shap-dir results/shap \
        --output-dir figures/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def figure1_cv_comparison(cv_results, output_dir):
    """Figure 1: Bar chart comparing Random CV, Cluster-aware CV, and LOGPSO."""
    if cv_results is None:
        print("[SKIP] Figure 1: CV results not found")
        return

    configs = []
    random_vals = []
    cluster_vals = []
    for key in ["baseline", "icl_stats", "icl_full"]:
        if key not in cv_results:
            continue
        label = {"baseline": "Global ESM", "icl_stats": "+ ICL stats", "icl_full": "+ ICL full"}[key]
        configs.append(label)
        random_vals.append(cv_results[key]["random_cv"]["auc_mean"])
        cluster_vals.append(cv_results[key]["cluster_cv"]["auc_mean"])

    if not configs:
        print("[SKIP] Figure 1: No usable configs")
        return

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, random_vals, width, label="Random CV", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cluster_vals, width, label="Cluster-aware CV", color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AUC-ROC", fontweight="bold")
    ax.set_xlabel("Feature configuration", fontweight="bold")
    ax.set_title("Cross-validation performance comparison", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc="lower right", frameon=True)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_cv_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure1_cv_comparison.pdf", bbox_inches="tight")
    print(f"[OK] Figure 1 saved")


def figure2_logpso(cv_results, output_dir):
    """Figure 2: LOGPSO per-family AUC."""
    if cv_results is None:
        print("[SKIP] Figure 2: CV results not found")
        return

    # Pick the best config with LOGPSO data
    logpso_data = None
    for key in ["icl_full", "icl_stats", "baseline"]:
        if key in cv_results and cv_results[key].get("logpso_cv"):
            logpso_data = cv_results[key]["logpso_cv"]
            break

    if not logpso_data:
        print("[SKIP] Figure 2: No LOGPSO data")
        return

    families = sorted(logpso_data.keys())
    aucs = [logpso_data[f]["auc"] for f in families]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71" if a >= 0.7 else "#e74c3c" for a in aucs]
    bars = ax.bar(families, aucs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("AUC-ROC", fontweight="bold")
    ax.set_xlabel("Left-out G-protein family", fontweight="bold")
    ax.set_title("LOGPSO: Leave-one-G-protein-family-out", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
    ax.legend()
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_logpso.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure2_logpso.pdf", bbox_inches="tight")
    print(f"[OK] Figure 2 saved")


def figure3_permutation_importance(perm_path, output_dir):
    """Figure 3: ICL2/3 stats permutation importance horizontal bar chart."""
    perm = load_json(perm_path)
    if perm is None:
        print("[SKIP] Figure 3: Permutation importance not found")
        return

    icl2 = perm.get("icl2_stats", {})
    icl3 = perm.get("icl3_stats", {})
    if not icl2 or not icl3:
        print("[SKIP] Figure 3: No ICL data")
        return

    # Sort by importance
    icl2_items = sorted(icl2.items(), key=lambda x: x[1])
    icl3_items = sorted(icl3.items(), key=lambda x: x[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, items, title in zip(axes, [icl2_items, icl3_items], ["ICL2", "ICL3"]):
        labels = [i[0] for i in items]
        vals = [i[1] for i in items]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, color="#3498db", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Permutation importance (AUC drop)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        sns.despine(top=True, right=True)

    plt.suptitle("ICL2/3 physicochemical feature importance", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure3_permutation_importance.pdf", bbox_inches="tight")
    print(f"[OK] Figure 3 saved")


def figure4_shap_regions(shap_dir, output_dir):
    """Figure 4: SHAP importance by feature region."""
    shap_summary = load_json(Path(shap_dir) / "shap_summary.json")
    if shap_summary is None:
        print("[SKIP] Figure 4: SHAP summary not found")
        return

    regions = shap_summary.get("regions", {})
    if not regions:
        print("[SKIP] Figure 4: No region data")
        return

    names = list(regions.keys())
    sums = [regions[n]["sum_abs_shap"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    bars = ax.bar(names, sums, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, sums):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(sums)*0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Sum |SHAP|", fontweight="bold")
    ax.set_xlabel("Feature region", fontweight="bold")
    ax.set_title("SHAP attribution by feature region", fontweight="bold", fontsize=13)
    plt.xticks(rotation=15, ha="right")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_dir / "figure4_shap_regions.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure4_shap_regions.pdf", bbox_inches="tight")
    print(f"[OK] Figure 4 saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-results", required=True)
    parser.add_argument("--permutation", required=True)
    parser.add_argument("--shap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results = load_json(args.cv_results)

    figure1_cv_comparison(cv_results, output_dir)
    figure2_logpso(cv_results, output_dir)
    figure3_permutation_importance(args.permutation, output_dir)
    figure4_shap_regions(args.shap_dir, output_dir)

    print(f"\n[OK] All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
