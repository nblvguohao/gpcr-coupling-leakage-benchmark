#!/usr/bin/env python3
"""
Generate publication-quality figures for the GPCR-G protein coupling manuscript.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_auc(d, key):
    try:
        return d[key]["cluster_cv"]["SVM-RBF C=10 balanced"]["auc_mean"]
    except (KeyError, TypeError):
        try:
            return d[key]["cluster_cv"]["auc_mean"]
        except (KeyError, TypeError):
            return None


def dl_auc(d, key):
    try:
        return d[key]["auc_mean"]
    except (KeyError, TypeError):
        return None


def figure1_auc_comparison():
    """Figure 1: Cluster-aware CV AUC comparison across all configurations."""
    svm_8m = load_json(DATA_DIR / "paired_cv_enhanced_results.json")
    svm_650m = load_json(DATA_DIR / "cv_results.json")
    dl_8m = load_json(DATA_DIR / "paired_cross_attention_results.json")
    dl_650m = load_json(DATA_DIR / "paired_cross_attention_650m_results.json")

    # Data for grouped bar chart
    categories = ["Baseline", "ICL-full", "Alpha (38-d)"]
    svm_8m_vals = [safe_auc(svm_8m, "baseline"), safe_auc(svm_8m, "icl_full"), safe_auc(svm_8m, "alpha")]
    svm_650m_vals = [safe_auc(svm_650m, "baseline"), safe_auc(svm_650m, "icl_full_v2"), safe_auc(svm_650m, "alpha")]
    dl_8m_vals = [dl_auc(dl_8m, "baseline"), dl_auc(dl_8m, "icl_full"), dl_auc(dl_8m, "alpha")]
    dl_650m_vals = [dl_auc(dl_650m, "baseline"), dl_auc(dl_650m, "icl_full"), dl_auc(dl_650m, "alpha")]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#1abc9c", "#e74c3c", "#f39c12"]
    labels = ["SVM (8M)", "SVM (650M)", "Cross-Attn (8M)", "Cross-Attn (650M)"]
    all_vals = [svm_8m_vals, svm_650m_vals, dl_8m_vals, dl_650m_vals]

    for i, (vals, color, label) in enumerate(zip(all_vals, colors, labels)):
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=9, rotation=0)

    ax.set_ylabel("Cluster-aware CV AUC", fontweight="bold")
    ax.set_xlabel("Feature configuration", fontweight="bold")
    ax.set_title("GPCR–G protein coupling prediction: architecture and scale comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.75, 0.90)
    ax.legend(loc="upper left", frameon=True)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure1_auc_comparison.pdf", bbox_inches="tight")
    print(f"[OK] Saved Figure 1 to {FIG_DIR / 'figure1_auc_comparison.png'}")


def figure2_embedding_scale_effect():
    """Figure 2: Effect of embedding scale (8M vs 650M) on SVM and cross-attention."""
    svm_8m = load_json(DATA_DIR / "paired_cv_enhanced_results.json")
    svm_650m = load_json(DATA_DIR / "cv_results.json")
    dl_8m = load_json(DATA_DIR / "paired_cross_attention_results.json")
    dl_650m = load_json(DATA_DIR / "paired_cross_attention_650m_results.json")

    configs = ["Baseline", "ICL-full", "Alpha"]
    svm_8m_vals = [safe_auc(svm_8m, "baseline"), safe_auc(svm_8m, "icl_full"), safe_auc(svm_8m, "alpha")]
    svm_650m_vals = [safe_auc(svm_650m, "baseline"), safe_auc(svm_650m, "icl_full_v2"), safe_auc(svm_650m, "alpha")]
    dl_8m_vals = [dl_auc(dl_8m, "baseline"), dl_auc(dl_8m, "icl_full"), dl_auc(dl_8m, "alpha")]
    dl_650m_vals = [dl_auc(dl_650m, "baseline"), dl_auc(dl_650m, "icl_full"), dl_auc(dl_650m, "alpha")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # SVM
    ax = axes[0]
    x = np.arange(len(configs))
    width = 0.35
    bars1 = ax.bar(x - width/2, svm_8m_vals, width, label="ESM-2 8M", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, svm_650m_vals, width, label="ESM-2 650M", color="#2980b9", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Cluster-aware CV AUC", fontweight="bold")
    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_title("SVM (RBF)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.78, 0.85)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Cross-Attention
    ax = axes[1]
    bars1 = ax.bar(x - width/2, dl_8m_vals, width, label="ESM-2 8M", color="#e74c3c", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, dl_650m_vals, width, label="ESM-2 650M", color="#c0392b", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_title("Cross-Attention DNN", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.78, 0.90)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    sns.despine(top=True, right=True)
    plt.suptitle("Effect of protein language model scale on prediction performance", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_embedding_scale.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure2_embedding_scale.pdf", bbox_inches="tight")
    print(f"[OK] Saved Figure 2 to {FIG_DIR / 'figure2_embedding_scale.png'}")


def figure3_icl_dimension_alignment():
    """Figure 3: Schematic + bar chart showing ICL dimension alignment insight."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["320-d Global\n+ 320-d ICL", "1280-d Global\n+ 320-d ICL", "1280-d Global\n+ 1280-d ICL"]
    svm_vals = [0.8301, 0.8234, 0.8304]
    dl_vals = [0.8247, None, 0.8599]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, svm_vals, width, label="SVM", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, [v if v is not None else 0 for v in dl_vals], width, label="Cross-Attn", color="#e74c3c", edgecolor="black", linewidth=0.5)

    # Mask the missing bar
    bars2[1].set_visible(False)

    ax.set_ylabel("Cluster-aware CV AUC", fontweight="bold")
    ax.set_xlabel("Feature dimension alignment", fontweight="bold")
    ax.set_title("ICL local features must match global embedding dimension", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.80, 0.87)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_visible():
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    # Add annotation for missing 1280+320 cross-attn
    ax.annotate("Not evaluated", xy=(1 + width/2, 0.825), ha="center", fontsize=9, color="gray")

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_icl_alignment.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure3_icl_alignment.pdf", bbox_inches="tight")
    print(f"[OK] Saved Figure 3 to {FIG_DIR / 'figure3_icl_alignment.png'}")


def figure4_alphafold_ablation():
    """Figure 4: AlphaFold feature ablation showing no incremental gain."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["SVM 8M", "SVM 650M", "Cross-Attn 8M", "Cross-Attn 650M"]
    icl_full = [0.8301, 0.8324, 0.8247, 0.8619]
    alpha = [0.8285, 0.8287, 0.8207, 0.8600]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, icl_full, width, label="ICL-full", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, alpha, width, label="+ AlphaFold (38-d)", color="#95a5a6", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Cluster-aware CV AUC", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_title("AlphaFold structural features do not improve performance beyond ESM-2 + ICL", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.81, 0.87)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Add delta annotations
    for i, (icl, alp) in enumerate(zip(icl_full, alpha)):
        delta = alp - icl
        ax.annotate(f"Δ={delta:+.3f}", xy=(i, min(icl, alp) - 0.003), ha="center", fontsize=9, color="#c0392b", fontweight="bold")

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_alphafold_ablation.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure4_alphafold_ablation.pdf", bbox_inches="tight")
    print(f"[OK] Saved Figure 4 to {FIG_DIR / 'figure4_alphafold_ablation.png'}")


def main():
    figure1_auc_comparison()
    figure2_embedding_scale_effect()
    figure3_icl_dimension_alignment()
    figure4_alphafold_ablation()
    print(f"\n[OK] All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
