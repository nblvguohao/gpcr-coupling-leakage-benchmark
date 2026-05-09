#!/usr/bin/env python3
"""
Manuscript main figures: AUC comparison, ICL alignment, AlphaFold ablation.
Uses shared style_config.py for consistent, colorblind-safe styling.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_config import apply_style, WONG, MODEL_COLOR, FAMILY_COLOR

apply_style()

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {**MODEL_COLOR, **FAMILY_COLOR}


def load(path):
    with open(path) as f:
        return json.load(f)


def figure_auc_comparison():
    """Cluster-aware CV AUC comparison (Table 1 values + error bars)."""
    # Values from manuscript Table 1 (cluster-CV means)
    categories = ["Baseline", "+ ICL", "+ ICL + AF"]
    x = np.arange(len(categories))
    width = 0.22

    # mean ± std from manuscript Table 1
    data = {
        "SVM (8M)": {
            "vals": [0.7972, 0.8301, 0.8285],
            "err":  [0.0143, 0.0126, 0.0142],
            "color": COLORS["SVM"],
            "hatch": "//",
        },
        "SVM (650M)": {
            "vals": [0.8188, 0.8324, 0.8287],
            "err":  [0.0141, 0.0135, 0.0163],
            "color": COLORS["SVM"],
            "hatch": "",
        },
        "Cross-Attn (8M)": {
            "vals": [0.8159, 0.8247, 0.8207],
            "err":  [0.0187, 0.0163, 0.0189],
            "color": COLORS["CA"],
            "hatch": "//",
        },
        "Cross-Attn (650M)": {
            "vals": [0.8378, 0.8619, 0.8600],
            "err":  [0.0229, 0.0249, 0.0231],
            "color": COLORS["CA"],
            "hatch": "",
        },
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (label, d) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, d["vals"], width, label=label,
                      color=d["color"], alpha=0.85 if d["hatch"] == "" else 0.55,
                      hatch=d["hatch"], edgecolor="white", linewidth=0.5)
        for j, (bar, val, err) in enumerate(zip(bars, d["vals"], d["err"])):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, val,
                        yerr=err, fmt="none", ecolor="black", capsize=3, linewidth=0.8)
            ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.006,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Cluster-aware CV AUC")
    ax.set_xlabel("Feature configuration")
    ax.set_title("GPCR–G Protein Coupling Prediction: Architecture Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.74, 0.92)  # Honest range showing baseline
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotation
    ax.annotate("Best: 0.8619", xy=(1, 0.8619), xytext=(1.8, 0.89),
                arrowprops=dict(arrowstyle="->", color=COLORS["CA"], lw=1),
                fontsize=8, color=COLORS["CA"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure1_auc_comparison.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure1_auc_comparison.pdf")


def figure_icl_alignment():
    """ICL dimension alignment effect (Table 2 values)."""
    configs = ["320-d global\n+ 320-d ICL", "1280-d global\n+ 320-d ICL", "1280-d global\n+ 1280-d ICL"]
    x = np.arange(len(configs))
    width = 0.35

    # Values from manuscript Table 2
    svm_vals = [0.8301, 0.8234, 0.8304]
    ca_vals =  [0.8247, 0.8155, 0.8599]
    svm_err = [0.0126, 0.015, 0.013]
    ca_err = [0.0163, 0.018, 0.025]

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - width/2, svm_vals, width, label="SVM (RBF)",
                color=COLORS["SVM"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, ca_vals, width, label="Cross-Attention",
                color=COLORS["CA"], alpha=0.85, edgecolor="white")

    for bars, errs in [(b1, svm_err), (b2, ca_err)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=err, fmt="none", ecolor="black", capsize=3, linewidth=0.8)

    # Highlight mismatched
    ax.axvspan(0.5, 1.5, alpha=0.06, color="red", label="Dimension mismatch")
    ax.annotate("Performance\ndrop", xy=(1, 0.818), fontsize=8, color="red",
                ha="center", fontweight="bold")

    ax.set_ylabel("Cluster-aware CV AUC")
    ax.set_title("ICL Feature Dimension Alignment")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.5, 0.90)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure3_icl_alignment.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure3_icl_alignment.pdf")


def figure_alphafold_ablation():
    """AlphaFold structural features: no incremental benefit."""
    configs = ["SVM\n8M", "SVM\n650M", "CA\n8M", "CA\n650M"]
    x = np.arange(len(configs))
    width = 0.35

    # AUC with ICL-full vs ICL-full+AlphaFold
    icl_vals = [0.8301, 0.8324, 0.8247, 0.8619]
    af_vals =  [0.8285, 0.8287, 0.8207, 0.8600]
    icl_err = [0.0126, 0.0135, 0.0163, 0.0249]
    af_err = [0.0142, 0.0163, 0.0189, 0.0231]

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - width/2, icl_vals, width, label="With ICL features",
                color=COLORS["Ensemble"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, af_vals, width, label="ICL + AlphaFold (38-d)",
                color=WONG["grey"], alpha=0.7, edgecolor="white", hatch="//")

    for bars, errs in [(b1, icl_err), (b2, af_err)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=err, fmt="none", ecolor="black", capsize=3, linewidth=0.8)

    # Delta annotations
    for i in range(len(configs)):
        delta = icl_vals[i] - af_vals[i]
        y_mid = (icl_vals[i] + af_vals[i]) / 2
        ax.annotate(f"Δ = {delta:+.4f}", (x[i], y_mid), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="red")

    ax.set_ylabel("Cluster-aware CV AUC")
    ax.set_title("AlphaFold Structural Features: No Incremental Benefit")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.5, 0.92)  # Full range, honest
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure4_alphafold_ablation.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure4_alphafold_ablation.pdf")


def main():
    print("=" * 60)
    print("  Manuscript Figure Generation")
    print("=" * 60)
    figure_auc_comparison()
    figure_icl_alignment()
    figure_alphafold_ablation()
    print("  Done.")


if __name__ == "__main__":
    main()
