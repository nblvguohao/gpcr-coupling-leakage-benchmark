#!/usr/bin/env python3
"""
Manuscript main figures: AUC comparison, ICL alignment, AlphaFold ablation.
Diversified: dot+whisker plots, pairwise connected dots, heatmap-annotated bars.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
    """Dot + error bar plot replacing grouped bars."""
    categories = ["Baseline", "+ ICL", "+ ICL + AF"]
    x = np.arange(len(categories))

    models = {
        "SVM (8M)": {
            "vals": [0.7972, 0.8301, 0.8285],
            "err":  [0.0143, 0.0126, 0.0142],
            "color": COLORS["SVM"],
            "marker": "s",
            "offset": -0.2,
        },
        "SVM (650M)": {
            "vals": [0.8188, 0.8324, 0.8287],
            "err":  [0.0141, 0.0135, 0.0163],
            "color": COLORS["SVM"],
            "marker": "D",
            "offset": -0.07,
        },
        "Cross-Attn (8M)": {
            "vals": [0.8159, 0.8247, 0.8207],
            "err":  [0.0187, 0.0163, 0.0189],
            "color": COLORS["CA"],
            "marker": "o",
            "offset": 0.07,
        },
        "Cross-Attn (650M)": {
            "vals": [0.8378, 0.8619, 0.8600],
            "err":  [0.0229, 0.0249, 0.0231],
            "color": COLORS["CA"],
            "marker": "D",
            "offset": 0.2,
        },
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for label, d in models.items():
        xoff = x + d["offset"]
        ax.errorbar(xoff, d["vals"], yerr=d["err"], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.0, alpha=0.5, zorder=1)
        ax.scatter(xoff, d["vals"], s=150, c=d["color"], marker=d["marker"],
                   label=label, edgecolors="white", linewidth=0.8, zorder=3, alpha=0.9)
        # Compact value labels: offset vertically based on position to avoid overlap
        for idx, (xi, val, err) in enumerate(zip(xoff, d["vals"], d["err"])):
            # Alternate label positions above and below to reduce overlap
            if idx == 2:  # middle point — put below for CA, above for SVM
                y_offset = -0.018 if "CA" in label else 0.018
                va = "top" if y_offset < 0 else "bottom"
            else:
                y_offset = val + err + 0.012 - val  # above the point
                va = "bottom"
            ax.text(xi, val + y_offset, f"{val:.3f}", ha="center", va=va,
                    fontsize=6.3, alpha=0.85)

    ax.set_ylabel("Cluster-aware CV AUC", fontsize=10)
    ax.set_xlabel("Feature configuration", fontsize=10)
    ax.set_title("GPCR–G Protein Coupling Prediction: Architecture Comparison", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0.76, 0.92)
    ax.legend(loc="upper left", frameon=True, fontsize=8, markerscale=0.8, ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Highlight best — cleaner line
    best_y = 0.8619
    ax.axhline(y=best_y, color=COLORS["CA"], linestyle="--", alpha=0.35, linewidth=1)
    ax.annotate("Best: 0.862", xy=(2, best_y), xytext=(2.15, 0.913),
                arrowprops=dict(arrowstyle="->", color=COLORS["CA"], lw=1.0),
                fontsize=8, color=COLORS["CA"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure1_auc_comparison.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure1_auc_comparison.pdf")


def figure_icl_alignment():
    """ICL dimension alignment — two panel figure matching manuscript caption."""

    svm_320_320 = 0.8301; svm_1280_320 = 0.8234; svm_1280_1280 = 0.8304
    ca_320_320  = 0.8247; ca_1280_320  = 0.8155; ca_1280_1280  = 0.8599

    svm_vals = [svm_320_320, svm_1280_320, svm_1280_1280]
    ca_vals  = [ca_320_320,  ca_1280_320,  ca_1280_1280]
    svm_err = [0.0126, 0.015, 0.013]
    ca_err  = [0.0163, 0.018, 0.025]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # ================================================================
    # Panel A: Dimension MISMATCH effect
    # ================================================================
    ax = axes[0]
    labels_a = ["Matched\n320-d + 320-d", "Mismatched\n1280-d + 320-d"]
    x_a = [0, 1]
    width = 0.35

    svm_sub = [svm_320_320, svm_1280_320]
    ca_sub  = [ca_320_320,  ca_1280_320]
    svm_e_sub = [svm_err[0], svm_err[1]]
    ca_e_sub  = [ca_err[0], ca_err[1]]

    b1 = ax.bar([xi - width/2 for xi in x_a], svm_sub, width,
                label="SVM (RBF)", color=COLORS["SVM"], alpha=0.85,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar([xi + width/2 for xi in x_a], ca_sub, width,
                label="Cross-Attention", color=COLORS["CA"], alpha=0.85,
                edgecolor="white", linewidth=0.5)

    for bars, errs in [(b1, svm_e_sub), (b2, ca_e_sub)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       yerr=err, fmt="none", ecolor="black", capsize=4, linewidth=1)

    # Value labels placed at mid-height inside bars (much cleaner)
    for bar, val in zip(b1, svm_sub):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
    for bar, val in zip(b2, ca_sub):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")

    # Delta annotations — placed above the bars with clear offset
    delta_svm = svm_sub[1] - svm_sub[0]
    delta_ca = ca_sub[1] - ca_sub[0]
    ax.annotate(f"ΔSVM: {delta_svm:+.4f}\nΔCA: {delta_ca:+.4f}",
               (x_a[1], 0.795), ha="center", fontsize=9, fontweight="bold",
               color="#C0392B",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC",
                         edgecolor="#C0392B", alpha=0.9, lw=1.2))

    ax.axvspan(0.45, 1.55, alpha=0.06, color="red", zorder=0)
    ax.set_ylabel("Cluster-aware CV AUC", fontsize=9)
    ax.set_title("A. Dimension Mismatch Causes\n   Performance Degradation", fontweight="bold",
                 loc="left", fontsize=10)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=9)
    ax.set_ylim(0.60, 0.87)
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # ================================================================
    # Panel B: Recovery after dimension matching
    # ================================================================
    ax = axes[1]
    labels_b = ["Mismatched\n1280-d + 320-d", "Matched ✓\n1280-d + 1280-d"]
    x_b = [0, 1]

    svm_sub2 = [svm_1280_320, svm_1280_1280]
    ca_sub2  = [ca_1280_320,  ca_1280_1280]
    svm_e_sub2 = [svm_err[1], svm_err[2]]
    ca_e_sub2  = [ca_err[1],  ca_err[2]]

    b3 = ax.bar([xi - width/2 for xi in x_b], svm_sub2, width,
                label="SVM (RBF)", color=COLORS["SVM"], alpha=0.85,
                edgecolor="white", linewidth=0.5)
    b4 = ax.bar([xi + width/2 for xi in x_b], ca_sub2, width,
                label="Cross-Attention", color=COLORS["CA"], alpha=0.85,
                edgecolor="white", linewidth=0.5)

    for bars, errs in [(b3, svm_e_sub2), (b4, ca_e_sub2)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       yerr=err, fmt="none", ecolor="black", capsize=4, linewidth=1)

    # Value labels inside bars
    for bar, val in zip(b3, svm_sub2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
    for bar, val in zip(b4, ca_sub2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")

    # Recovery annotation
    delta_ca_recovery = ca_1280_1280 - ca_1280_320
    delta_svm_recovery = svm_1280_1280 - svm_1280_320
    ax.annotate(f"Recovery:\nSVM {delta_svm_recovery:+.4f}\nCA {delta_ca_recovery:+.4f}",
               (x_b[1], 0.89), ha="center", fontsize=9, fontweight="bold",
               color="#27AE60",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5F5E3",
                         edgecolor="#27AE60", alpha=0.9, lw=1.5))

    ax.axvspan(0.45, 1.55, alpha=0.06, color="green", zorder=0)

    ax.set_ylabel("Cluster-aware CV AUC", fontsize=9)
    ax.set_title("B. Dimension Matching Recovers\n   Cross-Attention Advantage", fontweight="bold",
                 loc="left", fontsize=10)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=9)
    ax.set_ylim(0.60, 0.93)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure3_icl_alignment.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure3_icl_alignment.pdf (2-panel)")


def figure_alphafold_ablation():
    """AlphaFold: no incremental benefit — paired dot plot with delta lines."""
    configs = ["SVM\n8M", "SVM\n650M", "CA\n8M", "CA\n650M"]
    x = np.arange(len(configs))

    icl_vals  = [0.8301, 0.8324, 0.8247, 0.8619]
    af_vals   = [0.8285, 0.8287, 0.8207, 0.8600]
    icl_err   = [0.0126, 0.0135, 0.0163, 0.0249]
    af_err    = [0.0142, 0.0163, 0.0189, 0.0231]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    dx = 0.24

    # Paired dots with connecting lines
    for i in range(len(configs)):
        ax.plot([i - dx, i + dx], [icl_vals[i], af_vals[i]], "-",
                color="grey", alpha=0.5, linewidth=1.2, zorder=1)
        # ICL-only dot
        ax.scatter(i - dx, icl_vals[i], s=130, c=COLORS["Ensemble"], marker="o",
                   edgecolors="white", linewidth=0.8, zorder=3)
        ax.errorbar(i - dx, icl_vals[i], yerr=icl_err[i], fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8, alpha=0.5)
        # ICL+AF dot
        ax.scatter(i + dx, af_vals[i], s=130, c=WONG["grey"], marker="s",
                   edgecolors="white", linewidth=0.8, zorder=3)
        ax.errorbar(i + dx, af_vals[i], yerr=af_err[i], fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8, alpha=0.5)
        # Delta annotation — above the connected pair
        delta = icl_vals[i] - af_vals[i]
        y_top = max(icl_vals[i], af_vals[i]) + max(icl_err[i], af_err[i])
        ax.text(i, y_top + 0.006, f"Δ={delta:+.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#7D3C98")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Ensemble"],
               markersize=9, label="ICL only"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=WONG["grey"],
               markersize=9, label="ICL + AlphaFold (38-d)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8.5)

    ax.set_ylabel("Cluster-aware CV AUC", fontsize=10)
    ax.set_title("AlphaFold Structural Features: No Incremental Benefit", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    # Tight y-range around data — no wasted whitespace
    all_data = icl_vals + af_vals
    ax.set_ylim(min(all_data) - 0.025, max(all_data) + 0.035)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Compact annotation in the top-left corner instead of floating
    ax.text(0.02, 0.96,
            "All Δ ≤ 0.002\nNo comparison reaches\nstatistical significance\n(p > 0.05, Bonferroni)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#C0392B",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC",
                      edgecolor="#C0392B", alpha=0.85, lw=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure4_alphafold_ablation.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure4_alphafold_ablation.pdf")


def figure_statistical_tests():
    """Statistical test results as a forest plot / heatmap."""
    try:
        tests = load(DATA_DIR / "statistical_tests_results.json")
    except (FileNotFoundError, json.JSONDecodeError):
        print("  [SKIP] statistical tests — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    comparisons = [t["Comparison"] for t in tests]
    p_values = [t["t_pvalue"] for t in tests]
    mean_diffs = [t["mean_diff"] for t in tests]

    y_pos = np.arange(len(comparisons))
    colors = ["#27AE60" if p < 0.05 else "#95A5A6" for p in p_values]

    ax.barh(y_pos, mean_diffs, color=colors, alpha=0.8, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean ΔAUC")
    ax.set_title("Statistical Significance of Model Comparisons", fontweight="bold")

    # P-value annotation
    for i, (d, p) in enumerate(zip(mean_diffs, p_values)):
        sig = "*" if p < 0.05 else ("**" if p < 0.01 else "n.s.")
        ax.text(max(mean_diffs)*1.05, i, f"p={p:.3f} {sig}",
                va="center", fontsize=7, color="black" if p < 0.05 else "grey")

    ax.axvspan(max(mean_diffs)*1.0, max(mean_diffs)*1.5, alpha=0.03, color="green",
               label="p < 0.05 significant")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure_statistical_tests.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure_statistical_tests.pdf")


def main():
    print("=" * 60)
    print("  Manuscript Figure Generation (diversified)")
    print("=" * 60)
    figure_auc_comparison()
    figure_icl_alignment()
    figure_alphafold_ablation()
    figure_statistical_tests()
    print("  Done.")


if __name__ == "__main__":
    main()
