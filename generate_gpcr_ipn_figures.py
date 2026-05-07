#!/usr/bin/env python3
"""
Generate GPCR-IPN figures for the manuscript:
  - Figure 2: Performance comparison (all configurations)
  - Figure 3: GSCA gate analysis (gate distribution + pos/neg comparison)
  - Figure 4: Ablation table visualization
"""

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results():
    results = {}

    # GSCA results
    with open(DATA_DIR / "gsca_results.json") as f:
        gsca = json.load(f)
    for k, v in gsca.items():
        results[k] = v

    # IPLv2 results
    with open(DATA_DIR / "ipl_v2_results.json") as f:
        ipl = json.load(f)
    for k, v in ipl.items():
        results[k] = v

    # Final ablation results
    with open(DATA_DIR / "final_ablation_results.json") as f:
        ab = json.load(f)
    for k, v in ab.items():
        results[k] = v

    # Original baseline
    results["original_SVM_650M_baseline"] = {"auc_mean": 0.8188, "auc_std": 0.0167}
    results["original_SVM_650M_icl"] = {"auc_mean": 0.8324, "auc_std": 0.0185}
    results["original_CA_650M_baseline"] = {"auc_mean": 0.8378, "auc_std": 0.0168}
    results["original_CA_650M_icl"] = {"auc_mean": 0.8619, "auc_std": 0.0249}
    results["original_MLP_650M_icl"] = {"auc_mean": 0.8608, "auc_std": 0.0242}
    results["original_RF_650M_icl"] = {"auc_mean": 0.8262, "auc_std": 0.0310}
    results["original_XGB_650M_icl"] = {"auc_mean": 0.8331, "auc_std": 0.0216}

    return results


def fig_performance_comparison(results):
    """Figure 2: Bar chart comparing all model configurations."""
    configs = [
        ("SVM 650M", "original_SVM_650M_icl", "#e41a1c"),
        ("RF 650M", "original_RF_650M_icl", "#ff7f00"),
        ("XGB 650M", "original_XGB_650M_icl", "#984ea3"),
        ("MLP 650M", "original_MLP_650M_icl", "#4daf4a"),
        ("Uni-CA 650M", "uni_ca_icl_full", "#377eb8"),
        ("Bi-CA 650M", "bi_ca_avg_icl_full", "#a6cee3"),
        ("GSCA 650M", "gsca_icl_full", "#fb9a99"),
        ("GPCR-IPN", "mean_pooled_IPL", "#b2df8a"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = []
    means = []
    stds = []
    colors = []

    for label, key, color in configs:
        v = results.get(key)
        if v and v.get("auc_mean"):
            names.append(label)
            means.append(v["auc_mean"])
            stds.append(v.get("auc_std", 0))
            colors.append(color)

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Highlight GPCR-IPN
    bars[-1].set_edgecolor("red")
    bars[-1].set_linewidth(2)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("AUC (cluster-aware CV)")
    ax.set_ylim(0.75, 0.90)
    ax.axhline(y=0.8619, color="gray", linestyle="--", alpha=0.5, label="Original best (0.8619)")
    ax.legend(fontsize=9)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("GPCR-IPN Performance Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_performance.pdf")
    print(f"[OK] Figure 2 saved")
    plt.close(fig)


def fig_gate_analysis(results):
    """Figure 3: GSCA gate analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Panel A: Gate mean comparison (positive vs negative) ---
    ax = axes[0]
    gate_data = {
        "All": 0.6732,
        "Positive\n(coupling)": 0.6115,
        "Negative\n(non-coupling)": 0.6942,
    }
    colors_gate = ["#999999", "#4daf4a", "#e41a1c"]
    bars = ax.bar(range(len(gate_data)), list(gate_data.values()),
                  color=colors_gate, edgecolor="black", width=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Balanced (gate=0.5)")
    ax.set_xticks(range(len(gate_data)))
    ax.set_xticklabels(list(gate_data.keys()), fontsize=10)
    ax.set_ylabel("Gate value (mean)")
    ax.set_ylim(0, 1.0)
    ax.set_title("GSCA Gate: Positive vs Negative Pairs")
    ax.legend(fontsize=8)

    for i, (k, v) in enumerate(gate_data.items()):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # Annotation
    ax.text(0.5, 0.85, "GPCR-dominated\n(gate > 0.5)",
            transform=ax.transAxes, fontsize=9, ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.text(0.5, 0.15, "Gprot-dominated\n(gate < 0.5)",
            transform=ax.transAxes, fontsize=9, ha="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    # --- Panel B: Ablation bar chart ---
    ax = axes[1]
    abl_configs = [
        ("SVM\n650M", 0.8324, 0.0185, "#e41a1c"),
        ("Uni-CA\n650M", 0.8572, 0.0218, "#377eb8"),
        ("+ GSCA", 0.8552, 0.0246, "#fb9a99"),
        ("+ IPL", 0.8590, 0.0261, "#b2df8a"),
    ]

    names = [c[0] for c in abl_configs]
    means = [c[1] for c in abl_configs]
    errs = [c[2] for c in abl_configs]
    colors = [c[3] for c in abl_configs]

    bars = ax.bar(range(len(names)), means, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    bars[-1].set_edgecolor("red")
    bars[-1].set_linewidth(2)

    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.005, f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.78, 0.90)
    ax.set_title("Ablation: Components of GPCR-IPN")
    ax.axhline(y=0.8324, color="gray", linestyle=":", alpha=0.4)

    # Arrow showing improvement
    ax.annotate("", xy=(0, 0.8324), xytext=(3, 0.8590),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(1.5, 0.885, "+0.027", ha="center", fontsize=10, color="green", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_gate_analysis.pdf")
    print(f"[OK] Figure 3 saved")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  Generating GPCR-IPN figures")
    print("=" * 60)

    results = load_results()
    fig_performance_comparison(results)
    fig_gate_analysis(results)

    print(f"\n[OK] All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
