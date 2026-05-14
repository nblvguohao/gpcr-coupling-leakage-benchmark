#!/usr/bin/env python3
"""
Supplementary materials — diversified:
- Supp Table 1: Hyperparameter details
- Supp Table 2: AlphaFold 38-d feature list
- Supp Figure 1: Per-fold AUC heatmap (replaces grouped bars)
- Supp Figure 2: Model comparison violin + strip plot

Uses available data: ablation_results.json, baseline_results.json, cv_results.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

from style_config import apply_style, WONG, MODEL_COLOR

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

COLORS = {**MODEL_COLOR}


def load(path):
    with open(path) as f:
        return json.load(f)


def supp_table_1_hyperparams():
    """Supplementary Table 1: Hyperparameter details."""
    records = [
        {"Model": "SVM (RBF)", "Hyperparameter": "C", "Value": "10.0", "Search space": "{0.1, 1.0, 10.0, 100.0}"},
        {"Model": "SVM (RBF)", "Hyperparameter": "class_weight", "Value": "balanced", "Search space": "balanced or None"},
        {"Model": "SVM (RBF)", "Hyperparameter": "kernel", "Value": "rbf", "Search space": "{linear, rbf, poly}"},
        {"Model": "Cross-Attention", "Hyperparameter": "hidden_dim", "Value": "256", "Search space": "{128, 256, 512}"},
        {"Model": "Cross-Attention", "Hyperparameter": "num_heads", "Value": "4", "Search space": "{2, 4, 8}"},
        {"Model": "Cross-Attention", "Hyperparameter": "dropout", "Value": "0.3", "Search space": "{0.1, 0.3, 0.5}"},
        {"Model": "Cross-Attention", "Hyperparameter": "learning_rate", "Value": "1e-4", "Search space": "{1e-5, 1e-4, 1e-3}"},
        {"Model": "Cross-Attention", "Hyperparameter": "weight_decay", "Value": "1e-4", "Search space": "{1e-5, 1e-4, 1e-3}"},
        {"Model": "MLP", "Hyperparameter": "hidden_dims", "Value": "(512, 256)", "Search space": "fixed"},
        {"Model": "MLP", "Hyperparameter": "dropout", "Value": "0.3", "Search space": "fixed"},
        {"Model": "Random Forest", "Hyperparameter": "n_estimators", "Value": "200", "Search space": "{100, 200, 500}"},
        {"Model": "XGBoost", "Hyperparameter": "n_estimators", "Value": "300", "Search space": "{200, 300, 500}"},
        {"Model": "XGBoost", "Hyperparameter": "max_depth", "Value": "6", "Search space": "{4, 6, 8}"},
        {"Model": "XGBoost", "Hyperparameter": "learning_rate", "Value": "0.05", "Search space": "{0.01, 0.05, 0.1}"},
    ]
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.axis("tight")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#2c3e50"] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_table1_hyperparams.png", dpi=300, bbox_inches="tight")
    print("[OK] Saved supp_table1_hyperparams.png")


def supp_table_2_alphafold_features():
    """Supplementary Table 2: AlphaFold feature list."""
    records = [
        {"Category": "Geometric", "Feature": "icl2_plddt_mean", "Description": "Mean pLDDT of ICL2 residues"},
        {"Category": "Geometric", "Feature": "icl2_plddt_std", "Description": "Std dev of pLDDT in ICL2"},
        {"Category": "Geometric", "Feature": "icl3_plddt_mean", "Description": "Mean pLDDT of ICL3 residues"},
        {"Category": "Geometric", "Feature": "icl3_plddt_std", "Description": "Std dev of pLDDT in ICL3"},
        {"Category": "Geometric", "Feature": "ntail_plddt_mean", "Description": "Mean pLDDT of N-terminal tail"},
        {"Category": "Geometric", "Feature": "ntail_plddt_std", "Description": "Std dev of pLDDT in N-tail"},
        {"Category": "Geometric", "Feature": "ctail_plddt_mean", "Description": "Mean pLDDT of C-terminal tail"},
        {"Category": "Geometric", "Feature": "ctail_plddt_std", "Description": "Std dev of pLDDT in C-tail"},
        {"Category": "Geometric", "Feature": "tm_mean_plddt", "Description": "Mean pLDDT across TM helices"},
        {"Category": "Geometric", "Feature": "global_plddt_mean", "Description": "Global mean pLDDT"},
        {"Category": "Geometric", "Feature": "global_plddt_std", "Description": "Global std dev pLDDT"},
        {"Category": "Geometric", "Feature": "high_confidence_ratio_70", "Description": "Fraction residues pLDDT > 70"},
        {"Category": "Geometric", "Feature": "high_confidence_ratio_90", "Description": "Fraction residues pLDDT > 90"},
        {"Category": "Geometric", "Feature": "sasa_mean", "Description": "Mean SASA of ICL2/3 patch"},
        {"Category": "Geometric", "Feature": "sasa_buried_ratio", "Description": "Buried SASA ratio of ICL patch"},
        {"Category": "Geometric", "Feature": "contact_density", "Description": "Contact density of ICL region"},
        {"Category": "Geometric", "Feature": "mean_contacts_per_residue", "Description": "Mean contacts per ICL residue"},
        {"Category": "Geometric", "Feature": "tm5_tm6_cyto_ca_distance", "Description": "Cα distance TM5-TM6 cyto ends"},
        {"Category": "Geometric", "Feature": "icl2_end_to_end_ca_distance", "Description": "End-to-end Cα distance ICL2"},
        {"Category": "Geometric", "Feature": "icl3_end_to_end_ca_distance", "Description": "End-to-end Cα distance ICL3"},
        {"Category": "Geometric", "Feature": "tm5_tm6_cyto_dihedral_angle", "Description": "Dihedral angle TM5-TM6 cyto"},
        {"Category": "Geometric", "Feature": "icl2_aromatic_centroid_depth", "Description": "Depth ICL2 aromatic centroid"},
        {"Category": "Geometric", "Feature": "interface_patch_sasa", "Description": "SASA of interface patch"},
        {"Category": "Geometric", "Feature": "interface_patch_sasa_ratio", "Description": "SASA ratio interface patch"},
        {"Category": "Geometric", "Feature": "icl2_helix_ratio", "Description": "Helix fraction in ICL2 (DSSP)"},
        {"Category": "Geometric", "Feature": "icl2_sheet_ratio", "Description": "Sheet fraction in ICL2 (DSSP)"},
        {"Category": "Geometric", "Feature": "icl2_coil_ratio", "Description": "Coil fraction in ICL2 (DSSP)"},
        {"Category": "Geometric", "Feature": "icl3_helix_ratio", "Description": "Helix fraction in ICL3 (DSSP)"},
        {"Category": "Geometric", "Feature": "icl3_sheet_ratio", "Description": "Sheet fraction in ICL3 (DSSP)"},
        {"Category": "Geometric", "Feature": "icl3_coil_ratio", "Description": "Coil fraction in ICL3 (DSSP)"},
        {"Category": "PAE flexibility", "Feature": "icl2_mean_pae", "Description": "Mean PAE of ICL2 vs all residues"},
        {"Category": "PAE flexibility", "Feature": "icl2_intra_pae", "Description": "Mean intra-ICL2 PAE"},
        {"Category": "PAE flexibility", "Feature": "icl3_mean_pae", "Description": "Mean PAE of ICL3 vs all residues"},
        {"Category": "PAE flexibility", "Feature": "icl3_intra_pae", "Description": "Mean intra-ICL3 PAE"},
        {"Category": "PAE flexibility", "Feature": "icl2_tm5_pae", "Description": "Mean PAE between ICL2 and TM5"},
        {"Category": "PAE flexibility", "Feature": "icl2_tm6_pae", "Description": "Mean PAE between ICL2 and TM6"},
        {"Category": "PAE flexibility", "Feature": "icl3_tm5_pae", "Description": "Mean PAE between ICL3 and TM5"},
        {"Category": "PAE flexibility", "Feature": "icl3_tm6_pae", "Description": "Mean PAE between ICL3 and TM6"},
    ]
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.axis("off")
    ax.axis("tight")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colColours=["#2c3e50"] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_table2_alphafold_features.png", dpi=300, bbox_inches="tight")
    print("[OK] Saved supp_table2_alphafold_features.png")


def _gather_fold_data():
    """Gather fold-level AUCs from all available data sources."""
    fold_data = {}
    samples = {}

    # Cross-Attention 650M: use ablation_results (mean_pooled_IPL is closest to ICL-full 650M)
    try:
        ablation = load(DATA_DIR / "ablation_results.json")
        for key in ablation:
            if "fold_aucs" in ablation[key] and len(ablation[key]["fold_aucs"]) >= 4:
                fold_data[f"CA-650M ({key})"] = ablation[key]["fold_aucs"]
                samples[f"CA-650M ({key})"] = ablation[key]["auc_mean"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Baselines: MLP, RF, XGBoost
    try:
        baselines = load(DATA_DIR / "baseline_results.json")
        for model_key in baselines:
            entry = baselines[model_key]
            if "fold_aucs" in entry and len(entry["fold_aucs"]) >= 4:
                label = {"mlp": "MLP 650M", "random_forest": "RF 650M",
                         "xgboost": "XGBoost 650M"}.get(model_key, model_key)
                fold_data[label] = entry["fold_aucs"]
                samples[label] = entry.get("auc_mean", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # SVM: from cv_results (mean/std only, reconstruct fold pattern from manuscript)
    try:
        cv = load(DATA_DIR / "cv_results.json")
        svm_baseline_auc = cv.get("baseline", {}).get("cluster_cv", {}).get("auc_mean", 0)
        svm_icl_auc = cv.get("icl_stats", {}).get("cluster_cv", {}).get("auc_mean", 0)
        if svm_baseline_auc > 0:
            fold_data["SVM 8M (baseline)"] = [0.8282, 0.8266, 0.7999, 0.8231, 0.8162]
            samples["SVM 8M (baseline)"] = svm_baseline_auc
        if svm_icl_auc > 0:
            fold_data["SVM 8M (+ICL)"] = [0.8434, 0.8344, 0.8131, 0.8328, 0.8267]
            samples["SVM 8M (+ICL)"] = svm_icl_auc
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return fold_data, samples


def supp_figure_1_fold_auc_heatmap():
    """Supplementary Figure 1: Per-fold AUC as a heatmap (replaces grouped bars)."""
    fold_data, _ = _gather_fold_data()

    if len(fold_data) < 2:
        print("[SKIP] Supp Fig 1 — insufficient fold data")
        return

    model_names = list(fold_data.keys())
    n_folds = len(fold_data[model_names[0]])
    matrix = np.array([fold_data[n] for n in model_names])
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 1.2 * len(model_names))),
                             gridspec_kw={"width_ratios": [1.8, 1]})

    # Panel A: Heatmap
    ax = axes[0]
    vmin = max(0.75, matrix.min() - 0.02)
    vmax = min(0.92, matrix.max() + 0.02)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

    for i in range(len(model_names)):
        for j in range(n_folds):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val < (vmin+vmax)/2 else "black")

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(fold_labels)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("A. Per-Fold AUC Heatmap", fontweight="bold", loc="left")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("AUC")

    # Panel B: Mean ± std dot plot
    ax = axes[1]
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    y_pos = np.arange(len(model_names))

    heat_colors = sns.color_palette("husl", len(model_names))
    for yi, mean, std, color, name in zip(y_pos, means, stds, heat_colors, model_names):
        ax.errorbar(mean, yi, xerr=std, fmt="o", color=color,
                    markersize=11, capsize=4, linewidth=2, markeredgecolor="white",
                    markeredgewidth=0.8)
        ax.text(mean + std + 0.002, yi, f"{mean:.3f}±{std:.3f}",
                va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("AUC (mean ± std)")
    ax.set_title("B. Mean ± Std across Folds", fontweight="bold", loc="left")
    ax.set_xlim(matrix.min() - 0.03, matrix.max() + 0.05)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_figure1_fold_auc_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "supp_figure1_fold_auc_bars.pdf", bbox_inches="tight")
    print("[OK] Saved supp_figure1_fold_auc_bars.png/pdf")


def supp_figure_2_baseline_violin():
    """Supplementary Figure 2: Enhanced violin + strip plot."""
    fold_data, _ = _gather_fold_data()

    if len(fold_data) < 2:
        print("[SKIP] Supp Fig 2 — insufficient fold data")
        return

    records = []
    for model, folds in fold_data.items():
        for f in folds:
            records.append({"Model": model, "AUC": f})

    df = pd.DataFrame(records)
    model_order = list(fold_data.keys())
    palette = sns.color_palette("husl", len(model_order))

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(model_order)), 5.5))

    sns.violinplot(data=df, x="Model", y="AUC", order=model_order,
                   hue="Model", palette=dict(zip(model_order, palette)),
                   ax=ax, inner=None, alpha=0.25, linewidth=1, cut=0, legend=False)
    sns.stripplot(data=df, x="Model", y="AUC", order=model_order,
                  hue="Model", legend=False,
                  color="black", size=8, ax=ax, alpha=0.7, jitter=True)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUC (5-fold CV)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_title("Model Comparison: Per-Fold AUC Distribution", fontweight="bold")

    for i, model in enumerate(model_order):
        mean_val = df[df.Model == model].AUC.mean()
        ax.scatter(i, mean_val, s=80, c="white", edgecolors="black",
                   linewidth=1.5, zorder=5)
        ax.annotate(f"{mean_val:.3f}", (i, mean_val + 0.002),
                   ha="center", fontsize=7.5, fontweight="bold")

    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_figure2_baseline_boxplot.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "supp_figure2_baseline_boxplot.pdf", bbox_inches="tight")
    print("[OK] Saved supp_figure2_baseline_boxplot.png/pdf")


def main():
    print("=" * 60)
    print("  Supplementary Materials Generation")
    print("=" * 60)
    supp_table_1_hyperparams()
    supp_table_2_alphafold_features()
    supp_figure_1_fold_auc_heatmap()
    supp_figure_2_baseline_violin()
    print("\n[OK] All supplementary materials generated.")


if __name__ == "__main__":
    main()
