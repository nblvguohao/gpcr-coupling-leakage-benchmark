#!/usr/bin/env python3
"""
Generate supplementary materials:
- Supplementary Table 1: Hyperparameter details
- Supplementary Table 2: AlphaFold 38-d feature list
- Supplementary Figure 1: Per-fold AUC bar charts for key configurations
- Supplementary Figure 2: Model comparison with baselines
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


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
        {"Category": "Geometric", "Feature": "tm_mean_plddt", "Description": "Mean pLDDT across all TM helices"},
        {"Category": "Geometric", "Feature": "global_plddt_mean", "Description": "Global mean pLDDT"},
        {"Category": "Geometric", "Feature": "global_plddt_std", "Description": "Global std dev pLDDT"},
        {"Category": "Geometric", "Feature": "high_confidence_ratio_70", "Description": "Fraction of residues with pLDDT \u003e 70"},
        {"Category": "Geometric", "Feature": "high_confidence_ratio_90", "Description": "Fraction of residues with pLDDT \u003e 90"},
        {"Category": "Geometric", "Feature": "sasa_mean", "Description": "Mean SASA of ICL2/3 patch"},
        {"Category": "Geometric", "Feature": "sasa_buried_ratio", "Description": "Buried SASA ratio of ICL patch"},
        {"Category": "Geometric", "Feature": "contact_density", "Description": "Contact density of ICL region"},
        {"Category": "Geometric", "Feature": "mean_contacts_per_residue", "Description": "Mean contacts per ICL residue"},
        {"Category": "Geometric", "Feature": "tm5_tm6_cyto_ca_distance", "Description": "C\u03b1 distance between TM5 and TM6 cytoplasmic ends"},
        {"Category": "Geometric", "Feature": "icl2_end_to_end_ca_distance", "Description": "End-to-end C\u03b1 distance of ICL2"},
        {"Category": "Geometric", "Feature": "icl3_end_to_end_ca_distance", "Description": "End-to-end C\u03b1 distance of ICL3"},
        {"Category": "Geometric", "Feature": "tm5_tm6_cyto_dihedral_angle", "Description": "Dihedral angle of TM5-TM6 cytoplasmic segment"},
        {"Category": "Geometric", "Feature": "icl2_aromatic_centroid_depth", "Description": "Depth of ICL2 aromatic centroid"},
        {"Category": "Geometric", "Feature": "interface_patch_sasa", "Description": "SASA of predicted interface patch"},
        {"Category": "Geometric", "Feature": "interface_patch_sasa_ratio", "Description": "SASA ratio of interface patch"},
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
    fig, ax = plt.subplots(figsize=(14, 10))
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


def supp_figure_1_fold_bars():
    """Supplementary Figure 1: Per-fold AUC bar charts."""
    dl_650m = json.load(open(DATA_DIR / "paired_cross_attention_650m_results.json"))
    dl_8m = json.load(open(DATA_DIR / "paired_cross_attention_results.json"))
    svm_8m = json.load(open(DATA_DIR / "paired_cv_enhanced_results.json"))
    baselines = json.load(open(DATA_DIR / "baseline_results.json"))

    # Extract fold_aucs safely
    def get_dl_folds(d, key):
        return d.get(key, {}).get("fold_aucs", [])

    def get_svm_folds(d, key):
        nested = d.get(key, {}).get("cluster_cv", {})
        if "fold_aucs" in nested:
            return nested["fold_aucs"]
        for k, v in nested.items():
            if isinstance(v, dict) and "fold_aucs" in v:
                return v["fold_aucs"]
        return []

    data = {
        "Cross-Attn 650M ICL-full": get_dl_folds(dl_650m, "icl_full"),
        "Cross-Attn 8M ICL-full": get_dl_folds(dl_8m, "icl_full"),
        "SVM 8M ICL-full": get_svm_folds(svm_8m, "icl_full"),
        "MLP 650M ICL-full": baselines.get("mlp", {}).get("fold_aucs", []),
        "XGBoost 650M ICL-full": baselines.get("xgboost", {}).get("fold_aucs", []),
        "RF 650M ICL-full": baselines.get("random_forest", {}).get("fold_aucs", []),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.12
    colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71", "#1abc9c"]
    for i, (label, folds) in enumerate(data.items()):
        if len(folds) == 5:
            ax.bar(x + i * width, folds, width, label=label, color=colors[i])
    ax.set_ylabel("AUC", fontweight="bold")
    ax.set_xlabel("Fold", fontweight="bold")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"])
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.75, 0.92)
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_figure1_fold_auc_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "supp_figure1_fold_auc_bars.pdf", bbox_inches="tight")
    print("[OK] Saved supp_figure1_fold_auc_bars.png/pdf")


def supp_figure_2_baseline_comparison():
    """Supplementary Figure 2: Baseline model comparison box plot."""
    baselines = json.load(open(DATA_DIR / "baseline_results.json"))
    dl_650m = json.load(open(DATA_DIR / "paired_cross_attention_650m_results.json"))

    records = []
    for model, key, color in [
        ("Cross-Attn 650M ICL-full", "icl_full", "#e74c3c"),
        ("MLP 650M ICL-full", "mlp", "#f39c12"),
        ("XGBoost 650M ICL-full", "xgboost", "#2ecc71"),
        ("RF 650M ICL-full", "random_forest", "#3498db"),
    ]:
        if key == "icl_full":
            folds = dl_650m.get(key, {}).get("fold_aucs", [])
        else:
            folds = baselines.get(key, {}).get("fold_aucs", [])
        for f in folds:
            records.append({"Model": model, "AUC": f})

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 6))
    order = [
        "Cross-Attn 650M ICL-full",
        "MLP 650M ICL-full",
        "XGBoost 650M ICL-full",
        "RF 650M ICL-full",
    ]
    palette = {"Cross-Attn 650M ICL-full": "#e74c3c", "MLP 650M ICL-full": "#f39c12",
               "XGBoost 650M ICL-full": "#2ecc71", "RF 650M ICL-full": "#3498db"}
    sns.boxplot(data=df, x="Model", y="AUC", order=order, palette=palette, ax=ax)
    sns.stripplot(data=df, x="Model", y="AUC", order=order, color="black", size=6, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylabel("AUC", fontweight="bold")
    ax.set_xlabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_figure2_baseline_boxplot.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "supp_figure2_baseline_boxplot.pdf", bbox_inches="tight")
    print("[OK] Saved supp_figure2_baseline_boxplot.png/pdf")


def main():
    supp_table_1_hyperparams()
    supp_table_2_alphafold_features()
    supp_figure_1_fold_bars()
    supp_figure_2_baseline_comparison()
    print("\n[OK] All supplementary materials generated.")


if __name__ == "__main__":
    main()
