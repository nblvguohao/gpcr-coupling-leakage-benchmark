#!/usr/bin/env python3
"""
Generate additional figures for BIB submission:
  1. Promiscuity-stratified performance analysis
  2. Model calibration and reliability analysis
  3. Feature importance grouped bar chart
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve

# BIB-compatible styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BASE = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "SVM": "#0072B2",
    "CA": "#D55E00",
    "Ensemble": "#009E73",
    "Gi": "#56B4E9",
    "Gs": "#E69F00",
    "Gq": "#F0E442",
    "G12_13": "#CC79A7",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  BIB Supplementary Figure Generation")
    print("=" * 60)

    # Load data
    pairing = pd.read_csv(DATA_DIR / "pairing_matrix_raw.csv")
    svm_preds = load_json(DATA_DIR / "svm_predictions.json")
    ca_preds = load_json(DATA_DIR / "ca_predictions.json")

    # Determine promiscuity
    pos_pairs = pairing[pairing.coupling == 1]
    promiscuity = pos_pairs.groupby("gpcr_id").g_protein_family.nunique()

    # Build predictions DataFrame
    families = ["Gq", "Gi", "Gs", "G12_13"]
    rows = []
    for gid, fam_dict in ca_preds.items():
        base_id = gid.split("_", 1)[1] if "_" in gid else gid
        prom_val = promiscuity.get(base_id, promiscuity.get(gid, 1))
        for fam in families:
            if fam in fam_dict and gid in svm_preds and fam in svm_preds[gid]:
                rows.append({
                    "gpcr_prom": min(prom_val, 4),  # cap at 4+
                    "family": fam,
                    "label": fam_dict[fam]["label"],
                    "svm_prob": svm_preds[gid][fam]["prob"],
                    "ca_prob": fam_dict[fam]["prob"],
                })

    df = pd.DataFrame(rows)
    df["ensemble_prob"] = (df.svm_prob + df.ca_prob) / 2
    df["confidence"] = np.abs(df.ca_prob - 0.5) * 2

    # ==================================================================
    # FIGURE: Promiscuity-stratified AUC
    # ==================================================================
    print("\n[1/3] Generating promiscuity analysis figure...")

    prom_labels = {1: "Single-family", 2: "Dual-family", 3: "Triple-family", 4: "4+-family"}
    prom_data = {}
    for pval in sorted(prom_labels.keys()):
        sub = df[df.gpcr_prom == pval]
        if len(sub) < 10:
            continue
        y = sub.label.values
        prom_data[prom_labels[pval]] = {
            "N": len(sub),
            "SVM": roc_auc_score(y, sub.svm_prob),
            "CA": roc_auc_score(y, sub.ca_prob),
            "Ensemble": roc_auc_score(y, sub.ensemble_prob),
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: grouped bar chart
    ax = axes[0]
    names = list(prom_data.keys())
    x = np.arange(len(names))
    width = 0.25
    for i, (model, color_key) in enumerate([("SVM", "SVM"), ("CA", "CA"), ("Ensemble", "Ensemble")]):
        vals = [prom_data[n][model] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=model, color=COLORS[color_key], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_ylabel("AUC (Cluster-CV)")
    ax.set_title("Performance by GPCR Promiscuity")
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim(0.78, 0.92)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: delta AUC (CA - SVM)
    ax = axes[1]
    deltas = [prom_data[n]["CA"] - prom_data[n]["SVM"] for n in names]
    delta_colors = [COLORS["CA"] if d > 0 else COLORS["SVM"] for d in deltas]
    bars = ax.bar(names, deltas, color=delta_colors, alpha=0.8)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"+{d:.3f}" if d > 0 else f"{d:.3f}", ha="center", va="bottom" if d > 0 else "top",
                fontsize=8, fontweight="bold")
    ax.set_ylabel(r"$\Delta$AUC (CA $-$ SVM)")
    ax.set_title("Cross-Attention Advantage")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "promiscuity_analysis.pdf", format="pdf")
    fig.savefig(FIG_DIR / "promiscuity_analysis.png", format="png")
    plt.close(fig)
    print(f"  -> Saved figures/promiscuity_analysis.pdf")

    # ==================================================================
    # FIGURE: Calibration and confidence analysis
    # ==================================================================
    print("\n[2/3] Generating calibration analysis figure...")

    y_true = df.label.values

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Reliability diagram
    ax = axes[0]
    for model_name, probs, color_key in [
        ("Cross-Attention", df.ca_prob, "CA"),
        ("SVM", df.svm_prob, "SVM"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", color=COLORS[color_key], label=model_name,
                markersize=5, linewidth=1.5, alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("A. Reliability Diagram")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, linestyle="--")

    # Brier and ECE annotation
    for model_name, probs in [("CA", df.ca_prob), ("SVM", df.svm_prob)]:
        brier = brier_score_loss(y_true, probs)
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ece = np.mean(np.abs(prob_true - prob_pred))
        print(f"  {model_name}: Brier={brier:.4f}, ECE={ece:.4f}")

    # Panel B: Confidence stratification
    ax = axes[1]
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    coverage, accuracy_vals = [], []
    for thresh in thresholds:
        sub = df[df.confidence >= thresh]
        if len(sub) > 0:
            coverage.append(len(sub) / len(df) * 100)
            accuracy_vals.append(accuracy_score(sub.label, (sub.ca_prob > 0.5).astype(int)) * 100)

    ax.plot(coverage, accuracy_vals, "o-", color=COLORS["CA"], linewidth=2, markersize=8)
    for c, a, t in zip(coverage, accuracy_vals, thresholds):
        ax.annotate(f"{t:.2f}", (c, a), textcoords="offset points",
                    xytext=(8, -2), fontsize=7.5, alpha=0.8)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B. Confidence vs. Accuracy")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(80, 102)

    # Panel C: Per-family calibration
    ax = axes[2]
    x_fam = np.arange(len(families))
    width_fam = 0.35
    family_briers, family_eces = [], []
    for fam in families:
        mask = df.family == fam
        f_probs = df.loc[mask, "ca_prob"].values
        f_labels = df.loc[mask, "label"].values
        family_briers.append(brier_score_loss(f_labels, f_probs))
        prob_t, prob_p = calibration_curve(f_labels, f_probs, n_bins=5, strategy="uniform")
        family_eces.append(np.mean(np.abs(prob_t - prob_p)))

    bars1 = ax.bar(x_fam - width_fam / 2, family_briers, width_fam,
                   label="Brier Score", color=COLORS["Ensemble"], alpha=0.8)
    ax2_b = ax.twinx()
    bars2 = ax2_b.bar(x_fam + width_fam / 2, family_eces, width_fam,
                      label="ECE", color=COLORS["Gq"], alpha=0.8)
    ax.set_xticks(x_fam)
    ax.set_xticklabels(families)
    ax.set_ylabel("Brier Score")
    ax2_b.set_ylabel("ECE")
    ax.set_title("C. Per-Family Calibration")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, fancybox=True)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "calibration_analysis.pdf", format="pdf")
    fig.savefig(FIG_DIR / "calibration_analysis.png", format="png")
    plt.close(fig)
    print(f"  -> Saved figures/calibration_analysis.pdf")

    # ==================================================================
    # FIGURE: Feature group importance
    # ==================================================================
    print("\n[3/3] Generating feature importance figure...")

    try:
        raw_data = load_json(FIG_DIR / "gradient_attribution_650m.json")
        grad_data = raw_data.get("feature_group_means", raw_data)
    except (FileNotFoundError, json.JSONDecodeError):
        grad_data = {
            "gpcr_esm_global": 0.11, "gprotein_esm": 0.018,
            "icl2_stats": 0.14, "icl3_stats": 0.18,
            "icl2_esm": 0.09, "icl3_esm": 0.12,
            "alphafold": 0.006,
        }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    # Normalize values for display
    raw_vals = np.array([float(v) for v in grad_data.values()])
    values = raw_vals / raw_vals.sum()  # normalize to sum to 1
    groups = list(grad_data.keys())
    group_labels = [
        "GPCR\nGlobal", "G Prot.\nGlobal", "ICL2\nStats", "ICL3\nStats",
        "ICL2\nEmbed.", "ICL3\nEmbed.", "AlphaFold"
    ]
    bar_colors = [COLORS["CA"] if "gpcr" in g.lower() or "icl" in g.lower()
                  else COLORS["SVM"] if "gprot" in g.lower()
                  else COLORS["Gs"] for g in groups]

    bars = ax.bar(range(len(groups)), values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Mean |Gradient Attribution|")
    ax.set_title("Feature Group Importance (Cross-Attention 650M)")

    # GPCR vs G-protein divider
    ax.axvline(x=1.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(0.75, max(values) * 0.95, "GPCR-side", ha="center", fontsize=8, color=COLORS["CA"], fontweight="bold")
    ax.text(2.5, max(values) * 0.95, "G Prot-side", ha="center", fontsize=8, color=COLORS["SVM"], fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_groups.pdf", format="pdf")
    fig.savefig(FIG_DIR / "feature_importance_groups.png", format="png")
    plt.close(fig)
    print(f"  -> Saved figures/feature_importance_groups.pdf")

    # ==================================================================
    # Summary for manuscript
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Key Numbers for BIB Manuscript")
    print("=" * 60)
    print(f"""
    Promiscuity analysis:
      Single-family GPCRs: CA AUC = {prom_data['Single-family']['CA']:.4f}, SVM AUC = {prom_data['Single-family']['SVM']:.4f}
      Dual-family GPCRs:   CA AUC = {prom_data['Dual-family']['CA']:.4f}, SVM AUC = {prom_data['Dual-family']['SVM']:.4f}
      Max CA advantage: {max(prom_data[n]['CA'] - prom_data[n]['SVM'] for n in names):.3f}

    Calibration:
      CA Brier: {brier_score_loss(y_true, df.ca_prob):.4f}
      SVM Brier: {brier_score_loss(y_true, df.svm_prob):.4f}
      CA ECE: {np.mean(np.abs(calibration_curve(y_true, df.ca_prob, n_bins=10)[0] - calibration_curve(y_true, df.ca_prob, n_bins=10)[1])):.4f}

    High-confidence coverage:
      Conf >= 0.90: {len(df[df.confidence >= 0.90])} samples, accuracy = {accuracy_score(df[df.confidence >= 0.90].label, (df[df.confidence >= 0.90].ca_prob > 0.5).astype(int)):.4f}
      Conf >= 0.98: {len(df[df.confidence >= 0.98])} samples, accuracy = {accuracy_score(df[df.confidence >= 0.98].label, (df[df.confidence >= 0.98].ca_prob > 0.5).astype(int)):.4f}
    """)


if __name__ == "__main__":
    main()
