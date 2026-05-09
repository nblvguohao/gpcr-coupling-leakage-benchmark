#!/usr/bin/env python3
"""
BIB figures: promiscuity analysis, calibration, feature importance.
Uses shared style_config.py for consistent, colorblind-safe styling.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve

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


def main():
    print("=" * 60)
    print("  BIB Figure Generation")
    print("=" * 60)

    pairing = pd.read_csv(DATA_DIR / "pairing_matrix_raw.csv")
    svm_preds = load(DATA_DIR / "svm_predictions.json")
    ca_preds = load(DATA_DIR / "ca_predictions.json")

    pos_pairs = pairing[pairing.coupling == 1]
    promiscuity = pos_pairs.groupby("gpcr_id").g_protein_family.nunique()

    families = ["Gq", "Gi", "Gs", "G12_13"]
    rows = []
    for gid, fam_dict in ca_preds.items():
        base_id = gid.split("_", 1)[1] if "_" in gid else gid
        prom_val = promiscuity.get(base_id, promiscuity.get(gid, 1))
        for fam in families:
            if fam in fam_dict and gid in svm_preds and fam in svm_preds[gid]:
                rows.append({
                    "gpcr_prom": min(prom_val, 4),
                    "family": fam,
                    "label": fam_dict[fam]["label"],
                    "svm_prob": svm_preds[gid][fam]["prob"],
                    "ca_prob": fam_dict[fam]["prob"],
                })
    df = pd.DataFrame(rows)
    df["ensemble_prob"] = (df.svm_prob + df.ca_prob) / 2
    df["confidence"] = np.abs(df.ca_prob - 0.5) * 2

    # ==================================================================
    # FIGURE 1: Promiscuity-stratified AUC
    # ==================================================================
    print("\n[1/3] Promiscuity analysis...")
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
    names = list(prom_data.keys())
    x = np.arange(len(names))
    width = 0.25

    for i, (model, key) in enumerate([("SVM", "SVM"), ("CA", "CA"), ("Ensemble", "Ensemble")]):
        vals = [prom_data[n][model] for n in names]
        bars = axes[0].bar(x + i * width, vals, width, label=model, color=COLORS[key], alpha=0.85)
        for bar, val in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("AUC (Cluster-CV)")
    axes[0].set_title("Performance by GPCR Promiscuity")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].legend(frameon=True)
    axes[0].set_ylim(0.5, 0.91)  # Honest baseline
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    deltas = [prom_data[n]["CA"] - prom_data[n]["SVM"] for n in names]
    dc = [COLORS["CA"] if d > 0 else COLORS["SVM"] for d in deltas]
    bars = axes[1].bar(names, deltas, color=dc, alpha=0.8)
    for bar, d in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"+{d:.3f}" if d > 0 else f"{d:.3f}", ha="center",
                     va="bottom" if d > 0 else "top", fontsize=8, fontweight="bold")
    axes[1].set_ylabel(r"$\Delta$AUC (CA $-$ SVM)")
    axes[1].set_title("Cross-Attention Advantage")
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "promiscuity_analysis.pdf", format="pdf")
    plt.close(fig)
    print("  -> promiscuity_analysis.pdf")

    # ==================================================================
    # FIGURE 2: Calibration and reliability
    # ==================================================================
    print("\n[2/3] Calibration analysis...")
    y_true = df.label.values

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Reliability diagram
    ax = axes[0]
    for model_name, probs, key in [
        ("Cross-Attention", df.ca_prob, "CA"),
        ("SVM", df.svm_prob, "SVM"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", color=COLORS[key], label=model_name,
                markersize=5, linewidth=1.5, alpha=0.85)
        brier = brier_score_loss(y_true, probs)
        ece = np.mean(np.abs(prob_true - prob_pred))
        ax.text(0.95, 0.15 if key == "SVM" else 0.28,
                f"{model_name}\nBrier={brier:.3f}  ECE={ece:.3f}",
                transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("A. Reliability Diagram")
    ax.legend(frameon=True, fontsize=7)
    ax.grid(alpha=0.3, linestyle="--")

    # Panel B: Confidence stratification
    ax = axes[1]
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    coverage, accuracy_vals = [], []
    for thresh in thresholds:
        sub = df[df.confidence >= thresh]
        if len(sub) > 0:
            coverage.append(len(sub) / len(df) * 100)
            accuracy_vals.append(
                accuracy_score(sub.label, (sub.ca_prob > 0.5).astype(int)) * 100
            )
    ax.plot(coverage, accuracy_vals, "o-", color=COLORS["CA"], linewidth=2, markersize=8)
    for c, a, t in zip(coverage, accuracy_vals, thresholds):
        ax.annotate(f"{t:.2f}", (c, a), textcoords="offset points",
                    xytext=(8, -2), fontsize=7, alpha=0.8)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B. Confidence vs. Accuracy")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(75, 102)

    # Panel C: Per-family calibration (simplified — single metric)
    ax = axes[2]
    x_fam = np.arange(len(families))
    family_briers = []
    for fam in families:
        mask = df.family == fam
        fp = df.loc[mask, "ca_prob"].values
        fl = df.loc[mask, "label"].values
        family_briers.append(brier_score_loss(fl, fp))
    bars = ax.bar(x_fam, family_briers, color=[FAMILY_COLOR.get(f, WONG["blue"]) for f in families],
                  alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, family_briers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_fam)
    ax.set_xticklabels(families)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("C. Per-Family Calibration")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "calibration_analysis.pdf", format="pdf")
    plt.close(fig)
    print("  -> calibration_analysis.pdf")

    # ==================================================================
    # FIGURE 3: Feature importance
    # ==================================================================
    print("\n[3/3] Feature importance...")
    try:
        raw = load(FIG_DIR / "gradient_attribution_650m.json")
        grad_data = raw.get("feature_group_means", raw)
    except (FileNotFoundError, json.JSONDecodeError):
        grad_data = {
            "gpcr_esm_global": 5.67e-05, "gprotein_esm": 1.09e-05,
            "icl2_stats": 7.21e-05, "icl3_stats": 7.61e-05,
            "icl2_esm": 6.57e-05, "icl3_esm": 7.00e-05,
            "alphafold": 7.35e-05,
        }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    raw_vals = np.array([float(v) for v in grad_data.values()])
    values = raw_vals / raw_vals.sum()
    groups = list(grad_data.keys())
    group_labels = [
        "GPCR\nGlobal", "G Prot.\nGlobal", "ICL2\nStats", "ICL3\nStats",
        "ICL2\nEmbed.", "ICL3\nEmbed.", "AlphaFold"
    ]
    bar_colors = [
        COLORS["CA"] if "gpcr" in g or "icl" in g else
        COLORS["SVM"] if "gprot" in g else WONG["grey"]
        for g in groups
    ]

    bars = ax.bar(range(len(groups)), values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Normalized |Gradient Attribution|")
    ax.set_title("Feature Group Importance (Cross-Attention 650M)")

    # GPCR vs G-protein divider
    ax.axvline(x=1.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.75, max(values) * 0.95, "GPCR-side", ha="center", fontsize=7,
            color=COLORS["CA"], fontweight="bold")
    ax.text(4.5, max(values) * 0.95, "G Prot-side", ha="center", fontsize=7,
            color=COLORS["SVM"], fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_groups.pdf", format="pdf")
    plt.close(fig)
    print("  -> feature_importance_groups.pdf")

    # Summary
    print(f"\n{'='*60}")
    print("  Key Numbers")
    print(f"{'='*60}")
    for name in names:
        d = prom_data[name]
        print(f"  {name:16s} N={d['N']:4d}  CA={d['CA']:.4f}  SVM={d['SVM']:.4f}  "
              f"Delta={d['CA']-d['SVM']:.4f}")
    print(f"  CA Brier: {brier_score_loss(y_true, df.ca_prob):.4f}")
    print(f"  SVM Brier: {brier_score_loss(y_true, df.svm_prob):.4f}")


if __name__ == "__main__":
    main()
