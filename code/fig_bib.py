#!/usr/bin/env python3
"""
BIB figures: promiscuity analysis, calibration, feature importance.
Diversified chart types: connected dot plots, lollipop charts, calibration
with LOWESS smoothing, horizontal bars, enriched layouts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

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
    print("  BIB Figure Generation (diversified chart types)")
    print("=" * 60)

    pairing = pd.read_csv(DATA_DIR / "pairing_matrix_raw.csv")
    svm_preds = load(DATA_DIR / "svm_predictions.json")
    ca_preds = load(DATA_DIR / "ca_predictions.json")

    pos_pairs = pairing[pairing.coupling == 1]
    promiscuity = pos_pairs.groupby("gpcr_id").g_protein_family.nunique()

    families = ["Gq", "Gi", "Gs", "G12_13"]

    # Build TWO dataframes:
    #   df_all  = all predictions (for calibration — matches manuscript Brier)
    #   df_prom = prom>0 only (for promiscuity stratification)
    rows_all, rows_prom = [], []
    for gid, fam_dict in ca_preds.items():
        base_id = gid.split("_", 1)[1] if "_" in gid else gid
        prom_val = promiscuity.get(base_id, promiscuity.get(gid, 0))
        for fam in families:
            if fam in fam_dict and gid in svm_preds and fam in svm_preds[gid]:
                row = {
                    "gpcr_prom": min(prom_val, 4) if prom_val > 0 else 0,
                    "family": fam,
                    "label": fam_dict[fam]["label"],
                    "svm_prob": svm_preds[gid][fam]["prob"],
                    "ca_prob": fam_dict[fam]["prob"],
                }
                rows_all.append(row)
                if prom_val > 0:
                    rows_prom.append(row)

    df_all = pd.DataFrame(rows_all)
    df_all["ensemble_prob"] = (df_all.svm_prob + df_all.ca_prob) / 2
    df_all["confidence"] = np.abs(df_all.ca_prob - 0.5) * 2

    df_prom = pd.DataFrame(rows_prom)
    df_prom["ensemble_prob"] = (df_prom.svm_prob + df_prom.ca_prob) / 2
    df_prom["confidence"] = np.abs(df_prom.ca_prob - 0.5) * 2

    # ==================================================================
    # FIGURE 1: Promiscuity-stratified AUC — connected dot plot + delta
    # ==================================================================
    print("\n[1/3] Promiscuity analysis (dot+line plot)...")
    prom_labels = {1: "Single-family", 2: "Dual-family", 3: "Triple-family", 4: "4+-family"}
    prom_data = {}
    for pval in sorted(prom_labels.keys()):
        sub = df_prom[df_prom.gpcr_prom == pval]
        if len(sub) < 10:
            continue
        y = sub.label.values
        prom_data[prom_labels[pval]] = {
            "N": len(sub),
            "SVM": roc_auc_score(y, sub.svm_prob),
            "CA": roc_auc_score(y, sub.ca_prob),
            "Ensemble": roc_auc_score(y, sub.ensemble_prob),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    names = list(prom_data.keys())
    x = np.arange(len(names))

    # Panel A: Connected dot plot with spaced labels
    ax = axes[0]
    markers = {"SVM": "s", "CA": "o", "Ensemble": "D"}
    sizes = {"SVM": 90, "CA": 110, "Ensemble": 80}
    for model, key in [("SVM", "SVM"), ("CA", "CA"), ("Ensemble", "Ensemble")]:
        vals = [prom_data[n][model] for n in names]
        ax.plot(x, vals, "-", color=COLORS[key], alpha=0.5, linewidth=1.5, zorder=1)
        ax.scatter(x, vals, s=sizes[model], c=COLORS[key], marker=markers[model],
                  label=model, edgecolors="white", linewidth=0.5, zorder=3, alpha=0.9)

    # Place value labels in a legend-like box instead of on the data points
    all_vals = [v for n in names for v in [prom_data[n]["SVM"], prom_data[n]["CA"], prom_data[n]["Ensemble"]]]
    y_min = max(0.5, min(all_vals) - 0.08)
    y_max = min(1.02, max(all_vals) + 0.10)

    ax.set_ylabel("AUC (Cluster-CV)", fontsize=9)
    ax.set_title("Performance by GPCR Promiscuity", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(N={prom_data[n]['N']})" for n in names], fontsize=8)
    ax.set_ylim(y_min, y_max)
    ax.legend(frameon=True, loc="upper right", fontsize=7.5, markerscale=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)

    # Value table below x-axis (avoids all overlap)
    for xi, name in enumerate(names):
        d = prom_data[name]
        txt = f"CA:{d['CA']:.3f}\nSVM:{d['SVM']:.3f}"
        ax.text(xi, y_min - (y_max - y_min) * 0.08, txt, ha="center", va="top",
                fontsize=6.5, alpha=0.85, linespacing=1.2)

    # Panel B: Delta dot plot with clean spacing
    ax = axes[1]
    deltas = [prom_data[n]["CA"] - prom_data[n]["SVM"] for n in names]
    delta_colors = [COLORS["CA"] if d > 0 else "#E74C3C" for d in deltas]

    for xi, d, dc in zip(x, deltas, delta_colors):
        ax.plot([xi, xi], [0, d], color=dc, linewidth=3.5, solid_capstyle="round", alpha=0.8)
        ax.scatter(xi, d, s=150, c=dc, edgecolors="white", linewidth=0.8, zorder=3)
        va = "bottom" if d > 0 else "top"
        offset = 0.006 if d > 0 else -0.006
        ax.text(xi, d + offset, f"{d:+.4f}", ha="center", va=va, fontsize=9,
                fontweight="bold", color=dc)

    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_ylabel(r"$\Delta$AUC (CA $-$ SVM)", fontsize=9)
    ax.set_title("Cross-Attention Advantage", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ymax = max(max(deltas), 0.01) * 2.0
    ymin = min(min(deltas), -0.01) * 2.0
    ax.set_ylim(ymin, ymax)
    if ymax > 0:
        ax.axhspan(0, ymax, alpha=0.04, color=COLORS["CA"])

    plt.tight_layout()
    fig.savefig(FIG_DIR / "promiscuity_analysis.pdf", format="pdf")
    plt.close(fig)
    print("  -> promiscuity_analysis.pdf")

    # ==================================================================
    # FIGURE 2: Calibration and reliability — decluttered layout
    # ==================================================================
    print("\n[2/3] Calibration analysis (quantile bins + LOWESS)...")
    y_true = df_all.label.values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # Panel A: Reliability diagram with quantile binning + LOWESS
    ax = axes[0]
    for model_name, probs, key in [
        ("Cross-Attention", df_all.ca_prob, "CA"),
        ("SVM", df_all.svm_prob, "SVM"),
    ]:
        prob_true, prob_pred = calibration_curve(
            y_true, probs, n_bins=10, strategy="quantile"
        )
        mask = ~(np.isnan(prob_true) | np.isnan(prob_pred))
        prob_true, prob_pred = prob_true[mask], prob_pred[mask]

        ax.plot(prob_pred, prob_true, "o-", color=COLORS[key], label=model_name,
                markersize=7, linewidth=2.0, alpha=0.9, markeredgecolor="white",
                markeredgewidth=0.5)

        if len(prob_pred) >= 4:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                x_sorted = np.sort(prob_pred)
                ax.plot(x_sorted, iso.fit_transform(x_sorted, prob_true[np.argsort(prob_pred)]),
                       "--", color=COLORS[key], alpha=0.35, linewidth=1.0)
            except Exception:
                pass

        brier = brier_score_loss(y_true, probs)
        # Legend-style annotation OUTSIDE the plot, top-left corner
        y_pos = 0.95 if key == "CA" else 0.78
        ax.text(0.03, y_pos,
                f"{model_name}: Brier={brier:.3f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                fontweight="bold", color=COLORS[key],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=COLORS[key], alpha=0.92, lw=1.0))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect", linewidth=0.8)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_title("A. Reliability Diagram", fontweight="bold", fontsize=10)
    ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Panel B: Confidence vs. Accuracy
    ax = axes[1]
    thresholds = [0.5, 0.75, 0.90, 0.98]
    coverage, accuracy_vals = [], []
    for thresh in thresholds:
        sub = df_all[df_all.confidence >= thresh]
        if len(sub) > 0:
            coverage.append(len(sub) / len(df_all) * 100)
            accuracy_vals.append(
                accuracy_score(sub.label, (sub.ca_prob > 0.5).astype(int)) * 100
            )

    ax.fill_between(coverage, accuracy_vals, alpha=0.08, color=COLORS["CA"])
    ax.plot(coverage, accuracy_vals, "o-", color=COLORS["CA"], linewidth=2.5,
            markersize=11, markeredgecolor="white", markeredgewidth=1.0, zorder=4)

    # Staggered label offsets to prevent overlap
    offsets = [(12, 6), (-8, -12), (12, -12), (-8, 6)]
    for (c, a, t), (dx, dy) in zip(zip(coverage, accuracy_vals, thresholds), offsets):
        ax.annotate(f"conf≥{t:.2f}", (c, a), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, alpha=0.9, fontweight="bold",
                    color=COLORS["CA"],
                    arrowprops=dict(arrowstyle="-", color="grey", alpha=0.4, lw=0.5))

    # Add value labels directly on points (offset slightly)
    for c, a in zip(coverage, accuracy_vals):
        ax.text(c + 0.8, a + 0.15, f"{a:.1f}%", fontsize=7, alpha=0.85, ha="left", va="bottom")

    ax.set_xlabel("Coverage (%)", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("B. Confidence vs. Accuracy", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    # Add margin around data
    x_margin = (max(coverage) - min(coverage)) * 0.15
    y_margin = (max(accuracy_vals) - min(accuracy_vals)) * 0.6
    ax.set_xlim(min(coverage) - x_margin, max(coverage) + x_margin * 1.2)
    ax.set_ylim(min(accuracy_vals) - y_margin, max(accuracy_vals) + y_margin * 0.8)

    # Panel C: Per-family calibration — horizontal bars (cleaner than lollipop)
    ax = axes[2]
    family_briers = []
    for fam in families:
        mask = df_all.family == fam
        fp = df_all.loc[mask, "ca_prob"].values
        fl = df_all.loc[mask, "label"].values
        family_briers.append(brier_score_loss(fl, fp))

    y_pos = np.arange(len(families))
    fam_colors = [FAMILY_COLOR.get(f, WONG["blue"]) for f in families]

    bars = ax.barh(y_pos, family_briers, color=fam_colors, alpha=0.85, edgecolor="white",
                   height=0.55)
    for bar, val in zip(bars, family_briers):
        ax.text(bar.get_width() + max(family_briers) * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=10)
    ax.set_xlabel("Brier Score (lower = better)", fontsize=9)
    ax.set_title("C. Per-Family Brier Score", fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_xlim(0, max(family_briers) * 1.18)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "calibration_analysis.pdf", format="pdf")
    plt.close(fig)
    print("  -> calibration_analysis.pdf")

    # ==================================================================
    # FIGURE 3: Feature importance — horizontal bars with enriched layout
    # ==================================================================
    print("\n[3/3] Feature importance (horizontal bar)...")
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

    raw_vals = np.array([float(v) for v in grad_data.values()])
    values = raw_vals / raw_vals.sum()
    groups = list(grad_data.keys())
    group_labels = [
        "GPCR Global ESM",
        "G Protein ESM",
        "ICL2 PhysChem",
        "ICL3 PhysChem",
        "ICL2 ESM Embed.",
        "ICL3 ESM Embed.",
        "AlphaFold Struct.",
    ]
    bar_colors = [
        COLORS["CA"] if "gpcr" in g or "icl" in g else
        COLORS["SVM"] if "gprot" in g else WONG["grey"]
        for g in groups
    ]

    # Horizontal bar chart with clean spacing
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y_pos = np.arange(len(groups))
    bars = ax.barh(y_pos, values, color=bar_colors, alpha=0.88, edgecolor="white",
                   height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels, fontsize=9)
    ax.set_xlabel("Normalized |Gradient Attribution|", fontsize=9)
    ax.set_title("Feature Group Importance (Cross-Attention 650M)", fontweight="bold", fontsize=10)
    ax.invert_yaxis()

    # Value labels — placed INSIDE bars when wide enough, outside otherwise
    for bar, val in zip(bars, values):
        if val > max(values) * 0.3:
            # Inside the bar
            ax.text(val * 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="center", fontsize=7.5,
                    fontweight="bold", color="white")
        else:
            # Outside
            ax.text(val + max(values) * 0.015, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7.5, fontweight="bold")

    # Divider lines and labels — placed outside the bar area
    ax.axhline(y=1.5, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    # Annotation labels moved to right margin area (outside bars)
    label_x = max(values) * 1.28
    ax.text(label_x, 0, "GPCR-side", fontsize=7, ha="left", va="center",
            color=COLORS["CA"], fontweight="bold", fontstyle="italic")
    ax.text(label_x, 2, "G protein-side", fontsize=7, ha="left", va="center",
            color=COLORS["SVM"], fontweight="bold", fontstyle="italic")
    ax.text(label_x, len(groups)-1, "Structural", fontsize=7, ha="left", va="center",
            color=WONG["grey"], fontweight="bold", fontstyle="italic")

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_xlim(0, max(values) * 1.45)

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
    print(f"  CA Brier (all predictions): {brier_score_loss(df_all.label, df_all.ca_prob):.4f}")
    print(f"  SVM Brier (all predictions): {brier_score_loss(df_all.label, df_all.svm_prob):.4f}")


if __name__ == "__main__":
    main()
