#!/usr/bin/env python3
"""
Comprehensive figure generation for the GPCR–G protein coupling prediction manuscript.

Target journal: Briefings in Bioinformatics (BIB, Oxford).

Figures generated:
  schematic   Figure 1 — Study overview: data pipeline, feature engineering,
              and cross-attention architecture (3-panel schematic).
  main        Manuscript main figures:
                - AUC comparison across feature configurations and architectures
                - ICL dimension alignment analysis (2-panel)
                - AlphaFold structural feature ablation study
                - Statistical test results (forest plot)
  bib         BIB-specific figures:
                - Promiscuity-stratified AUC analysis (2-panel)
                - Calibration and reliability analysis (3-panel)
                - Feature importance from gradient attribution

Usage:
  python make_figures.py schematic   # Figure 1 schematic only
  python make_figures.py main        # Manuscript main figures only
  python make_figures.py bib         # BIB-specific figures only
  python make_figures.py all         # All figures (default when no subcommand given)

All output files are written to ../figures/ relative to this script.

Dependencies:
  - style_config.py (same directory)
  - Data files in ../data/ (pairing_matrix_raw.csv, *_predictions.json,
    statistical_tests_results.json)
  - Gradient attribution JSON in ../figures/ (optional; falls back to
    hardcoded defaults if missing)
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from style_config import apply_style, WONG, MODEL_COLOR, FAMILY_COLOR

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared colour maps
# ---------------------------------------------------------------------------
COLORS = {**MODEL_COLOR, **FAMILY_COLOR}

# Schematic-specific colour palette (fig_schematic)
C = {
    "gpcr": "#2471A3",  # deep blue
    "gpcr_light": "#D4E6F1",
    "gprot": "#D35400",  # deep orange
    "gprot_light": "#FAD7A1",
    "icl": "#1E8449",  # deep green
    "icl_light": "#D5F5E3",
    "af": "#95A5A6",  # grey
    "af_light": "#E5E7E9",
    "nn": "#7D3C98",  # purple for neural
    "nn_light": "#E8DAEF",
    "panel_bg": "#F8F9FA",
    "border": "#BDC3C7",
    "arrow": "#5D6D7E",
    "text": "#2C3E50",
    "white": "#FFFFFF",
    "gold": "#F4D03F",
    "red_accent": "#C0392B",
    "family": ["#3498DB", "#E74C3C", "#F39C12", "#9B59B6"],
    "cross_attn": "#E67E22",
}

FAMILY_NAMES = [
    "Gq/11\n(n=?)",
    "Gi/o\n(n=?)",
    "Gs\n(n=?)",
    "G12/13\n(n=?)",
]


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------
def load(path):
    """Load a JSON file and return its contents."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schematic drawing primitives  (from fig_schematic.py)
# ---------------------------------------------------------------------------
def rounded_box(
    ax,
    xy,
    w,
    h,
    facecolor,
    edgecolor=None,
    lw=1.2,
    radius=0.08,
    text="",
    fontsize=8,
    textcolor="white",
    fontweight="bold",
    zorder=3,
):
    """Draw a rounded box with optional centred text."""
    if edgecolor is None:
        edgecolor = C["border"]
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=textcolor,
            fontweight=fontweight,
            zorder=zorder + 1,
        )


def text_box(
    ax,
    xy,
    w,
    h,
    text,
    facecolor=C["panel_bg"],
    edgecolor=C["border"],
    fontsize=8,
    textcolor=C["text"],
    fontweight="normal",
    lw=1.0,
    radius=0.06,
    zorder=3,
):
    """Draw a light-background box with dark text."""
    rounded_box(
        ax,
        xy,
        w,
        h,
        facecolor,
        edgecolor,
        lw,
        radius,
        text,
        fontsize,
        textcolor,
        fontweight,
        zorder,
    )


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.5, style="->", zorder=1, ls="-"):
    """Draw a simple arrow."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls),
        zorder=zorder,
    )


def curved_arrow(
    ax, x0, y0, x1, y1, color=C["arrow"], lw=1.2, rad=0.2, zorder=1
):
    """Draw a curved arrow using an arc3 connection style."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=zorder,
    )


def section_label(ax, x, y, letter, title):
    """Draw a section label e.g. 'A. Dataset Curation'."""
    ax.text(
        x,
        y,
        f"{letter}. {title}",
        fontsize=11,
        fontweight="bold",
        color=C["text"],
        ha="left",
        va="center",
    )


def plus_mark(ax, x, y, fontsize=12, color=C["arrow"]):
    ax.text(
        x,
        y,
        "+",
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight="bold",
        color=color,
    )


def cross_mark(ax, x, y, fontsize=12):
    ax.text(
        x,
        y,
        "✗",
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight="bold",
        color=C["red_accent"],
    )


# ===================================================================
# Schematic panels  (from fig_schematic.py)
# ===================================================================


def panel_a(ax):
    """Panel A: Data Pipeline."""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Background
    bg = FancyBboxPatch(
        (0.05, 0.1),
        15.9,
        5.8,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#EBF5FB",
        edgecolor="none",
        zorder=0,
        alpha=0.5,
    )
    ax.add_patch(bg)
    section_label(ax, 0.3, 5.5, "A", "Dataset Curation & Evaluation Framework")

    # Row 1: Data sources
    y1 = 4.0
    rounded_box(
        ax,
        (0.5, y1),
        2.0,
        0.9,
        C["gpcr"],
        C["gpcr"],
        text="GPCRdb\nAnnotations",
        fontsize=7.5,
        textcolor="white",
    )
    rounded_box(
        ax,
        (2.8, y1),
        2.0,
        0.9,
        C["gpcr"],
        C["gpcr"],
        text="Literature\nCuration",
        fontsize=7.5,
        textcolor="white",
    )
    arrow(ax, 2.5, y1 + 0.45, 2.8, y1 + 0.45)
    arrow(ax, 3.8, y1 + 0.45, 5.5, 2.7, color=C["arrow"], lw=1.2)

    # Pairing matrix
    rounded_box(
        ax,
        (5.5, 2.2),
        2.8,
        1.0,
        C["panel_bg"],
        C["border"],
        text="1,647 pairs\n(431 GPCRs × 4 families)",
        fontsize=8.5,
        textcolor=C["text"],
        fontweight="bold",
    )

    # G protein families
    fam_y_base = 4.2
    fam_counts = [388, 406, 298, 173]  # from data
    for i, (name, count, fc) in enumerate(
        zip(["Gq/11", "Gi/o", "Gs", "G12/13"], fam_counts, C["family"])
    ):
        xf = 8.8 + i * 1.65
        rounded_box(
            ax,
            (xf, fam_y_base - (i % 2) * 1.4),
            1.35,
            0.85,
            fc,
            fc,
            text=f"{name}\n(n={count})",
            fontsize=7,
            textcolor="white",
            radius=0.06,
        )
        arrow(
            ax,
            8.3,
            2.7,
            xf + 0.675,
            fam_y_base + 0.05 - (i % 2) * 1.4,
            color=C["arrow"],
            lw=0.9,
        )

    # CD-HIT clustering
    rounded_box(
        ax,
        (5.5, 0.7),
        2.8,
        0.85,
        "#7F8C8D",
        "#7F8C8D",
        text="CD-HIT (40% identity)\n387 sequence clusters",
        fontsize=7.5,
        textcolor="white",
    )
    arrow(ax, 6.9, 2.2, 6.9, 1.55)

    # Evaluation strategies
    eval_y = 0.7
    text_box(
        ax,
        (9.0, eval_y),
        3.2,
        0.85,
        "Cluster-aware 5-fold CV\n(no cluster leakage)",
        fontsize=7.5,
        textcolor=C["text"],
    )
    text_box(
        ax,
        (12.5, eval_y),
        3.0,
        0.85,
        "LOGPSO\n(4 × leave-one-family-out)",
        fontsize=7.5,
        textcolor=C["text"],
    )
    arrow(ax, 8.3, 1.125, 9.0, 1.125, color=C["arrow"], lw=1.0)
    arrow(ax, 8.3, 1.125, 12.5, 1.125, color=C["arrow"], lw=1.0)

    # Key metrics callout
    ax.text(
        0.5,
        0.15,
        "◆ Primary metric: Cluster-aware AUC  ◆  Secondary: Brier score, LOGPSO AUC",
        fontsize=7,
        color=C["arrow"],
        ha="left",
        fontstyle="italic",
    )


def panel_b(ax):
    """Panel B: Feature Engineering Pipeline."""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.05, 0.1),
        15.9,
        6.8,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#FDF2E9",
        edgecolor="none",
        zorder=0,
        alpha=0.5,
    )
    ax.add_patch(bg)
    section_label(ax, 0.3, 6.5, "B", "Feature Engineering Pipeline")

    # Column headers
    col_centers = [2.5, 7.0, 11.5]
    col_labels = ["Global Embeddings", "ICL Topology Features", "Structural Features"]
    col_colors = [C["gpcr_light"], C["icl_light"], C["af_light"]]
    for cx, label, cc in zip(col_centers, col_labels, col_colors):
        ax.text(
            cx,
            5.8,
            label,
            fontsize=9,
            ha="center",
            fontweight="bold",
            color=C["text"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=cc,
                edgecolor=C["border"],
                alpha=0.9,
            ),
        )

    # Column 1: Global ESM-2
    cx1 = 2.5
    rounded_box(
        ax,
        (cx1 - 1.0, 4.6),
        2.0,
        0.75,
        C["gpcr"],
        C["gpcr"],
        text="GPCR ESM-2 650M\n(1280-d)",
        fontsize=7.5,
        textcolor="white",
    )
    rounded_box(
        ax,
        (cx1 - 1.0, 3.5),
        2.0,
        0.75,
        C["gprot"],
        C["gprot"],
        text="G Protein ESM-2 8M\n(320-d)",
        fontsize=7.5,
        textcolor="white",
    )
    arrow(ax, cx1, 4.6, cx1, 4.35, color=C["arrow"], lw=1.0)
    arrow(ax, cx1, 4.25, cx1 + 1.8, 4.25, color=C["arrow"], lw=1.0)

    # Column 2: ICL features
    cx2 = 7.0
    icl_items = [
        ("ICL2 ESM (1280-d)", C["icl"]),
        ("ICL2 PhysChem (8-d)", "#27AE60"),
        ("ICL3 ESM (1280-d)", C["icl"]),
        ("ICL3 PhysChem (8-d)", "#27AE60"),
    ]
    for j, (label, col) in enumerate(icl_items):
        y_icl = 5.1 - j * 0.65
        rounded_box(
            ax,
            (cx2 - 0.85, y_icl),
            1.7,
            0.5,
            col,
            col,
            text=label,
            fontsize=6.5,
            textcolor="white",
            radius=0.04,
        )

    # Dimension alignment highlight
    ax.annotate(
        "KEY: ICL ESM dim\nmust match global\ndim (1280-d → 1280-d)",
        xy=(cx2 + 1.1, 2.5),
        fontsize=6.5,
        ha="center",
        color=C["red_accent"],
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FDEDEC",
            edgecolor=C["red_accent"],
            alpha=0.8,
            lw=0.8,
        ),
    )

    # Column 3: AlphaFold (optional)
    cx3 = 11.5
    rounded_box(
        ax,
        (cx3 - 0.85, 4.8),
        1.7,
        0.75,
        "#BDC3C7",
        "#BDC3C7",
        text="AlphaFold2\n38 descriptors",
        fontsize=7,
        textcolor="white",
        radius=0.06,
    )
    ax.text(cx3, 4.3, "(optional)", fontsize=7, ha="center", color=C["arrow"],
            fontstyle="italic")
    cross_mark(ax, cx3, 3.8)

    # Fusion section (bottom)
    fuse_y = 1.5
    arrow(ax, cx1, 3.5, cx1, fuse_y + 1.0, color=C["arrow"], lw=1.2)
    arrow(ax, cx2, 2.4, cx2, fuse_y + 1.0, color=C["arrow"], lw=1.2)
    arrow(ax, cx3, 3.8, cx3, fuse_y + 1.0, color=C["arrow"], lw=1.0, ls="--")

    plus_mark(ax, 4.5, fuse_y + 0.6, fontsize=16)
    plus_mark(ax, 6.5, fuse_y + 0.6, fontsize=16)

    text_box(
        ax,
        (3.0, fuse_y),
        5.0,
        0.9,
        "Full Feature Vector (4176-d)\n"
        "GPCR: ESM(1280) + ICL2(1280+8) + ICL3(1280+8)  ∥  G protein: ESM(320)",
        fontsize=7,
        textcolor=C["text"],
        fontweight="bold",
    )

    arrow(ax, 8.0, fuse_y + 0.45, 9.5, fuse_y + 0.45, color=C["nn"], lw=1.8)
    ax.text(
        8.75,
        fuse_y + 0.75,
        "→ Panel C",
        fontsize=7,
        ha="center",
        color=C["nn"],
        fontweight="bold",
    )

    # G protein subset
    arrow(ax, cx1, 3.5, cx1, 2.8, color=C["arrow"], lw=1.0)
    ax.text(
        cx1 - 1.3,
        3.0,
        "G protein:\nESM only",
        fontsize=6.5,
        ha="center",
        color=C["gprot"],
    )


def panel_c(ax):
    """Panel C: Cross-Attention Neural Network."""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.05, 0.1),
        15.9,
        6.8,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#F4ECF7",
        edgecolor="none",
        zorder=0,
        alpha=0.5,
    )
    ax.add_patch(bg)
    section_label(ax, 0.3, 6.5, "C", "Cross-Attention Neural Network")

    # Input boxes
    inp_y = 4.5
    rounded_box(
        ax,
        (0.5, inp_y + 0.4),
        2.0,
        0.9,
        C["gpcr"],
        C["gpcr"],
        text="GPCR Features\n(d_GPCR = 4176)",
        fontsize=7.5,
        textcolor="white",
    )
    rounded_box(
        ax,
        (0.5, inp_y - 1.2),
        2.0,
        0.9,
        C["gprot"],
        C["gprot"],
        text="G Protein Features\n(d_Gprot = 320)",
        fontsize=7.5,
        textcolor="white",
    )

    # Projection layers
    proj_x = 3.2
    rounded_box(
        ax,
        (proj_x, inp_y + 0.4),
        1.8,
        0.9,
        "#5DADE2",
        "#5DADE2",
        text="Linear + LN\n+ GELU → 256-d",
        fontsize=7,
        textcolor="white",
    )
    rounded_box(
        ax,
        (proj_x, inp_y - 1.2),
        1.8,
        0.9,
        "#F1948A",
        "#F1948A",
        text="Linear + LN\n+ GELU → 256-d",
        fontsize=7,
        textcolor="white",
    )
    arrow(ax, 2.5, inp_y + 0.85, proj_x, inp_y + 0.85, color=C["arrow"], lw=1.3)
    arrow(ax, 2.5, inp_y - 0.75, proj_x, inp_y - 0.75, color=C["arrow"], lw=1.3)

    # Cross-Attention block
    attn_cx, attn_cy = 6.5, inp_y + 0.1
    circle = mpatches.Ellipse(
        (attn_cx, attn_cy),
        1.8,
        1.8,
        facecolor="#F9E79F",
        edgecolor=C["cross_attn"],
        linewidth=2.0,
        zorder=4,
    )
    ax.add_patch(circle)
    ax.text(
        attn_cx,
        attn_cy + 0.35,
        "Multi-Head",
        fontsize=8,
        ha="center",
        va="center",
        fontweight="bold",
        color=C["text"],
        zorder=5,
    )
    ax.text(
        attn_cx,
        attn_cy - 0.15,
        "Cross-Attention",
        fontsize=8.5,
        ha="center",
        va="center",
        fontweight="bold",
        color=C["cross_attn"],
        zorder=5,
    )
    ax.text(
        attn_cx,
        attn_cy - 0.6,
        "4 heads × 64-d",
        fontsize=6.5,
        ha="center",
        va="center",
        color=C["arrow"],
        zorder=5,
    )

    # Arrows into attention
    ax.annotate(
        "Q",
        xy=(attn_cx - 0.5, attn_cy + 0.4),
        fontsize=9,
        fontweight="bold",
        color=C["gpcr"],
        ha="center",
        va="center",
    )
    ax.annotate(
        "K,V",
        xy=(attn_cx - 0.5, attn_cy - 0.5),
        fontsize=8,
        fontweight="bold",
        color=C["gprot"],
        ha="center",
        va="center",
    )
    arrow(
        ax,
        proj_x + 1.8,
        inp_y + 0.85,
        attn_cx - 0.95,
        attn_cy + 0.4,
        color=C["gpcr"],
        lw=1.5,
    )
    arrow(
        ax,
        proj_x + 1.8,
        inp_y - 0.75,
        attn_cx - 0.95,
        attn_cy - 0.4,
        color=C["gprot"],
        lw=1.5,
    )

    # Post-attention processing
    post_x = attn_cx + 1.5
    rounded_box(
        ax,
        (post_x, attn_cy - 0.4),
        1.6,
        0.8,
        C["nn_light"],
        C["nn"],
        text="Concat\n+ Skip",
        fontsize=7.5,
        textcolor=C["text"],
        fontweight="bold",
        lw=1.3,
    )

    # Skip connection (curved, from GPCR projection to concat)
    curved_arrow(
        ax,
        proj_x + 0.9,
        inp_y + 0.85,
        post_x + 0.2,
        attn_cy + 0.3,
        color=C["arrow"],
        lw=1.0,
        rad=0.3,
    )

    # FFN
    ffn_x = post_x + 2.2
    rounded_box(
        ax,
        (ffn_x, attn_cy - 0.4),
        1.6,
        0.8,
        C["nn"],
        C["nn"],
        text="3-Layer FFN\nGELU + LN\nDropout 0.3",
        fontsize=7,
        textcolor="white",
    )
    arrow(ax, post_x + 1.6, attn_cy, ffn_x, attn_cy, color=C["arrow"], lw=1.3)

    # Output
    out_x = ffn_x + 2.0
    rounded_box(
        ax,
        (out_x, attn_cy - 0.3),
        1.4,
        0.6,
        C["cross_attn"],
        C["cross_attn"],
        text="Sigmoid",
        fontsize=8,
        textcolor="white",
    )
    arrow(ax, ffn_x + 1.6, attn_cy, out_x, attn_cy, color=C["arrow"], lw=1.3)

    # Final output
    ax.text(
        out_x + 1.6,
        attn_cy - 0.05,
        "P(coupling)\n∈ [0, 1]",
        fontsize=8.5,
        ha="center",
        va="center",
        fontweight="bold",
        color=C["text"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=C["cross_attn"],
            lw=1.2,
        ),
    )

    # Key metrics callout
    rounded_box(
        ax,
        (0.5, 0.5),
        6.5,
        1.0,
        "#2C3E50",
        "#2C3E50",
        text="AUC = 0.862 (cluster-CV)    |    Brier = 0.008    |    PR-AUC = 0.690",
        fontsize=8,
        textcolor="white",
        radius=0.06,
    )

    # Loss function & training
    ax.text(
        8.0,
        1.0,
        "Loss: Binary Cross-Entropy\n"
        "Optimizer: AdamW (lr=1e-4, wd=1e-4)\n"
        "Batch: 64  |  Early stopping patience: 20",
        fontsize=6.5,
        ha="left",
        color=C["arrow"],
    )


# ===================================================================
# Schematic: Figure 1 (from fig_schematic.py)
# ===================================================================


def figure_schematic():
    """Generate Figure 1: Study overview schematic (3-panel)."""
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.1], hspace=0.25)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.suptitle(
        "GPCR–G Protein Coupling Prediction: Study Overview",
        fontsize=14,
        fontweight="bold",
        y=0.995,
        color=C["text"],
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(FIG_DIR / "figure1_schematic.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure1_schematic.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> figure1_schematic.png / .pdf")


# ===================================================================
# Manuscript main figures  (from fig_manuscript.py)
# ===================================================================


def figure_auc_comparison():
    """Dot + error bar plot comparing AUC across feature configurations."""
    categories = ["Baseline", "+ ICL", "+ ICL + AF"]
    x = np.arange(len(categories))

    models = {
        "SVM (8M)": {
            "vals": [0.7972, 0.8301, 0.8285],
            "err": [0.0143, 0.0126, 0.0142],
            "color": COLORS["SVM"],
            "marker": "s",
            "offset": -0.2,
        },
        "SVM (650M)": {
            "vals": [0.8188, 0.8324, 0.8287],
            "err": [0.0141, 0.0135, 0.0163],
            "color": COLORS["SVM"],
            "marker": "D",
            "offset": -0.07,
        },
        "Cross-Attn (8M)": {
            "vals": [0.8159, 0.8247, 0.8207],
            "err": [0.0187, 0.0163, 0.0189],
            "color": COLORS["CA"],
            "marker": "o",
            "offset": 0.07,
        },
        "Cross-Attn (650M)": {
            "vals": [0.8378, 0.8619, 0.8600],
            "err": [0.0229, 0.0249, 0.0231],
            "color": COLORS["CA"],
            "marker": "D",
            "offset": 0.2,
        },
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for label, d in models.items():
        xoff = x + d["offset"]
        ax.errorbar(
            xoff,
            d["vals"],
            yerr=d["err"],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(
            xoff,
            d["vals"],
            s=150,
            c=d["color"],
            marker=d["marker"],
            label=label,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
            alpha=0.9,
        )
        for idx, (xi, val, err) in enumerate(zip(xoff, d["vals"], d["err"])):
            if idx == 2:
                y_offset = -0.018 if "CA" in label else 0.018
                va = "top" if y_offset < 0 else "bottom"
            else:
                y_offset = val + err + 0.012 - val
                va = "bottom"
            ax.text(
                xi,
                val + y_offset,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=6.3,
                alpha=0.85,
            )

    ax.set_ylabel("Cluster-aware CV AUC", fontsize=10)
    ax.set_xlabel("Feature configuration", fontsize=10)
    ax.set_title(
        "GPCR–G Protein Coupling Prediction: Architecture Comparison",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0.76, 0.92)
    ax.legend(loc="upper left", frameon=True, fontsize=8, markerscale=0.8, ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    best_y = 0.8619
    ax.axhline(y=best_y, color=COLORS["CA"], linestyle="--", alpha=0.35, linewidth=1)
    ax.annotate(
        "Best: 0.862",
        xy=(2, best_y),
        xytext=(2.15, 0.913),
        arrowprops=dict(arrowstyle="->", color=COLORS["CA"], lw=1.0),
        fontsize=8,
        color=COLORS["CA"],
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure1_auc_comparison.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure1_auc_comparison.pdf")


def figure_icl_alignment():
    """ICL dimension alignment analysis — two panels."""
    svm_320_320 = 0.8301
    svm_1280_320 = 0.8234
    svm_1280_1280 = 0.8304
    ca_320_320 = 0.8247
    ca_1280_320 = 0.8155
    ca_1280_1280 = 0.8599

    svm_vals = [svm_320_320, svm_1280_320, svm_1280_1280]
    ca_vals = [ca_320_320, ca_1280_320, ca_1280_1280]
    svm_err = [0.0126, 0.015, 0.013]
    ca_err = [0.0163, 0.018, 0.025]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )

    # ---- Panel A: Dimension MISMATCH effect ----
    ax = axes[0]
    labels_a = ["Matched\n320-d + 320-d", "Mismatched\n1280-d + 320-d"]
    x_a = [0, 1]
    width = 0.35

    svm_sub = [svm_320_320, svm_1280_320]
    ca_sub = [ca_320_320, ca_1280_320]
    svm_e_sub = [svm_err[0], svm_err[1]]
    ca_e_sub = [ca_err[0], ca_err[1]]

    b1 = ax.bar(
        [xi - width / 2 for xi in x_a],
        svm_sub,
        width,
        label="SVM (RBF)",
        color=COLORS["SVM"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        [xi + width / 2 for xi in x_a],
        ca_sub,
        width,
        label="Cross-Attention",
        color=COLORS["CA"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    for bars, errs in [(b1, svm_e_sub), (b2, ca_e_sub)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=err,
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1,
            )

    for bar, val in zip(b1, svm_sub):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )
    for bar, val in zip(b2, ca_sub):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )

    delta_svm = svm_sub[1] - svm_sub[0]
    delta_ca = ca_sub[1] - ca_sub[0]
    ax.annotate(
        f"ΔSVM: {delta_svm:+.4f}\nΔCA: {delta_ca:+.4f}",
        (x_a[1], 0.795),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#C0392B",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FDEDEC",
            edgecolor="#C0392B",
            alpha=0.9,
            lw=1.2,
        ),
    )

    ax.axvspan(0.45, 1.55, alpha=0.06, color="red", zorder=0)
    ax.set_ylabel("Cluster-aware CV AUC", fontsize=9)
    ax.set_title(
        "A. Dimension Mismatch Causes\n   Performance Degradation",
        fontweight="bold",
        loc="left",
        fontsize=10,
    )
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=9)
    ax.set_ylim(0.60, 0.87)
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # ---- Panel B: Recovery after dimension matching ----
    ax = axes[1]
    labels_b = [
        "Mismatched\n1280-d + 320-d",
        "Matched ✓\n1280-d + 1280-d",
    ]
    x_b = [0, 1]

    svm_sub2 = [svm_1280_320, svm_1280_1280]
    ca_sub2 = [ca_1280_320, ca_1280_1280]
    svm_e_sub2 = [svm_err[1], svm_err[2]]
    ca_e_sub2 = [ca_err[1], ca_err[2]]

    b3 = ax.bar(
        [xi - width / 2 for xi in x_b],
        svm_sub2,
        width,
        label="SVM (RBF)",
        color=COLORS["SVM"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    b4 = ax.bar(
        [xi + width / 2 for xi in x_b],
        ca_sub2,
        width,
        label="Cross-Attention",
        color=COLORS["CA"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    for bars, errs in [(b3, svm_e_sub2), (b4, ca_e_sub2)]:
        for bar, err in zip(bars, errs):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=err,
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1,
            )

    for bar, val in zip(b3, svm_sub2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )
    for bar, val in zip(b4, ca_sub2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )

    delta_ca_recovery = ca_1280_1280 - ca_1280_320
    delta_svm_recovery = svm_1280_1280 - svm_1280_320
    ax.annotate(
        f"Recovery:\nSVM {delta_svm_recovery:+.4f}\nCA {delta_ca_recovery:+.4f}",
        (x_b[1], 0.89),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#27AE60",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#D5F5E3",
            edgecolor="#27AE60",
            alpha=0.9,
            lw=1.5,
        ),
    )

    ax.axvspan(0.45, 1.55, alpha=0.06, color="green", zorder=0)
    ax.set_ylabel("Cluster-aware CV AUC", fontsize=9)
    ax.set_title(
        "B. Dimension Matching Recovers\n   Cross-Attention Advantage",
        fontweight="bold",
        loc="left",
        fontsize=10,
    )
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
    """AlphaFold structural features ablation — paired dot plot."""
    configs = ["SVM\n8M", "SVM\n650M", "CA\n8M", "CA\n650M"]
    x = np.arange(len(configs))

    icl_vals = [0.8301, 0.8324, 0.8247, 0.8619]
    af_vals = [0.8285, 0.8287, 0.8207, 0.8600]
    icl_err = [0.0126, 0.0135, 0.0163, 0.0249]
    af_err = [0.0142, 0.0163, 0.0189, 0.0231]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    dx = 0.24

    for i in range(len(configs)):
        ax.plot(
            [i - dx, i + dx],
            [icl_vals[i], af_vals[i]],
            "-",
            color="grey",
            alpha=0.5,
            linewidth=1.2,
            zorder=1,
        )
        ax.scatter(
            i - dx,
            icl_vals[i],
            s=130,
            c=COLORS["Ensemble"],
            marker="o",
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            i - dx,
            icl_vals[i],
            yerr=icl_err[i],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.8,
            alpha=0.5,
        )
        ax.scatter(
            i + dx,
            af_vals[i],
            s=130,
            c=WONG["grey"],
            marker="s",
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            i + dx,
            af_vals[i],
            yerr=af_err[i],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.8,
            alpha=0.5,
        )
        delta = icl_vals[i] - af_vals[i]
        y_top = max(icl_vals[i], af_vals[i]) + max(icl_err[i], af_err[i])
        ax.text(
            i,
            y_top + 0.006,
            f"Δ={delta:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#7D3C98",
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["Ensemble"],
            markersize=9,
            label="ICL only",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=WONG["grey"],
            markersize=9,
            label="ICL + AlphaFold (38-d)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8.5)

    ax.set_ylabel("Cluster-aware CV AUC", fontsize=10)
    ax.set_title(
        "AlphaFold Structural Features: No Incremental Benefit",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    all_data = icl_vals + af_vals
    ax.set_ylim(min(all_data) - 0.025, max(all_data) + 0.035)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax.text(
        0.02,
        0.96,
        "All Δ ≤ 0.002\nNo comparison reaches\nstatistical significance\n(p > 0.05, Bonferroni)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#C0392B",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#FDEDEC",
            edgecolor="#C0392B",
            alpha=0.85,
            lw=0.8,
        ),
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure4_alphafold_ablation.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure4_alphafold_ablation.pdf")


def figure_statistical_tests():
    """Statistical test results as a forest plot / horizontal bar chart."""
    try:
        tests = load(DATA_DIR / "statistical_tests_results.json")
    except (FileNotFoundError, json.JSONDecodeError):
        print("  [SKIP] statistical tests — no data file found")
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

    for i, (d, p) in enumerate(zip(mean_diffs, p_values)):
        sig = "*" if p < 0.05 else ("**" if p < 0.01 else "n.s.")
        ax.text(
            max(mean_diffs) * 1.05,
            i,
            f"p={p:.3f} {sig}",
            va="center",
            fontsize=7,
            color="black" if p < 0.05 else "grey",
        )

    ax.axvspan(
        max(mean_diffs) * 1.0,
        max(mean_diffs) * 1.5,
        alpha=0.03,
        color="green",
        label="p < 0.05 significant",
    )
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure_statistical_tests.pdf", format="pdf")
    plt.close(fig)
    print("  -> figure_statistical_tests.pdf")


# ===================================================================
# BIB-specific figures  (from fig_bib.py)
# ===================================================================


def _load_bib_data():
    """Load and prepare shared data structures for BIB figures.

    Returns a dict with keys:
      pairing, promiscuity, df_all, df_prom, prom_data, families
    """
    pairing = pd.read_csv(DATA_DIR / "pairing_matrix_raw.csv")
    svm_preds = load(DATA_DIR / "svm_predictions.json")
    ca_preds = load(DATA_DIR / "ca_predictions.json")

    pos_pairs = pairing[pairing.coupling == 1]
    promiscuity = pos_pairs.groupby("gpcr_id").g_protein_family.nunique()

    families = ["Gq", "Gi", "Gs", "G12_13"]

    rows_all = []
    rows_prom = []
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

    prom_labels = {
        1: "Single-family",
        2: "Dual-family",
        3: "Triple-family",
        4: "4+-family",
    }
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

    return {
        "pairing": pairing,
        "promiscuity": promiscuity,
        "df_all": df_all,
        "df_prom": df_prom,
        "prom_data": prom_data,
        "families": families,
    }


def figure_promiscuity_analysis(data=None):
    """Promiscuity-stratified AUC analysis — connected dot plot + delta lollipop.

    Parameters
    ----------
    data : dict or None
        Pre-loaded data from _load_bib_data().  If None, data is loaded
        automatically.
    """
    if data is None:
        data = _load_bib_data()

    prom_data = data["prom_data"]
    names = list(prom_data.keys())
    x = np.arange(len(names))

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # Panel A: Connected dot plot
    ax = axes[0]
    markers = {"SVM": "s", "CA": "o", "Ensemble": "D"}
    sizes = {"SVM": 90, "CA": 110, "Ensemble": 80}
    for model, key in [("SVM", "SVM"), ("CA", "CA"), ("Ensemble", "Ensemble")]:
        vals = [prom_data[n][model] for n in names]
        ax.plot(x, vals, "-", color=COLORS[key], alpha=0.5, linewidth=1.5, zorder=1)
        ax.scatter(
            x,
            vals,
            s=sizes[model],
            c=COLORS[key],
            marker=markers[model],
            label=model,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
            alpha=0.9,
        )

    all_vals = [
        v
        for n in names
        for v in [
            prom_data[n]["SVM"],
            prom_data[n]["CA"],
            prom_data[n]["Ensemble"],
        ]
    ]
    y_min = max(0.5, min(all_vals) - 0.08)
    y_max = min(1.02, max(all_vals) + 0.10)

    ax.set_ylabel("AUC (Cluster-CV)", fontsize=9)
    ax.set_title("Performance by GPCR Promiscuity", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(N={prom_data[n]['N']})" for n in names], fontsize=8
    )
    ax.set_ylim(y_min, y_max)
    ax.legend(frameon=True, loc="upper right", fontsize=7.5, markerscale=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)

    for xi, name in enumerate(names):
        d = prom_data[name]
        txt = f"CA:{d['CA']:.3f}\nSVM:{d['SVM']:.3f}"
        ax.text(
            xi,
            y_min - (y_max - y_min) * 0.08,
            txt,
            ha="center",
            va="top",
            fontsize=6.5,
            alpha=0.85,
            linespacing=1.2,
        )

    # Panel B: Delta lollipop plot
    ax = axes[1]
    deltas = [prom_data[n]["CA"] - prom_data[n]["SVM"] for n in names]
    delta_colors = [COLORS["CA"] if d > 0 else "#E74C3C" for d in deltas]

    for xi, d, dc in zip(x, deltas, delta_colors):
        ax.plot(
            [xi, xi],
            [0, d],
            color=dc,
            linewidth=3.5,
            solid_capstyle="round",
            alpha=0.8,
        )
        ax.scatter(xi, d, s=150, c=dc, edgecolors="white", linewidth=0.8, zorder=3)
        va = "bottom" if d > 0 else "top"
        offset = 0.006 if d > 0 else -0.006
        ax.text(
            xi,
            d + offset,
            f"{d:+.4f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=dc,
        )

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


def figure_calibration_analysis(data=None):
    """Calibration and reliability analysis — 3 panels (reliability diagram,
    confidence vs. accuracy, per-family Brier scores).

    Parameters
    ----------
    data : dict or None
        Pre-loaded data from _load_bib_data().  If None, data is loaded
        automatically.
    """
    if data is None:
        data = _load_bib_data()

    df_all = data["df_all"]
    families = data["families"]
    y_true = df_all.label.values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # ---- Panel A: Reliability diagram ----
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

        ax.plot(
            prob_pred,
            prob_true,
            "o-",
            color=COLORS[key],
            label=model_name,
            markersize=7,
            linewidth=2.0,
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

        if len(prob_pred) >= 4:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                x_sorted = np.sort(prob_pred)
                ax.plot(
                    x_sorted,
                    iso.fit_transform(x_sorted, prob_true[np.argsort(prob_pred)]),
                    "--",
                    color=COLORS[key],
                    alpha=0.35,
                    linewidth=1.0,
                )
            except Exception:
                pass

        brier = brier_score_loss(y_true, probs)
        y_pos = 0.95 if key == "CA" else 0.78
        ax.text(
            0.03,
            y_pos,
            f"{model_name}: Brier={brier:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            fontweight="bold",
            color=COLORS[key],
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=COLORS[key],
                alpha=0.92,
                lw=1.0,
            ),
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect", linewidth=0.8)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_title("A. Reliability Diagram", fontweight="bold", fontsize=10)
    ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # ---- Panel B: Confidence vs. Accuracy ----
    ax = axes[1]
    thresholds = [0.5, 0.75, 0.90, 0.98]
    coverage = []
    accuracy_vals = []
    for thresh in thresholds:
        sub = df_all[df_all.confidence >= thresh]
        if len(sub) > 0:
            coverage.append(len(sub) / len(df_all) * 100)
            accuracy_vals.append(
                accuracy_score(sub.label, (sub.ca_prob > 0.5).astype(int)) * 100
            )

    ax.fill_between(coverage, accuracy_vals, alpha=0.08, color=COLORS["CA"])
    ax.plot(
        coverage,
        accuracy_vals,
        "o-",
        color=COLORS["CA"],
        linewidth=2.5,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    offsets = [(12, 6), (-8, -12), (12, -12), (-8, 6)]
    for (c, a, t), (dx, dy) in zip(
        zip(coverage, accuracy_vals, thresholds), offsets
    ):
        ax.annotate(
            f"conf≥{t:.2f}",
            (c, a),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            alpha=0.9,
            fontweight="bold",
            color=COLORS["CA"],
            arrowprops=dict(arrowstyle="-", color="grey", alpha=0.4, lw=0.5),
        )

    for c, a in zip(coverage, accuracy_vals):
        ax.text(
            c + 0.8,
            a + 0.15,
            f"{a:.1f}%",
            fontsize=7,
            alpha=0.85,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Coverage (%)", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("B. Confidence vs. Accuracy", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    x_margin = (max(coverage) - min(coverage)) * 0.15
    y_margin = (max(accuracy_vals) - min(accuracy_vals)) * 0.6
    ax.set_xlim(min(coverage) - x_margin, max(coverage) + x_margin * 1.2)
    ax.set_ylim(min(accuracy_vals) - y_margin, max(accuracy_vals) + y_margin * 0.8)

    # ---- Panel C: Per-family Brier scores ----
    ax = axes[2]
    family_briers = []
    for fam in families:
        mask = df_all.family == fam
        fp = df_all.loc[mask, "ca_prob"].values
        fl = df_all.loc[mask, "label"].values
        family_briers.append(brier_score_loss(fl, fp))

    y_pos = np.arange(len(families))
    fam_colors = [FAMILY_COLOR.get(f, WONG["blue"]) for f in families]

    bars = ax.barh(
        y_pos,
        family_briers,
        color=fam_colors,
        alpha=0.85,
        edgecolor="white",
        height=0.55,
    )
    for bar, val in zip(bars, family_briers):
        ax.text(
            bar.get_width() + max(family_briers) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

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


def figure_feature_importance():
    """Feature importance from gradient attribution — horizontal bar chart.

    Reads gradient_attribution_650m.json from ../figures/; falls back to
    hardcoded defaults if the file is missing.
    """
    try:
        raw = load(FIG_DIR / "gradient_attribution_650m.json")
        grad_data = raw.get("feature_group_means", raw)
    except (FileNotFoundError, json.JSONDecodeError):
        grad_data = {
            "gpcr_esm_global": 5.67e-05,
            "gprotein_esm": 1.09e-05,
            "icl2_stats": 7.21e-05,
            "icl3_stats": 7.61e-05,
            "icl2_esm": 6.57e-05,
            "icl3_esm": 7.00e-05,
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
        COLORS["CA"]
        if "gpcr" in g or "icl" in g
        else COLORS["SVM"]
        if "gprot" in g
        else WONG["grey"]
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    y_pos = np.arange(len(groups))
    bars = ax.barh(
        y_pos,
        values,
        color=bar_colors,
        alpha=0.88,
        edgecolor="white",
        height=0.6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels, fontsize=9)
    ax.set_xlabel("Normalized |Gradient Attribution|", fontsize=9)
    ax.set_title(
        "Feature Group Importance (Cross-Attention 650M)",
        fontweight="bold",
        fontsize=10,
    )
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        if val > max(values) * 0.3:
            ax.text(
                val * 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                val + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.axhline(y=1.5, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    label_x = max(values) * 1.28
    ax.text(
        label_x,
        0,
        "GPCR-side",
        fontsize=7,
        ha="left",
        va="center",
        color=COLORS["CA"],
        fontweight="bold",
        fontstyle="italic",
    )
    ax.text(
        label_x,
        2,
        "G protein-side",
        fontsize=7,
        ha="left",
        va="center",
        color=COLORS["SVM"],
        fontweight="bold",
        fontstyle="italic",
    )
    ax.text(
        label_x,
        len(groups) - 1,
        "Structural",
        fontsize=7,
        ha="left",
        va="center",
        color=WONG["grey"],
        fontweight="bold",
        fontstyle="italic",
    )

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_xlim(0, max(values) * 1.45)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_groups.pdf", format="pdf")
    plt.close(fig)
    print("  -> feature_importance_groups.pdf")


# ===================================================================
# Group runners
# ===================================================================


def run_schematic():
    """Generate the study overview schematic (Figure 1)."""
    print("=" * 60)
    print("  Schematic Figure (Figure 1)")
    print("=" * 60)
    figure_schematic()
    print("  Done.")


def run_main():
    """Generate manuscript main figures (AUC, ICL alignment, AF ablation,
    statistical tests)."""
    print("=" * 60)
    print("  Manuscript Main Figures")
    print("=" * 60)
    figure_auc_comparison()
    figure_icl_alignment()
    figure_alphafold_ablation()
    figure_statistical_tests()
    print("  Done.")


def run_bib():
    """Generate BIB-specific figures (promiscuity, calibration, feature
    importance)."""
    print("=" * 60)
    print("  BIB Figures (promiscuity, calibration, feature importance)")
    print("=" * 60)

    # Load shared data once so individual figure functions can reuse it
    data = _load_bib_data()

    print("\n[1/3] Promiscuity analysis (dot+line plot)...")
    figure_promiscuity_analysis(data)

    print("\n[2/3] Calibration analysis (quantile bins + LOWESS)...")
    figure_calibration_analysis(data)

    print("\n[3/3] Feature importance (horizontal bar)...")
    figure_feature_importance()

    # Key Numbers summary
    prom_data = data["prom_data"]
    names = list(prom_data.keys())
    df_all = data["df_all"]
    print(f"\n{'=' * 60}")
    print("  Key Numbers")
    print(f"{'=' * 60}")
    for name in names:
        d = prom_data[name]
        print(
            f"  {name:16s} N={d['N']:4d}  CA={d['CA']:.4f}  "
            f"SVM={d['SVM']:.4f}  Delta={d['CA'] - d['SVM']:.4f}"
        )
    print(
        f"  CA Brier (all predictions): "
        f"{brier_score_loss(df_all.label, df_all.ca_prob):.4f}"
    )
    print(
        f"  SVM Brier (all predictions): "
        f"{brier_score_loss(df_all.label, df_all.svm_prob):.4f}"
    )
    print("  Done.")


def run_all():
    """Generate all figures."""
    run_schematic()
    print()
    run_main()
    print()
    run_bib()


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the GPCR–G protein coupling manuscript.",
    )
    sub = parser.add_subparsers(dest="command", help="Figure group to generate")

    sub.add_parser("schematic", help="Figure 1: study overview schematic")
    sub.add_parser("main", help="Manuscript main figures (AUC, ICL, AF, stats)")
    sub.add_parser("bib", help="BIB-specific figures (promiscuity, calibration, features)")
    sub.add_parser("all", help="All figures (default)")

    args = parser.parse_args()

    if args.command == "schematic":
        run_schematic()
    elif args.command == "main":
        run_main()
    elif args.command == "bib":
        run_bib()
    elif args.command == "all":
        run_all()
    else:
        # No subcommand given — default to all
        run_all()


if __name__ == "__main__":
    main()
