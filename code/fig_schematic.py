#!/usr/bin/env python3
"""
Figure 1: Professional schematic overview — data pipeline, feature engineering,
and cross-attention architecture. Publication-quality for Briefings in Bioinformatics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path as PathLib

from style_config import apply_style, WONG

apply_style()

BASE = PathLib(__file__).parent
FIG_DIR = BASE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────
C = {
    "gpcr":       "#2471A3",   # deep blue
    "gpcr_light": "#D4E6F1",
    "gprot":      "#D35400",   # deep orange
    "gprot_light":"#FAD7A1",
    "icl":        "#1E8449",   # deep green
    "icl_light":  "#D5F5E3",
    "af":         "#95A5A6",   # grey
    "af_light":   "#E5E7E9",
    "nn":         "#7D3C98",   # purple for neural
    "nn_light":   "#E8DAEF",
    "panel_bg":   "#F8F9FA",
    "border":     "#BDC3C7",
    "arrow":      "#5D6D7E",
    "text":       "#2C3E50",
    "white":      "#FFFFFF",
    "gold":       "#F4D03F",
    "red_accent": "#C0392B",
    "family":     ["#3498DB", "#E74C3C", "#F39C12", "#9B59B6"],
    "cross_attn": "#E67E22",
}

FAMILY_NAMES = ["Gq/11\n(n=?)", "Gi/o\n(n=?)", "Gs\n(n=?)", "G12/13\n(n=?)"]

# ── Drawing primitives ─────────────────────────────────────────────

def rounded_box(ax, xy, w, h, facecolor, edgecolor=None, lw=1.2, radius=0.08,
                text="", fontsize=8, textcolor="white", fontweight="bold", zorder=3):
    """Draw a rounded box with optional text."""
    if edgecolor is None:
        edgecolor = C["border"]
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=fontweight, zorder=zorder+1)


def text_box(ax, xy, w, h, text, facecolor=C["panel_bg"], edgecolor=C["border"],
             fontsize=8, textcolor=C["text"], fontweight="normal", lw=1.0, radius=0.06, zorder=3):
    """Draw a light-background box with dark text."""
    rounded_box(ax, xy, w, h, facecolor, edgecolor, lw, radius,
                text, fontsize, textcolor, fontweight, zorder)


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.5, style="->", zorder=1, ls="-"):
    """Draw a simple arrow."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls),
                zorder=zorder)


def curved_arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.2, rad=0.2, zorder=1):
    """Draw a curved arrow using FancyArrowPatch."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)


def section_label(ax, x, y, letter, title):
    """Draw a section label like 'A. Dataset Curation'."""
    ax.text(x, y, f"{letter}. {title}", fontsize=11, fontweight="bold",
            color=C["text"], ha="left", va="center")


def plus_mark(ax, x, y, fontsize=12, color=C["arrow"]):
    ax.text(x, y, "+", fontsize=fontsize, ha="center", va="center",
            fontweight="bold", color=color)


def cross_mark(ax, x, y, fontsize=12):
    ax.text(x, y, "✗", fontsize=fontsize, ha="center", va="center",
            fontweight="bold", color=C["red_accent"])


# ── Panel A: Data Pipeline ─────────────────────────────────────────

def panel_a(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Background
    bg = FancyBboxPatch((0.05, 0.1), 15.9, 5.8,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor="#EBF5FB", edgecolor="none", zorder=0, alpha=0.5)
    ax.add_patch(bg)
    section_label(ax, 0.3, 5.5, "A", "Dataset Curation & Evaluation Framework")

    # ── Row 1: Data sources ──
    y1 = 4.0
    rounded_box(ax, (0.5, y1), 2.0, 0.9, C["gpcr"], C["gpcr"],
                text="GPCRdb\nAnnotations", fontsize=7.5, textcolor="white")
    rounded_box(ax, (2.8, y1), 2.0, 0.9, C["gpcr"], C["gpcr"],
                text="Literature\nCuration", fontsize=7.5, textcolor="white")
    arrow(ax, 2.5, y1+0.45, 2.8, y1+0.45)
    arrow(ax, 3.8, y1+0.45, 5.5, 2.7, color=C["arrow"], lw=1.2)

    # ── Pairing matrix ──
    rounded_box(ax, (5.5, 2.2), 2.8, 1.0, C["panel_bg"], C["border"],
                text="1,647 pairs\n(431 GPCRs × 4 families)", fontsize=8.5,
                textcolor=C["text"], fontweight="bold")

    # ── G protein families ──
    fam_y_base = 4.2
    fam_counts = [388, 406, 298, 173]  # from data
    for i, (name, count, fc) in enumerate(zip(
        ["Gq/11", "Gi/o", "Gs", "G12/13"],
        fam_counts,
        C["family"]
    )):
        xf = 8.8 + i * 1.65
        rounded_box(ax, (xf, fam_y_base - (i%2)*1.4), 1.35, 0.85,
                    fc, fc, text=f"{name}\n(n={count})", fontsize=7, textcolor="white",
                    radius=0.06)
        arrow(ax, 8.3, 2.7, xf+0.675, fam_y_base+0.05 - (i%2)*1.4,
              color=C["arrow"], lw=0.9)

    # ── CD-HIT clustering ──
    rounded_box(ax, (5.5, 0.7), 2.8, 0.85, "#7F8C8D", "#7F8C8D",
                text="CD-HIT (40% identity)\n387 sequence clusters", fontsize=7.5,
                textcolor="white")
    arrow(ax, 6.9, 2.2, 6.9, 1.55)

    # ── Evaluation strategies ──
    eval_y = 0.7
    text_box(ax, (9.0, eval_y), 3.2, 0.85,
             "Cluster-aware 5-fold CV\n(no cluster leakage)", fontsize=7.5, textcolor=C["text"])
    text_box(ax, (12.5, eval_y), 3.0, 0.85,
             "LOGPSO\n(4 × leave-one-family-out)", fontsize=7.5, textcolor=C["text"])
    arrow(ax, 8.3, 1.125, 9.0, 1.125, color=C["arrow"], lw=1.0)
    arrow(ax, 8.3, 1.125, 12.5, 1.125, color=C["arrow"], lw=1.0)

    # Key metrics callout
    ax.text(0.5, 0.15, "◆ Primary metric: Cluster-aware AUC  ◆  Secondary: Brier score, LOGPSO AUC",
            fontsize=7, color=C["arrow"], ha="left", fontstyle="italic")


# ── Panel B: Feature Engineering ───────────────────────────────────

def panel_b(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    bg = FancyBboxPatch((0.05, 0.1), 15.9, 6.8,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor="#FDF2E9", edgecolor="none", zorder=0, alpha=0.5)
    ax.add_patch(bg)
    section_label(ax, 0.3, 6.5, "B", "Feature Engineering Pipeline")

    # ── Column headers ──
    col_centers = [2.5, 7.0, 11.5]
    col_labels = ["Global Embeddings", "ICL Topology Features", "Structural Features"]
    col_colors = [C["gpcr_light"], C["icl_light"], C["af_light"]]
    for cx, label, cc in zip(col_centers, col_labels, col_colors):
        ax.text(cx, 5.8, label, fontsize=9, ha="center", fontweight="bold",
                color=C["text"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=cc, edgecolor=C["border"], alpha=0.9))

    # ── Column 1: Global ESM-2 ──
    cx1 = 2.5
    rounded_box(ax, (cx1-1.0, 4.6), 2.0, 0.75, C["gpcr"], C["gpcr"],
                text="GPCR ESM-2 650M\n(1280-d)", fontsize=7.5, textcolor="white")
    rounded_box(ax, (cx1-1.0, 3.5), 2.0, 0.75, C["gprot"], C["gprot"],
                text="G Protein ESM-2 8M\n(320-d)", fontsize=7.5, textcolor="white")
    arrow(ax, cx1, 4.6, cx1, 4.35, color=C["arrow"], lw=1.0)
    arrow(ax, cx1, 4.25, cx1+1.8, 4.25, color=C["arrow"], lw=1.0)

    # ── Column 2: ICL features ──
    cx2 = 7.0
    icl_items = [
        ("ICL2 ESM (1280-d)", C["icl"]),
        ("ICL2 PhysChem (8-d)", "#27AE60"),
        ("ICL3 ESM (1280-d)", C["icl"]),
        ("ICL3 PhysChem (8-d)", "#27AE60"),
    ]
    for j, (label, col) in enumerate(icl_items):
        y_icl = 5.1 - j * 0.65
        rounded_box(ax, (cx2-0.85, y_icl), 1.7, 0.5, col, col,
                    text=label, fontsize=6.5, textcolor="white", radius=0.04)

    # Dimension alignment highlight
    ax.annotate("KEY: ICL ESM dim\nmust match global\ndim (1280-d → 1280-d)",
                xy=(cx2+1.1, 2.5), fontsize=6.5, ha="center", color=C["red_accent"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC",
                          edgecolor=C["red_accent"], alpha=0.8, lw=0.8))

    # ── Column 3: AlphaFold (optional) ──
    cx3 = 11.5
    rounded_box(ax, (cx3-0.85, 4.8), 1.7, 0.75,
                "#BDC3C7", "#BDC3C7", text="AlphaFold2\n38 descriptors", fontsize=7,
                textcolor="white", radius=0.06)
    ax.text(cx3, 4.3, "(optional)", fontsize=7, ha="center", color=C["arrow"], fontstyle="italic")
    cross_mark(ax, cx3, 3.8)

    # ── Fusion section (bottom) ──
    fuse_y = 1.5
    # Coming from col 1 (global)
    arrow(ax, cx1, 3.5, cx1, fuse_y+1.0, color=C["arrow"], lw=1.2)
    # Coming from col 2 (ICL)
    arrow(ax, cx2, 2.4, cx2, fuse_y+1.0, color=C["arrow"], lw=1.2)
    # Coming from col 3 (AF, dashed)
    arrow(ax, cx3, 3.8, cx3, fuse_y+1.0, color=C["arrow"], lw=1.0, ls="--")

    # Fusion box
    plus_mark(ax, 4.5, fuse_y+0.6, fontsize=16)
    plus_mark(ax, 6.5, fuse_y+0.6, fontsize=16)

    text_box(ax, (3.0, fuse_y), 5.0, 0.9,
             "Full Feature Vector (4176-d)\n"
             "GPCR: ESM(1280) + ICL2(1280+8) + ICL3(1280+8)  ∥  G protein: ESM(320)",
             fontsize=7, textcolor=C["text"], fontweight="bold")

    arrow(ax, 8.0, fuse_y+0.45, 9.5, fuse_y+0.45, color=C["nn"], lw=1.8)
    ax.text(8.75, fuse_y+0.75, "→ Panel C", fontsize=7, ha="center", color=C["nn"],
            fontweight="bold")

    # G protein subset
    arrow(ax, cx1, 3.5, cx1, 2.8, color=C["arrow"], lw=1.0)
    ax.text(cx1-1.3, 3.0, "G protein:\nESM only", fontsize=6.5, ha="center", color=C["gprot"])


# ── Panel C: Cross-Attention Architecture ──────────────────────────

def panel_c(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    bg = FancyBboxPatch((0.05, 0.1), 15.9, 6.8,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor="#F4ECF7", edgecolor="none", zorder=0, alpha=0.5)
    ax.add_patch(bg)
    section_label(ax, 0.3, 6.5, "C", "Cross-Attention Neural Network")

    # ── Input boxes ──
    inp_y = 4.5
    rounded_box(ax, (0.5, inp_y+0.4), 2.0, 0.9, C["gpcr"], C["gpcr"],
                text="GPCR Features\n(d_GPCR = 4176)", fontsize=7.5, textcolor="white")
    rounded_box(ax, (0.5, inp_y-1.2), 2.0, 0.9, C["gprot"], C["gprot"],
                text="G Protein Features\n(d_Gprot = 320)", fontsize=7.5, textcolor="white")

    # ── Projection layers ──
    proj_x = 3.2
    rounded_box(ax, (proj_x, inp_y+0.4), 1.8, 0.9, "#5DADE2", "#5DADE2",
                text="Linear + LN\n+ GELU → 256-d", fontsize=7, textcolor="white")
    rounded_box(ax, (proj_x, inp_y-1.2), 1.8, 0.9, "#F1948A", "#F1948A",
                text="Linear + LN\n+ GELU → 256-d", fontsize=7, textcolor="white")
    arrow(ax, 2.5, inp_y+0.85, proj_x, inp_y+0.85, color=C["arrow"], lw=1.3)
    arrow(ax, 2.5, inp_y-0.75, proj_x, inp_y-0.75, color=C["arrow"], lw=1.3)

    # ── Cross-Attention block ──
    attn_cx, attn_cy = 6.5, inp_y+0.1
    # Main attention circle
    circle = mpatches.Ellipse((attn_cx, attn_cy), 1.8, 1.8,
                              facecolor="#F9E79F", edgecolor=C["cross_attn"],
                              linewidth=2.0, zorder=4)
    ax.add_patch(circle)
    ax.text(attn_cx, attn_cy+0.35, "Multi-Head", fontsize=8, ha="center",
            va="center", fontweight="bold", color=C["text"], zorder=5)
    ax.text(attn_cx, attn_cy-0.15, "Cross-Attention", fontsize=8.5, ha="center",
            va="center", fontweight="bold", color=C["cross_attn"], zorder=5)
    ax.text(attn_cx, attn_cy-0.6, "4 heads × 64-d", fontsize=6.5, ha="center",
            va="center", color=C["arrow"], zorder=5)

    # Arrows into attention
    ax.annotate("Q", xy=(attn_cx-0.5, attn_cy+0.4), fontsize=9, fontweight="bold",
                color=C["gpcr"], ha="center", va="center")
    ax.annotate("K,V", xy=(attn_cx-0.5, attn_cy-0.5), fontsize=8, fontweight="bold",
                color=C["gprot"], ha="center", va="center")
    arrow(ax, proj_x+1.8, inp_y+0.85, attn_cx-0.95, attn_cy+0.4,
          color=C["gpcr"], lw=1.5)
    arrow(ax, proj_x+1.8, inp_y-0.75, attn_cx-0.95, attn_cy-0.4,
          color=C["gprot"], lw=1.5)

    # ── Post-attention processing ──
    post_x = attn_cx + 1.5
    rounded_box(ax, (post_x, attn_cy-0.4), 1.6, 0.8, C["nn_light"], C["nn"],
                text="Concat\n+ Skip", fontsize=7.5, textcolor=C["text"],
                fontweight="bold", lw=1.3)

    # Skip connection (curved, from GPCR projection to concat)
    curved_arrow(ax, proj_x+0.9, inp_y+0.85, post_x+0.2, attn_cy+0.3,
                 color=C["arrow"], lw=1.0, rad=0.3)

    # FFN
    ffn_x = post_x + 2.2
    rounded_box(ax, (ffn_x, attn_cy-0.4), 1.6, 0.8, C["nn"], C["nn"],
                text="3-Layer FFN\nGELU + LN\nDropout 0.3", fontsize=7,
                textcolor="white")
    arrow(ax, post_x+1.6, attn_cy, ffn_x, attn_cy, color=C["arrow"], lw=1.3)

    # ── Output ──
    out_x = ffn_x + 2.0
    rounded_box(ax, (out_x, attn_cy-0.3), 1.4, 0.6, C["cross_attn"], C["cross_attn"],
                text="Sigmoid", fontsize=8, textcolor="white")
    arrow(ax, ffn_x+1.6, attn_cy, out_x, attn_cy, color=C["arrow"], lw=1.3)

    # Final output
    ax.text(out_x+1.6, attn_cy-0.05, "P(coupling)\n∈ [0, 1]", fontsize=8.5,
            ha="center", va="center", fontweight="bold", color=C["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C["cross_attn"], lw=1.2))

    # ── Key metrics callout ──
    rounded_box(ax, (0.5, 0.5), 6.5, 1.0, "#2C3E50", "#2C3E50",
                text="AUC = 0.862 (cluster-CV)    |    Brier = 0.008    |    PR-AUC = 0.690",
                fontsize=8, textcolor="white", radius=0.06)

    # ── Loss function & training ──
    ax.text(8.0, 1.0, "Loss: Binary Cross-Entropy\n"
            "Optimizer: AdamW (lr=1e-4, wd=1e-4)\n"
            "Batch: 64  |  Early stopping patience: 20",
            fontsize=6.5, ha="left", color=C["arrow"])


# ── Main ───────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.1], hspace=0.25)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    # Global title
    fig.suptitle("GPCR–G Protein Coupling Prediction: Study Overview",
                 fontsize=14, fontweight="bold", y=0.995, color=C["text"])

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(FIG_DIR / "figure1_schematic.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure1_schematic.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure1_schematic.png / .pdf")


if __name__ == "__main__":
    main()
