#!/usr/bin/env python3
"""
Generate Figure 1: Schematic overview of the study design, feature engineering,
and cross-attention architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Color palette (consistent with other figures)
COLORS = {
    "gpcr": "#3498db",
    "gprot": "#e74c3c",
    "icl": "#2ecc71",
    "alpha": "#f39c12",
    "esm": "#9b59b6",
    "box_bg": "#ecf0f1",
    "arrow": "#2c3e50",
    "text": "#2c3e50",
}


def draw_box(ax, xy, width, height, text, color, fontsize=9, text_color="white", radius=0.02):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor="black", linewidth=1.2, zorder=2
    )
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=3)


def draw_arrow(ax, start, end, color=COLORS["arrow"]):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=1)


def panel_a(ax):
    """Data curation and evaluation framework."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Dataset curation and evaluation", fontweight="bold", fontsize=13, loc="left", pad=10)

    # GPCRdb box
    draw_box(ax, (0.5, 7.0), 2.0, 0.8, "GPCRdb\nannotations", COLORS["gpcr"], fontsize=8)
    # Literature box
    draw_box(ax, (0.5, 5.5), 2.0, 0.8, "Literature\ncuration", "#1abc9c", fontsize=8)
    # Arrow to pairing matrix
    draw_arrow(ax, (1.5, 7.0), (3.0, 6.5))
    draw_arrow(ax, (1.5, 6.3), (3.0, 6.5))

    # 1,639 pairs
    draw_box(ax, (3.0, 6.0), 2.2, 1.0, "1,639 pairs\n(431 GPCRs)", COLORS["box_bg"], text_color=COLORS["text"], fontsize=9)

    # Arrow to families
    draw_arrow(ax, (5.2, 6.5), (6.2, 6.5))

    # Families
    fam_y = 7.8
    fam_colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    fam_names = ["Gq\n(388)", "Gi\n(406)", "Gs\n(298)", "G12/13\n(173)"]
    for i, (name, c) in enumerate(zip(fam_names, fam_colors)):
        draw_box(ax, (6.4 + i*0.85, fam_y - (i%2)*1.2), 0.75, 0.9, name, c, fontsize=7)
        ax.annotate("", xy=(6.8 + i*0.85, fam_y + 1.0 if i%2==0 else fam_y - 0.4),
                    xytext=(7.1, 6.5),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                    zorder=1)

    # CD-HIT clustering
    draw_box(ax, (3.0, 4.0), 2.2, 0.8, "CD-HIT 40%\n387 clusters", "#95a5a6", fontsize=8)
    draw_arrow(ax, (4.1, 6.0), (4.1, 4.8))

    # CV strategies
    draw_box(ax, (6.0, 4.0), 1.8, 0.8, "Cluster-aware\n5-fold CV", COLORS["box_bg"], text_color=COLORS["text"], fontsize=8)
    draw_box(ax, (8.0, 4.0), 1.6, 0.8, "LOGPSO\n(4 families)", COLORS["box_bg"], text_color=COLORS["text"], fontsize=8)
    draw_arrow(ax, (5.2, 4.4), (6.0, 4.4))
    draw_arrow(ax, (5.2, 4.4), (8.0, 4.4))


def panel_b(ax):
    """Feature engineering pipeline."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("B. Feature engineering", fontweight="bold", fontsize=13, loc="left", pad=10)

    # GPCR global ESM
    draw_box(ax, (0.3, 6.5), 1.8, 1.0, "GPCR ESM-2\n650M (1280-d)", COLORS["esm"], fontsize=8)
    # G-protein global ESM
    draw_box(ax, (0.3, 4.8), 1.8, 1.0, "G-protein ESM-2\n8M (320-d)", COLORS["gprot"], fontsize=8)

    # Concatenation
    ax.annotate("", xy=(2.5, 6.0), xytext=(2.1, 7.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5))
    ax.annotate("", xy=(2.5, 6.0), xytext=(2.1, 5.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5))
    ax.text(2.3, 6.5, "+", fontsize=14, ha="center", va="center", fontweight="bold")

    draw_box(ax, (2.5, 5.5), 1.4, 1.0, "Global\nconcat\n(1600-d)", COLORS["box_bg"], text_color=COLORS["text"], fontsize=8)

    # ICL features
    draw_box(ax, (4.3, 7.2), 1.6, 0.8, "ICL2 ESM\n(1280-d)", COLORS["icl"], fontsize=8)
    draw_box(ax, (4.3, 6.1), 1.6, 0.8, "ICL2 stats\n(8-d)", "#27ae60", fontsize=8)
    draw_box(ax, (4.3, 5.0), 1.6, 0.8, "ICL3 ESM\n(1280-d)", COLORS["icl"], fontsize=8)
    draw_box(ax, (4.3, 3.9), 1.6, 0.8, "ICL3 stats\n(8-d)", "#27ae60", fontsize=8)

    # ICL concat
    draw_box(ax, (6.2, 5.5), 1.2, 1.0, "ICL\nconcat\n(2576-d)", COLORS["box_bg"], text_color=COLORS["text"], fontsize=8)
    for y_src in [7.6, 6.5, 5.4, 4.3]:
        ax.annotate("", xy=(6.2, 6.3), xytext=(5.9, y_src),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Global + ICL
    ax.annotate("", xy=(7.7, 6.3), xytext=(3.9, 6.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5))
    draw_box(ax, (7.7, 5.8), 1.6, 1.0, "ICL-full\n(4176-d)", COLORS["box_bg"], text_color=COLORS["text"], fontsize=9)

    # AlphaFold
    draw_box(ax, (7.7, 4.0), 1.6, 1.0, "AlphaFold\n(38-d)", COLORS["alpha"], fontsize=8)
    ax.annotate("", xy=(8.5, 5.8), xytext=(8.5, 5.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, ls="--"))
    ax.text(8.8, 5.4, "optional", fontsize=7, color="gray", style="italic")


def panel_c(ax):
    """Cross-attention architecture schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C. Cross-attention architecture", fontweight="bold", fontsize=13, loc="left", pad=10)

    # Input boxes
    draw_box(ax, (0.5, 6.5), 1.6, 0.8, "GPCR feat\n(d_GPCR)", COLORS["gpcr"], fontsize=8)
    draw_box(ax, (0.5, 4.5), 1.6, 0.8, "G-protein feat\n(d_Gprot)", COLORS["gprot"], fontsize=8)

    # Projection layers
    draw_box(ax, (2.8, 6.5), 1.4, 0.8, "Linear +\nLayerNorm +\nGELU", "#5dade2", fontsize=7)
    draw_box(ax, (2.8, 4.5), 1.4, 0.8, "Linear +\nLayerNorm +\nGELU", "#ec7063", fontsize=7)

    draw_arrow(ax, (2.1, 7.0), (2.8, 7.0))
    draw_arrow(ax, (2.1, 4.9), (2.8, 4.9))

    # Cross-attention circle
    circle = Circle((5.5, 6.0), 0.7, facecolor="#f4d03f", edgecolor="black", linewidth=1.5, zorder=2)
    ax.add_patch(circle)
    ax.text(5.5, 6.0, "Cross-\nAttn", ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

    # Arrows into attention
    ax.annotate("", xy=(4.8, 6.3), xytext=(4.2, 6.9),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2))
    ax.text(4.3, 6.8, "q", fontsize=8, color=COLORS["gpcr"], fontweight="bold")

    ax.annotate("", xy=(4.8, 5.7), xytext=(4.2, 5.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2))
    ax.text(4.3, 5.3, "k,v", fontsize=8, color=COLORS["gprot"], fontweight="bold")

    # Concatenation and FFN
    draw_box(ax, (6.6, 6.2), 1.4, 0.8, "Concat +\nFFN", "#aab7b8", fontsize=8)
    ax.annotate("", xy=(6.6, 6.6), xytext=(6.2, 6.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2))
    # Skip connection from GPCR projection
    ax.annotate("", xy=(6.6, 6.4), xytext=(4.2, 6.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, connectionstyle="arc3,rad=0.3"))

    # Output
    draw_box(ax, (8.4, 6.4), 1.2, 0.6, "Sigmoid", COLORS["box_bg"], text_color=COLORS["text"], fontsize=8)
    draw_arrow(ax, (8.0, 6.6), (8.4, 6.7))

    # Output label
    ax.text(9.6, 6.7, "P(coupling)", fontsize=9, ha="left", va="center", fontweight="bold")

    # Architecture parameters box
    draw_box(ax, (6.2, 3.8), 3.2, 1.2, "Hidden dim = 256\nNum heads = 4\nDropout = 0.3",
             "#fdfefe", text_color=COLORS["text"], fontsize=8, radius=0.05)
    ax.text(7.8, 5.2, "Architecture hyperparameters", fontsize=9, ha="center", va="bottom",
            fontweight="bold", color=COLORS["text"])


def main():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_schematic.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure1_schematic.pdf", bbox_inches="tight")
    print(f"[OK] Saved schematic figure to {FIG_DIR / 'figure1_schematic.png'}")


if __name__ == "__main__":
    main()
