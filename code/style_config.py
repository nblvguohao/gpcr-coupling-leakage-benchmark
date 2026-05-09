"""Shared style configuration for all BIB figures — Wong 2011 colorblind-safe palette."""

import matplotlib.pyplot as plt

# Wong 2011 colorblind-safe palette (Nature Methods)
WONG = {
    "blue":   "#0072B2",
    "orange": "#D55E00",
    "green":  "#009E73",
    "cyan":   "#56B4E9",
    "yellow": "#E69F00",
    "pink":   "#CC79A7",
    "grey":   "#95A5A6",
    "dark":   "#2C3E50",
}

# Semantic mapping
MODEL_COLOR = {
    "SVM":        WONG["blue"],
    "CA":         WONG["orange"],
    "MLP":        WONG["cyan"],
    "RF":         WONG["grey"],
    "XGBoost":    WONG["pink"],
    "Ensemble":   WONG["green"],
    "MultiTask":  WONG["yellow"],
}

FAMILY_COLOR = {
    "Gq":     WONG["yellow"],
    "Gi":     WONG["cyan"],
    "Gs":     WONG["orange"],
    "G12_13": WONG["pink"],
}

FEATURE_COLOR = {
    "gpcr":    WONG["blue"],
    "gprot":   WONG["orange"],
    "icl":     WONG["green"],
    "alpha":   WONG["yellow"],
    "esm":     WONG["cyan"],
}

STYLE = {
    "font.family":      "sans-serif",
    "font.size":        8,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}


def apply_style():
    """Apply shared BIB figure style. Call once at start of each script."""
    plt.rcParams.update(STYLE)
