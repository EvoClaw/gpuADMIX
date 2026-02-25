"""
plot_style.py — Venue-matched figure style for gpuADMIX paper.
Target venue: Bioinformatics (Oxford Academic) — follows Life Science / CNS profile.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Style (Life Science / Bioinformatics Oxford) ────────────────────────────
STYLE_CONFIG = {
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'lines.linewidth':   1.2,
    'lines.markersize':  5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,   # embed fonts in PDF
    'ps.fonttype':       42,
}

mpl.rcParams.update(STYLE_CONFIG)

# ── Color palette (Okabe-Ito colorblind-safe) ──────────────────────────────
COLORS = [
    '#56B4E9',   # 0 sky blue
    '#E69F00',   # 1 orange
    '#009E73',   # 2 green
    '#CC79A7',   # 3 pink/magenta
    '#D55E00',   # 4 vermillion
    '#0072B2',   # 5 deep blue
    '#F0E442',   # 6 yellow
    '#999999',   # 7 grey
]

# ── Method-color mapping (consistent across ALL figures) ──────────────────
METHOD_COLORS = {
    'gpuADMIX':     COLORS[0],   # sky blue  — always
    'fastmixture':  COLORS[1],   # orange    — always
    'ADMIXTURE':    COLORS[4],   # vermillion — always
    'Plain-EM':     COLORS[2],   # green
    'Nesterov':     COLORS[0],   # same as gpuADMIX (it IS gpuADMIX with Nesterov)
}

METHOD_MARKERS = {
    'gpuADMIX':     'o',
    'fastmixture':  's',
    'ADMIXTURE':    '^',
    'Plain-EM':     's',
    'Nesterov':     'o',
}

# Column widths for Bioinformatics Oxford (approx):
# Single column: 86mm = 3.39 in
# Double column: 178mm = 7.01 in
SINGLE_COL_WIDTH = 3.39
DOUBLE_COL_WIDTH = 7.01

def apply_style():
    """Call at the top of every plotting script."""
    mpl.rcParams.update(STYLE_CONFIG)

def panel_label(ax, label, x=-0.12, y=1.05, fontsize=9, fontweight='bold'):
    """Add bold panel label (a, b, c) to top-left of axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight, va='top', ha='left')
