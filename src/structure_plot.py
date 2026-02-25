"""
structure_plot.py — Publication-quality STRUCTURE / ADMIXTURE bar plots.

Produces:
  • Per-K individual bar chart
  • Multi-K stacked panel (like CLUMPAK output), population-sorted
  • Colour palette consistent across K values (component 0 always same colour)
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (colorblind-safe, up to K=10)
# ─────────────────────────────────────────────────────────────────────────────

# Based on Okabe-Ito (8 colours) + 2 extras; component index → colour
_PALETTE_HEX = [
    '#56B4E9',   # 0 sky blue    (EUR)
    '#E69F00',   # 1 orange      (EAS)
    '#009E73',   # 2 green       (AFR)
    '#F0E442',   # 3 yellow      (AMR)
    '#CC79A7',   # 4 pink/purple (SAS)
    '#D55E00',   # 5 vermillion
    '#0072B2',   # 6 blue
    '#999999',   # 7 grey
    '#44AA99',   # 8 teal
    '#AA4499',   # 9 purple
]

def _q_to_rgb(Q: np.ndarray, colours: List[str],
              height: int = 200) -> np.ndarray:
    """
    Convert (N, K) admixture matrix to (N, height, 3) RGB image using
    stacked-bar encoding. Uses imshow for fast rendering.
    """
    N, K = Q.shape
    img = np.ones((N, height, 3), dtype=np.float32)
    for i, col in enumerate(colours[:K]):
        rgb = np.array(to_rgb(col), dtype=np.float32)
        cum_low  = Q[:, :i].sum(axis=1)          # (N,)
        cum_high = cum_low + Q[:, i]              # (N,)
        for n in range(N):
            y0 = int(cum_low[n]  * height)
            y1 = max(y0 + 1, int(cum_high[n] * height))
            img[n, y0:y1] = rgb
    return img


def _palette(K: int) -> List[str]:
    """Return K distinct colours (consistent across K values)."""
    if K <= len(_PALETTE_HEX):
        return _PALETTE_HEX[:K]
    # extend with HSV-spaced colours
    import colorsys
    extra = []
    for i in range(K - len(_PALETTE_HEX)):
        h = (i / (K - len(_PALETTE_HEX)))
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
        extra.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
    return _PALETTE_HEX + extra


# ─────────────────────────────────────────────────────────────────────────────
# Single-K bar plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_k(Q: np.ndarray,
                  ax: plt.Axes,
                  pop_labels: Optional[np.ndarray] = None,
                  sort_by_pop: bool = True,
                  show_pop_dividers: bool = True,
                  show_xticks: bool = False,
                  title: str = '',
                  colours: Optional[List[str]] = None) -> None:
    """
    Draw a single STRUCTURE bar plot on ax.

    Parameters
    ----------
    Q          : (N, K) admixture matrix (rows sum to 1)
    ax         : matplotlib Axes to draw on
    pop_labels : (N,) array of population strings; if given, individuals are
                 sorted by population
    sort_by_pop: if True (default) sort individuals by population then major component
    show_pop_dividers: draw vertical lines between populations
    show_xticks: label individuals on x-axis (only useful for small N)
    title      : axes title string
    colours    : K-length list of colours; uses default palette if None
    """
    N, K = Q.shape
    cols = colours if colours is not None else _palette(K)

    # Determine sort order
    if sort_by_pop and pop_labels is not None:
        unique_pops = list(dict.fromkeys(pop_labels))   # preserve order
        pop_to_idx  = {p: i for i, p in enumerate(unique_pops)}
        pop_int = np.array([pop_to_idx[p] for p in pop_labels])
        # Within each pop, sort by dominant component (descending)
        dom_comp = Q.argmax(axis=1)
        order = np.lexsort((Q[np.arange(N), dom_comp], dom_comp, pop_int))
    elif sort_by_pop:
        dom_comp = Q.argmax(axis=1)
        order = np.lexsort((-Q[np.arange(N), dom_comp], dom_comp))
    else:
        order = np.arange(N)

    Q_sorted = Q[order]
    pops_sorted = pop_labels[order] if pop_labels is not None else None

    # Render as RGB image (much faster than individual bar patches for large N)
    rgb_img = _q_to_rgb(Q_sorted, cols)   # (N, H, 3)
    ax.imshow(rgb_img.transpose(1, 0, 2), aspect='auto',
              extent=[-0.5, N - 0.5, 0, 1], interpolation='nearest')

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Ancestry', fontsize=7)
    if title:
        ax.set_title(title, fontsize=8, pad=2)

    if not show_xticks:
        ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=6)

    # Population dividers and labels
    if show_pop_dividers and pops_sorted is not None:
        prev_pop, dividers, centres = None, [], []
        run_start = 0
        for i, pop in enumerate(pops_sorted):
            if pop != prev_pop:
                if prev_pop is not None:
                    dividers.append(i - 0.5)
                    centres.append((run_start + i - 1) / 2)
                run_start = i
                prev_pop = pop
        centres.append((run_start + N - 1) / 2)   # last group

        for d in dividers:
            ax.axvline(d, color='black', linewidth=0.6, zorder=5)

        unique_pops_sorted = list(dict.fromkeys(pops_sorted))
        for pop, cx in zip(unique_pops_sorted, centres):
            ax.text(cx, -0.12, pop, ha='center', va='top', fontsize=5,
                    transform=ax.get_xaxis_transform(), rotation=45)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-K structure panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_structure_panel(Q_by_k: Dict[int, np.ndarray],
                         pop_labels: Optional[np.ndarray] = None,
                         sample_ids: Optional[np.ndarray] = None,
                         figsize: Optional[tuple] = None,
                         out_path: str = 'structure_plot.pdf',
                         dpi: int = 150,
                         suptitle: str = '') -> str:
    """
    Create a multi-K STRUCTURE panel (one row per K value).

    Colours are consistent across K: component 0 always sky-blue, etc.
    Individual sort order is determined by K=max (most informative) and
    kept constant across all K panels.

    Parameters
    ----------
    Q_by_k    : {K: (N, K) aligned Q matrix} — use output of align_across_k()
    pop_labels: (N,) population assignment string per individual
    sample_ids: (N,) sample IDs (used if pop_labels is None)
    figsize   : (width, height) in inches; auto-sized if None
    out_path  : output file path (.pdf, .png, .svg)
    dpi       : raster DPI (only for .png)
    suptitle  : overall figure title

    Returns
    -------
    out_path : the path where the figure was saved
    """
    K_list = sorted(Q_by_k.keys())
    n_rows = len(K_list)
    N      = next(iter(Q_by_k.values())).shape[0]

    if figsize is None:
        width  = max(8, N / 100)   # ~1 px per 10 individuals; min 8 in
        height = n_rows * 0.9 + 0.8
        figsize = (width, height)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize,
                              gridspec_kw={'hspace': 0.05})
    if n_rows == 1:
        axes = [axes]

    # Use K_max to define the canonical sort order (most components = most info)
    K_max = K_list[-1]
    Q_ref_sorted  = Q_by_k[K_max]
    dom_comp      = Q_ref_sorted.argmax(axis=1)
    if pop_labels is not None:
        unique_pops = list(dict.fromkeys(pop_labels))
        pop_to_idx  = {p: i for i, p in enumerate(unique_pops)}
        pop_int     = np.array([pop_to_idx[p] for p in pop_labels])
        sort_order  = np.lexsort((Q_ref_sorted[np.arange(N), dom_comp],
                                   dom_comp, pop_int))
    else:
        sort_order = np.lexsort((-Q_ref_sorted[np.arange(N), dom_comp], dom_comp))

    for ax_idx, (K, ax) in enumerate(zip(K_list, axes)):
        Q = Q_by_k[K][sort_order]
        cols = _palette(max(K_list))[:K]   # same palette slice for all K

        # Fast imshow rendering
        rgb_img = _q_to_rgb(Q, cols)
        ax.imshow(rgb_img.transpose(1, 0, 2), aspect='auto',
                  extent=[-0.5, N - 0.5, 0, 1], interpolation='nearest')

        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f'K={K}', fontsize=7, rotation=0, labelpad=20, va='center')

        # Population dividers (bottom-most panel only labels)
        show_labels = (ax_idx == n_rows - 1)
        if pop_labels is not None:
            pops_s = np.array(pop_labels)[sort_order]
            prev_pop, dividers, centres = None, [], []
            run_start = 0
            for i, pop in enumerate(pops_s):
                if pop != prev_pop:
                    if prev_pop is not None:
                        dividers.append(i - 0.5)
                        centres.append((run_start + i - 1) / 2)
                    run_start = i
                    prev_pop = pop
            centres.append((run_start + N - 1) / 2)
            for d in dividers:
                ax.axvline(d, color='black', linewidth=0.5, zorder=5)
            if show_labels:
                unique_pops_sorted = list(dict.fromkeys(pops_s))
                for pop, cx in zip(unique_pops_sorted, centres):
                    ax.text(cx, -0.06, pop, ha='center', va='top',
                            fontsize=4.5, transform=ax.get_xaxis_transform(),
                            rotation=45)

    # Legend (component colours up to K_max)
    legend_patches = [mpatches.Patch(color=_palette(K_max)[k], label=f'Comp {k+1}')
                       for k in range(K_max)]
    fig.legend(handles=legend_patches, loc='center right',
               fontsize=6, ncol=1, bbox_to_anchor=(1.01, 0.5),
               frameon=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)

    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"Structure plot saved: {out_path}")
    return out_path
