#!/usr/bin/env python3
"""
make_figures.py — Generate all publication-quality figures for gpuADMIX paper.

Figures produced:
  Fig 1: Main speed + accuracy panel (a: wall time; b: LL gain; c: Q r²)
  Fig 2: K-scan results (a: LL vs K; b: within-K RMSE; c: time vs K)
  Fig 3: Ablation panel (a: Nesterov vs plain EM; b: mini-batch size sweep)
  Fig 4: Structure plot (K=2..7, from CLUMPAK-lite)
"""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (apply_style, COLORS, METHOD_COLORS,
                              SINGLE_COL_WIDTH, DOUBLE_COL_WIDTH, panel_label)

apply_style()

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
METHODS   = ['ADMIXTURE', 'fastmixture', 'gpuADMIX']
TIMES     = [3583.0, 694.0, 16.8]
TIME_STDS = [0.0, 40.0, 3.4]
LLS       = [-241227839, -241227643, -241224751]
LL_STDS   = [0, 0.4, 98]
# Q r² of fastmixture vs ADMIXTURE, and gpuADMIX vs ADMIXTURE
FM_ADMIX_R2  = 0.999984
GPU_ADMIX_R2 = 0.999987

K_VALUES  = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# mean LL across 5 seeds (K=2..7) and fresh 5-seed runs (K=8..10)
GPU_LLS   = [-252414278, -245521514, -243334232, -241224751, -240881211, -240609208,
             -240367954, -240130122, -239950870]
GPU_LL_STDS = [31, 20, 63440, 98, 1021, 15550, 15948, 39920, 30305]
# best LL across 5 seeds (from stored metrics K=2..7; fresh 5-seed runs K=8..10)
GPU_BEST_LLS = [-252414251, -245521479, -243207351, -241224601, -240879190, -240595379,
                -240346229, -240103348, -239893430]
GPU_TIMES = [13.7, 14.4, 15.9, 16.8, 22.0, 48.8, 60.1, 65.1, 70.6]
GPU_TIME_STDS = [3.1, 3.6, 3.4, 3.4, 2.7, 11.4, 24.0, 20.0, 28.0]
WITHIN_K_RMSE = [0.0003, 0.0002, 0.1729, 0.0013, 0.0162, 0.1230]  # K=2..7
FM_K    = [2, 3, 4, 5, 6, 7, 8, 9, 10]
FM_LLS  = [-252413106, -245523810, -243368672, -241227643, -240879255, -240626481,
           -240346727, -240105415, -239939093]

CV_K   = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CV_LLS = [-105227322, -82043982, -77744714, -74662386,
          -70879014,  -69896182, -70673458, -69525200, -70127534]

# Ablation
NEST_TAGS  = ['gpuADMIX\n(Nesterov)', 'gpuADMIX\n(Plain-EM)']
NEST_LLS   = [-241224912, -241232777]
NEST_STDS  = [317, 745]
NEST_ITERS = [47, 107]
NEST_TIMES = [13.5, 12.2]

MB_NB  = [1, 4, 8, 16, 32]
MB_LLS = [-241225734, -241225231, -241225168, -241224710, -241224960]
MB_T   = [27.5, 17.1, 14.3, 14.3, 11.6]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Main performance panel
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.4),
                             gridspec_kw={'wspace': 0.45})
    ax_t, ax_ll, ax_q = axes
    colors = [METHOD_COLORS[m] for m in METHODS]
    x = np.arange(3)
    labels = ['ADMIXTURE', 'fastmixture', 'gpuADMIX']

    # (a) Wall time — log scale
    ax_t.bar(x, TIMES, color=colors, width=0.5, zorder=3,
             yerr=TIME_STDS, capsize=3, error_kw={'linewidth': 0.8})
    ax_t.set_yscale('log')
    ax_t.set_xticks(x); ax_t.set_xticklabels(labels, rotation=25, ha='right', fontsize=6.5)
    ax_t.set_ylabel('Wall time (s)')
    ax_t.set_ylim(5, 15000)
    # speedup annotation
    ax_t.annotate('41×', xy=(2, 30), xytext=(1.3, 150),
                  fontsize=6, color='black',
                  arrowprops=dict(arrowstyle='->', color='black', lw=0.7))
    ax_t.text(0.5, 0.92, '213× vs ADMIXTURE', transform=ax_t.transAxes,
              ha='center', va='top', fontsize=5.5, color='gray')
    panel_label(ax_t, 'a')

    # (b) ΔLL vs ADMIXTURE
    delta_ll = [ll - LLS[0] for ll in LLS]   # [0, +196, +3088]
    ax_ll.bar(x, delta_ll, color=colors, width=0.5, zorder=3,
              yerr=LL_STDS, capsize=3, error_kw={'linewidth': 0.8})
    ax_ll.set_xticks(x); ax_ll.set_xticklabels(labels, rotation=25, ha='right', fontsize=6.5)
    ax_ll.set_ylabel('ΔLog-likelihood\nvs ADMIXTURE')
    ax_ll.axhline(0, color='gray', lw=0.5, ls='--', zorder=2)
    for i, d in enumerate(delta_ll):
        if d > 0:
            ax_ll.text(i, d + 60, f'+{d:,.0f}', ha='center', va='bottom',
                       fontsize=5.5, color=colors[i])
    panel_label(ax_ll, 'b')

    # (c) Q r² vs ADMIXTURE — direct values with zoomed axis
    q_vals = [FM_ADMIX_R2, GPU_ADMIX_R2]
    ax_q.bar([0, 1], q_vals,
             color=[METHOD_COLORS['fastmixture'], METHOD_COLORS['gpuADMIX']],
             width=0.5, zorder=3)
    ax_q.set_xticks([0, 1])
    ax_q.set_xticklabels(['fastmixture', 'gpuADMIX'], rotation=25, ha='right', fontsize=6.5)
    ax_q.set_ylabel('Q r² vs ADMIXTURE')
    ax_q.set_ylim(0.99996, 1.000015)
    ax_q.axhline(1.0, color='gray', lw=0.5, ls='--', zorder=2)
    ax_q.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.6f'))
    for i, v in enumerate(q_vals):
        ax_q.text(i, v - 0.000005, f'{v:.6f}', ha='center', va='top', fontsize=5.5)
    panel_label(ax_q, 'c')

    out = os.path.join(FIGDIR, 'fig1_main_performance.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: K-scan results
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2():
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL_WIDTH, 2.2),
                             gridspec_kw={'wspace': 0.5})
    ax_ll, ax_rmse, ax_cv, ax_t = axes

    # (a) ΔLL (GPU - fastmixture) vs K — shows advantage clearly
    fm_arr   = np.array(FM_LLS)
    delta_mean = np.array(GPU_LLS)      - fm_arr    # mean
    delta_best = np.array(GPU_BEST_LLS) - fm_arr    # best seed

    zero_line = ax_ll.axhline(0, color=METHOD_COLORS['fastmixture'],
                               lw=1.0, ls='--', zorder=2, label='fastmixture')
    ax_ll.bar(np.array(K_VALUES) - 0.18, delta_mean / 1e3,
              width=0.34, color=METHOD_COLORS['gpuADMIX'],
              alpha=0.75, label='gpuADMIX mean', zorder=3)
    ax_ll.bar(np.array(K_VALUES) + 0.18, delta_best / 1e3,
              width=0.34, color=METHOD_COLORS['gpuADMIX'],
              alpha=1.0, label='gpuADMIX best', zorder=3)
    ax_ll.set_xlabel('K')
    ax_ll.set_ylabel('ΔLog-likelihood vs FM\n(×10³, positive = better)')
    ax_ll.set_xticks(K_VALUES)
    ax_ll.tick_params(axis='x', labelsize=6)
    ax_ll.legend(loc='upper right', frameon=False, fontsize=5.5, ncol=1)
    panel_label(ax_ll, 'a')

    # (b) Within-K RMSE — stability (K=2..7)
    K_rmse = [2, 3, 4, 5, 6, 7]
    bar_cols = [METHOD_COLORS['gpuADMIX'] if r < 0.05 else COLORS[4] for r in WITHIN_K_RMSE]
    ax_rmse.bar(K_rmse, WITHIN_K_RMSE, color=bar_cols, width=0.6, zorder=3)
    ax_rmse.axhline(0.05, color='gray', lw=0.5, ls='--', zorder=2)
    ax_rmse.set_xlabel('K')
    ax_rmse.set_ylabel('Within-K RMSE')
    ax_rmse.set_xticks(K_rmse)
    for k, r in zip(K_rmse, WITHIN_K_RMSE):
        if r > 0.05:
            ax_rmse.text(k, r + 0.005, f'{r:.2f}', ha='center', va='bottom',
                         fontsize=6, color=COLORS[4], fontweight='bold')
    ax_rmse.text(5, 0.003, 'K=5\n(stable)', ha='center', va='bottom', fontsize=5.5,
                 color=METHOD_COLORS['gpuADMIX'])
    panel_label(ax_rmse, 'b')

    # (c) 5-fold CV LL vs K
    best_k = CV_K[np.argmax(CV_LLS)]
    cv_arr = np.array(CV_LLS) / 1e6
    ax_cv.plot(CV_K, cv_arr, 'o-', color=METHOD_COLORS['gpuADMIX'],
               markersize=3.5, zorder=4)
    ax_cv.scatter([best_k], [cv_arr[CV_K.index(best_k)]],
                  color=COLORS[4], s=30, zorder=5)
    ax_cv.set_xlabel('K')
    ax_cv.set_ylabel('5-fold CV LL (×10⁶)')
    ax_cv.set_xticks(CV_K)
    ax_cv.tick_params(axis='x', labelsize=6)
    ax_cv.text(best_k, cv_arr[CV_K.index(best_k)] - 1.0,
               f'K={best_k}', ha='center', va='top', fontsize=6, color=COLORS[4])
    panel_label(ax_cv, 'c')

    # (d) Wall time vs K
    ax_t.errorbar(K_VALUES, GPU_TIMES, yerr=GPU_TIME_STDS,
                  fmt='o-', color=METHOD_COLORS['gpuADMIX'],
                  capsize=3, linewidth=1.2, markersize=3.5)
    ax_t.set_xlabel('K')
    ax_t.set_ylabel('Wall time (s)')
    ax_t.set_xticks(K_VALUES)
    ax_t.tick_params(axis='x', labelsize=6)
    panel_label(ax_t, 'd')

    out = os.path.join(FIGDIR, 'fig2_k_scan.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Ablation panel
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH * 0.67, 2.2),
                             gridspec_kw={'wspace': 0.55})
    ax_n, ax_mb = axes

    # (a) Nesterov vs Plain-EM — show ΔLL (Nesterov gain over Plain-EM)
    # Nesterov is better by +7865; bar heights: Nesterov=+7865, Plain-EM=0
    delta_n = [NEST_LLS[0] - NEST_LLS[1], 0]   # [+7865, 0]
    bar_colors = [METHOD_COLORS['gpuADMIX'], COLORS[2]]
    x = [0, 1]
    ax_n.bar(x, delta_n, color=bar_colors, width=0.5, zorder=3,
             yerr=NEST_STDS, capsize=3, error_kw={'linewidth': 0.8})
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(['gpuADMIX\n(Nesterov)', 'gpuADMIX\n(Plain-EM)'], fontsize=6.5)
    ax_n.set_ylabel('ΔLog-likelihood\nvs Plain-EM')
    ax_n.axhline(0, color='gray', lw=0.5, ls='--')
    ax_n.text(0, delta_n[0] * 0.45, f'+{NEST_LLS[0]-NEST_LLS[1]:,}\n({NEST_ITERS[0]} vs {NEST_ITERS[1]} iters)',
              ha='center', va='center', fontsize=6, color=METHOD_COLORS['gpuADMIX'])
    panel_label(ax_n, 'a')

    # (b) Mini-batch size sweep
    ax_mb2 = ax_mb.twinx()
    ax_mb.plot(MB_NB, [-l/1e6 for l in MB_LLS],
               'o-', color=METHOD_COLORS['gpuADMIX'], markersize=5, label='−LL', zorder=4)
    ax_mb2.plot(MB_NB, MB_T,
                's--', color=COLORS[1], markersize=4, label='Time (s)', zorder=3)
    ax_mb.set_xlabel('Number of mini-batches')
    ax_mb.set_ylabel('−Log-likelihood (×10⁶)', color=METHOD_COLORS['gpuADMIX'])
    ax_mb2.set_ylabel('Wall time (s)', color=COLORS[1])
    ax_mb.tick_params(axis='y', labelcolor=METHOD_COLORS['gpuADMIX'])
    ax_mb2.tick_params(axis='y', labelcolor=COLORS[1])
    ax_mb.set_xticks(MB_NB)
    # mark default (n_batches=16)
    ax_mb.axvline(16, color='gray', lw=0.5, ls=':', zorder=2)
    ax_mb.text(17, ax_mb.get_ylim()[0] + (ax_mb.get_ylim()[1]-ax_mb.get_ylim()[0])*0.02,
               'default', ha='left', va='bottom', fontsize=5.5, color='gray')
    lines1, labs1 = ax_mb.get_legend_handles_labels()
    lines2, labs2 = ax_mb2.get_legend_handles_labels()
    ax_mb.legend(lines1+lines2, labs1+labs2, loc='upper right', frameon=False, fontsize=6)
    panel_label(ax_mb, 'b')

    out = os.path.join(FIGDIR, 'fig3_ablation.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Structure plot
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4():
    from src.structure_plot import plot_structure_panel

    BNAME  = '1kGP_200k_ldpruned'
    K_LIST = [2, 3, 4, 5, 6, 7]
    ALIGNED_DIR = os.path.join(ROOT, 'results', 'aligned')
    Q_by_k = {}
    for K in K_LIST:
        p = os.path.join(ALIGNED_DIR, f'{BNAME}.K{K}.Q_struct')
        if os.path.exists(p):
            Q_by_k[K] = np.loadtxt(p)

    fam_ids   = [l.split()[1] for l in open(os.path.join(ROOT, f'{BNAME}.fam'))]
    info      = {l.split()[0]: l.split()[1]
                 for l in open(os.path.join(ROOT, 'samples.info')) if len(l.split()) >= 2}
    pop_labels = np.array([info.get(sid, 'UNK') for sid in fam_ids])

    out = os.path.join(FIGDIR, 'fig4_structure_plot.pdf')
    plot_structure_panel(Q_by_k=Q_by_k, pop_labels=pop_labels,
                         out_path=out, dpi=300, suptitle='')
    return out


if __name__ == '__main__':
    import matplotlib
    print("Generating publication-quality figures...")
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print(f"\nAll figures saved to {FIGDIR}")
    print("Per-figure checklist (Life Science / Bioinformatics Oxford profile):")
    print("  ✓ Sans-serif font (Helvetica/Arial/DejaVu)")
    print("  ✓ Colorblind-safe Okabe-Ito palette")
    print("  ✓ Consistent method-color mapping (blue=gpuADMIX, orange=fastmixture, red=ADMIXTURE)")
    print("  ✓ Error bars / std shown where applicable")
    print("  ✓ No figure titles on the image (go in LaTeX caption)")
    print("  ✓ Panel labels: a, b, c")
    print("  ✓ Vector format (PDF)")
    print("  ✓ No top/right spines")
