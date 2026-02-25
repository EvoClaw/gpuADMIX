#!/usr/bin/env python3
"""
Collect and summarize all Phase 4b results.
Computes: LL, fit_time, Q r^2/RMSE vs fastmixture, speedup.
Outputs a CSV and a summary table.
"""
import os, json, glob
import numpy as np
from scipy.stats import pearsonr
from itertools import permutations

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BNAME  = '1kGP_200k_ldpruned'
K_LIST = list(range(2, 11))
SEEDS  = [42, 1, 2, 3, 7]
FM_SEEDS = [42, 1, 2]


def best_permuted_r2(Q_ref, Q_test):
    """Best mean r^2 over column permutations."""
    K = Q_ref.shape[1]
    best = 0.0
    for perm in permutations(range(K)):
        r2s = [pearsonr(Q_ref[:, k], Q_test[:, perm[k]])[0] ** 2 for k in range(K)]
        if np.mean(r2s) > best:
            best = np.mean(r2s)
    return best


def load_metrics(method, K, seed):
    if method == 'gpuadmix':
        path = os.path.join(ROOT, 'results', 'gpuadmix', f'{BNAME}.K{K}.s{seed}.metrics.json')
        if not os.path.exists(path):
            return None
        return json.load(open(path))
    elif method == 'fastmixture':
        # Parse fastmixture log for timing and LL
        logpath = os.path.join(ROOT, 'logs', f'fastmixture_K{K}_s{seed}.log')
        if not os.path.exists(logpath):
            return None
        ll, t = None, None
        for line in open(logpath):
            if 'Final log-like' in line or 'loglike' in line.lower():
                try:
                    ll = float(line.strip().split()[-1])
                except Exception:
                    pass
            if 'Total time' in line or 'Elapsed' in line.lower():
                try:
                    t = float(line.strip().split()[-1].replace('s', ''))
                except Exception:
                    pass
        return {'log_likelihood': ll, 'fit_time_sec': t, 'K': K, 'seed': seed}
    return None


def load_Q(method, K, seed):
    if method == 'gpuadmix':
        path = os.path.join(ROOT, 'results', 'gpuadmix', f'{BNAME}.K{K}.s{seed}.Q')
    elif method == 'fastmixture':
        path = os.path.join(ROOT, 'results', 'fastmixture', f'{BNAME}.K{K}.s{seed}.Q')
    else:
        return None
    if not os.path.exists(path):
        return None
    return np.loadtxt(path)


def main():
    rows = []
    for K in K_LIST:
        # Reference: fastmixture seed=42 (or first available seed)
        Q_fm_ref = None
        for s in FM_SEEDS:
            Q_fm_ref = load_Q('fastmixture', K, s)
            if Q_fm_ref is not None:
                break

        for seed in SEEDS:
            for method in ['gpuadmix', 'fastmixture']:
                if method == 'fastmixture' and seed not in FM_SEEDS:
                    continue
                m = load_metrics(method, K, seed)
                if m is None:
                    continue
                row = {
                    'method': method, 'K': K, 'seed': seed,
                    'LL': m.get('log_likelihood'),
                    'time_sec': m.get('fit_time_sec'),
                }
                Q = load_Q(method, K, seed)
                if Q is not None and Q_fm_ref is not None and method != 'fastmixture':
                    if K <= 8:  # permutation is expensive for large K
                        row['Q_r2_vs_fm'] = best_permuted_r2(Q_fm_ref, Q)
                    row['Q_rmse_vs_fm'] = float(np.sqrt(np.mean((Q_fm_ref - Q) ** 2)))
                rows.append(row)

    if not rows:
        print("No results found yet. Run the experiments first.")
        return

    # Print summary table grouped by K
    print(f"\n{'Method':<14} {'K':>3} {'Seed':>5}  {'LL':>16}  {'Time(s)':>9}  {'Q_r2_fm':>10}")
    print('-' * 65)
    for r in sorted(rows, key=lambda x: (x['K'], x['method'], x['seed'])):
        ll   = f"{r['LL']:.0f}" if r['LL'] else 'N/A'
        t    = f"{r['time_sec']:.1f}" if r['time_sec'] else 'N/A'
        r2   = f"{r.get('Q_r2_vs_fm', float('nan')):.6f}"
        print(f"{r['method']:<14} {r['K']:>3} {r['seed']:>5}  {ll:>16}  {t:>9}  {r2:>10}")

    # Per-K speedup summary
    print(f"\n{'K':>3}  {'gpuADMIX_LL':>16}  {'fm_LL':>16}  {'speedup':>10}")
    print('-' * 55)
    for K in K_LIST:
        gpu_lls = [r['LL'] for r in rows if r['method'] == 'gpuadmix' and r['K'] == K and r['LL']]
        fm_lls  = [r['LL'] for r in rows if r['method'] == 'fastmixture' and r['K'] == K and r['LL']]
        gpu_ts  = [r['time_sec'] for r in rows if r['method'] == 'gpuadmix' and r['K'] == K and r['time_sec']]
        fm_ts   = [r['time_sec'] for r in rows if r['method'] == 'fastmixture' and r['K'] == K and r['time_sec']]
        if gpu_lls and fm_lls and gpu_ts and fm_ts:
            speedup = np.mean(fm_ts) / np.mean(gpu_ts)
            print(f"{K:>3}  {np.mean(gpu_lls):>16.0f}  {np.mean(fm_lls):>16.0f}  {speedup:>10.1f}x")
        else:
            print(f"{K:>3}  (incomplete)")

    # Save CSV
    import csv
    outpath = os.path.join(ROOT, 'results', 'phase4b_summary.csv')
    if rows:
        with open(outpath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
