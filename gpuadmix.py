#!/usr/bin/env python3
"""
gpuADMIX: GPU-accelerated model-based admixture estimation.
Usage: python gpuadmix.py --bfile PREFIX --K 5 --out results/
"""
import argparse, json, os, sys, time
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.bed_reader import read_bed_tensor
from src.gpuadmix_core import fit


def save_results(result, out_dir, basename):
    K = result['K']
    seed = result['seed']
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{basename}.K{K}.s{seed}"
    np.savetxt(os.path.join(out_dir, f"{stem}.Q"), result['Q'], fmt='%.6f')
    np.savetxt(os.path.join(out_dir, f"{stem}.P"), result['P'], fmt='%.6f')
    metrics = {
        'K': K, 'seed': seed,
        'log_likelihood': float(result['log_likelihoods'][-1][1]),
        'n_iter': result['n_iter_run'],
        'fit_time_sec': result['elapsed_sec'],
    }
    with open(os.path.join(out_dir, f"{stem}.metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile', required=True)
    parser.add_argument('-K', '--K', type=int, required=True)
    parser.add_argument('-o', '--out', default='results/gpuadmix')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--n-iter', type=int, default=500)
    parser.add_argument('--n-batches', type=int, default=-1,
                        help='Number of SNP mini-batches (-1 = auto: M/12500)')
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--check-interval', type=int, default=10)
    parser.add_argument('--no-nesterov', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    t0 = time.time()
    basename = os.path.basename(args.bfile)
    print(f"Loading {args.bfile}...")
    G, sample_ids, snp_ids = read_bed_tensor(args.bfile, device='cpu', impute=True)
    print(f"Loaded N={G.shape[0]}, M={G.shape[1]} ({time.time()-t0:.1f}s)")

    result = fit(
        G=G.to(args.device), K=args.K, seed=args.seed,
        n_iter=args.n_iter, n_batches=args.n_batches, tol=args.tol,
        check_interval=args.check_interval, use_nesterov=not args.no_nesterov,
        verbose=True,
    )
    result['K'] = args.K
    result['seed'] = args.seed

    metrics = save_results(result, args.out, basename)
    print(f"Total: {time.time()-t0:.1f}s | LL={metrics['log_likelihood']:.2f}")


if __name__ == '__main__':
    main()
