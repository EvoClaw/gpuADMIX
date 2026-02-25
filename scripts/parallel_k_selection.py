#!/usr/bin/env python3
"""
Multi-GPU parallel K selection: run K=2..K_max simultaneously, one K per GPU.
Selects optimal K by BIC after all runs complete.
Total wall time ≈ time(K_max) instead of sum of all K times.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
import numpy as np

from src.bed_reader import read_bed_tensor
from src.gpuadmix_core import fit


def run_k_on_gpu(gpu_id: int, K: int, G_shared: torch.Tensor, seed: int,
                 result_queue: mp.Queue):
    """Worker: run gpuADMIX for one K on one GPU, return result."""
    try:
        torch.cuda.set_device(gpu_id)
        G = G_shared.to(f'cuda:{gpu_id}')
        r = fit(G, K=K, seed=seed, verbose=False)
        ll = float(r['log_likelihoods'][-1][1])
        t  = r['elapsed_sec']
        result_queue.put({'K': K, 'gpu': gpu_id, 'LL': ll, 'time': t,
                          'n_iter': r['n_iter_run'], 'status': 'ok',
                          'Q': r['Q'], 'P': r['P']})
    except Exception as e:
        result_queue.put({'K': K, 'gpu': gpu_id, 'status': 'error', 'error': str(e)})


def bic(ll: float, K: int, N: int, M: int) -> float:
    d = (K - 1) * N + (K - 1) * M   # free parameters
    return -2.0 * ll + d * np.log(N * M)


def parallel_k_select(bfile: str, K_min: int = 2, K_max: int = 10,
                       seed: int = 42, out_dir: str = 'results/gpuadmix',
                       n_gpus: int = -1) -> dict:
    """
    Run K=K_min..K_max in parallel across GPUs.

    Returns dict with:
        results: list of per-K results (LL, time, Q, P)
        best_K:  BIC-optimal K
        total_wall_time: elapsed time from start to finish
        speedup_vs_serial: estimated speedup vs serial execution
    """
    n_avail = torch.cuda.device_count()
    if n_gpus == -1:
        n_gpus = n_avail
    n_gpus = min(n_gpus, n_avail)

    K_list = list(range(K_min, K_max + 1))
    n_K    = len(K_list)

    print(f"Parallel K selection: K={K_min}..{K_max} on {n_gpus} GPUs (seed={seed})")

    # Load data once (CPU → shared memory)
    t_load = time.time()
    G_cpu, _, _ = read_bed_tensor(bfile, device='cpu', impute=True)
    N, M = G_cpu.shape
    print(f"Data loaded: N={N}, M={M} ({time.time()-t_load:.1f}s)")

    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(bfile)
    results_by_K = {}
    t_wall = time.time()

    # Dispatch jobs to GPUs in batches (n_gpus per batch)
    ctx = mp.get_context('spawn')
    for batch_start in range(0, n_K, n_gpus):
        batch = K_list[batch_start: batch_start + n_gpus]
        q = ctx.Queue()
        procs = []
        for i, K in enumerate(batch):
            gpu = i % n_gpus
            p = ctx.Process(target=run_k_on_gpu,
                            args=(gpu, K, G_cpu, seed, q))
            p.start()
            procs.append(p)
            print(f"  Dispatched K={K} → GPU {gpu}")

        # Collect results
        for _ in batch:
            res = q.get(timeout=600)
            K = res['K']
            if res['status'] == 'ok':
                results_by_K[K] = res
                stem = f"{basename}.K{K}.s{seed}"
                np.savetxt(os.path.join(out_dir, f"{stem}.Q"), res['Q'], fmt='%.6f')
                np.savetxt(os.path.join(out_dir, f"{stem}.P"), res['P'], fmt='%.6f')
                print(f"  K={K}: LL={res['LL']:.0f}, t={res['time']:.1f}s")
            else:
                print(f"  K={K}: ERROR — {res.get('error', '?')}")

        for p in procs:
            p.join()

    total_wall = time.time() - t_wall

    # BIC selection
    bic_vals = {K: bic(r['LL'], K, N, M) for K, r in results_by_K.items()}
    best_K   = min(bic_vals, key=bic_vals.get)

    # Serial time estimate (sum of individual times)
    serial_est = sum(r['time'] for r in results_by_K.values())
    speedup    = serial_est / total_wall if total_wall > 0 else 1.0

    print(f"\n{'K':>3}  {'LL':>18}  {'BIC':>18}  {'time(s)':>9}")
    print('-' * 55)
    for K in sorted(results_by_K):
        r = results_by_K[K]
        mark = ' ← optimal' if K == best_K else ''
        print(f"K={K:>2}  {r['LL']:>18.0f}  {bic_vals[K]:>18.0f}  {r['time']:>8.1f}s{mark}")

    print(f"\nBIC-optimal K: {best_K}")
    print(f"Total wall time: {total_wall:.1f}s  (serial estimate: {serial_est:.0f}s, "
          f"parallel speedup: {speedup:.1f}x)")

    summary = {
        'K_min': K_min, 'K_max': K_max, 'seed': seed,
        'best_K': best_K,
        'total_wall_time': total_wall,
        'serial_estimate': serial_est,
        'parallel_speedup': speedup,
        'bic_vals': {str(K): v for K, v in bic_vals.items()},
        'per_K': {str(K): {'LL': r['LL'], 'time': r['time'], 'n_iter': r['n_iter']}
                  for K, r in results_by_K.items()},
    }
    with open(os.path.join(out_dir, f'parallel_k_summary_s{seed}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bfile', default='1kGP_200k_ldpruned')
    p.add_argument('--K-min', type=int, default=2)
    p.add_argument('--K-max', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='results/gpuadmix')
    p.add_argument('--n-gpus', type=int, default=-1)
    args = p.parse_args()

    parallel_k_select(
        bfile=args.bfile, K_min=args.K_min, K_max=args.K_max,
        seed=args.seed, out_dir=args.out, n_gpus=args.n_gpus,
    )
