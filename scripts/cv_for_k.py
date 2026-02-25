#!/usr/bin/env python3
"""
cv_for_k.py — 5-fold cross-validation for K selection.

Standard ADMIXTURE CV approach:
  - Split M SNPs into 5 folds
  - For each fold, train on 80% of SNPs (4 folds)
  - Evaluate hold-out log-likelihood on remaining 20% using trained Q + P
  - CV score(K) = mean hold-out LL across folds (higher = better)
  - Select K maximising CV score

Multi-GPU: assign each K value to a separate GPU (embarrassingly parallel).
"""
import sys, os, json, time
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bed_reader import read_bed_tensor
from src.gpuadmix_core import fit

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BNAME = '1kGP_200k_ldpruned'


def estimate_P_test(Q: np.ndarray, G_test: torch.Tensor,
                    n_iter: int = 30) -> torch.Tensor:
    """
    Estimate P for held-out SNPs given fixed Q (M-step only, 30 iterations).
    Q: (N, K) numpy
    G_test: (N, M_test) float32 GPU tensor
    Returns P_test: (M_test, K) GPU tensor
    """
    EPS = 1e-6
    N, M_test = G_test.shape
    K = Q.shape[1]
    Q_t = torch.tensor(Q, dtype=torch.float32, device=G_test.device)  # (N, K)
    # Initialize P_test with moment estimator: (Q^T @ G_test).T / (2 * Q.sum(0))
    Q_col_sum = Q_t.sum(dim=0)                       # (K,)
    P_t = (Q_t.T @ G_test).T / (2.0 * Q_col_sum)    # (M_test, K)
    P_t   = P_t.clamp(EPS, 1.0 - EPS)
    # Refine with EM M-step iterations
    for _ in range(n_iter):
        H = Q_t @ P_t.T                              # (N, M_test)
        H = H.clamp(EPS, 1.0 - EPS)
        R_minor = G_test      / H
        R_major = (2 - G_test) / (1.0 - H)
        P_num   = (R_minor.T @ Q_t) * P_t           # (M_test, K)
        P_sum   = P_num.sum(dim=1, keepdim=True)
        P_t     = (P_num / P_sum.clamp(EPS)).clamp(EPS, 1.0 - EPS)
    return P_t


def holdout_ll(Q: np.ndarray, G_test: torch.Tensor) -> float:
    """Estimate P for test SNPs, then compute binomial LL."""
    EPS  = 1e-6
    P_t  = estimate_P_test(Q, G_test, n_iter=30)
    Q_t  = torch.tensor(Q, dtype=torch.float32, device=G_test.device)
    H    = Q_t @ P_t.T
    H    = H.clamp(EPS, 1.0 - EPS)
    ll   = (G_test * torch.log(H / 2.0)
            + (2.0 - G_test) * torch.log(1.0 - H / 2.0))
    return ll.sum().item()


def cv_one_k(gpu_id: int, K: int, G_np: np.ndarray, n_folds: int,
             seed: int, result_queue):
    """Run 5-fold CV for one K value on a dedicated GPU."""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    G = torch.tensor(G_np, dtype=torch.float32, device=device)
    N, M = G.shape

    np.random.seed(seed + K * 100)
    perm = np.random.permutation(M)
    fold_size = M // n_folds
    fold_lls = []

    for fold in range(n_folds):
        test_idx  = perm[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([
            perm[:fold * fold_size],
            perm[(fold + 1) * fold_size:]
        ])

        G_train = G[:, train_idx]
        G_test  = G[:, test_idx]

        r = fit(G_train, K=K, seed=seed + fold,
                n_iter=300, tol=1e-4, verbose=False)
        Q  = r['Q']                              # (N, K) — trained on 80% SNPs

        ll = holdout_ll(Q, G_test)               # estimate P_test + evaluate LL
        fold_lls.append(ll)
        print(f"  [GPU {gpu_id}] K={K} fold {fold+1}/{n_folds}  "
              f"hold-out LL={ll:.0f}", flush=True)

    mean_ll = float(np.mean(fold_lls))
    result_queue.put({'K': K, 'cv_ll': mean_ll, 'fold_lls': fold_lls})


def run_cv_parallel(k_range, n_folds=5, seed=42):
    """Distribute K values across available GPUs."""
    n_gpus  = torch.cuda.device_count()
    print(f"Running 5-fold CV for K={k_range[0]}..{k_range[-1]} "
          f"on {n_gpus} GPUs", flush=True)

    G_np = read_bed_tensor(BNAME, device='cpu')[0].numpy()   # load once to RAM

    ctx = mp.get_context('spawn')
    q   = ctx.Queue()
    procs = []
    k_list = list(k_range)

    # Schedule: each GPU handles ceil(|K|/n_gpus) values
    for i, K in enumerate(k_list):
        gpu_id = i % n_gpus
        p = ctx.Process(target=cv_one_k,
                        args=(gpu_id, K, G_np, n_folds, seed, q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = [q.get() for _ in k_list]
    results.sort(key=lambda x: x['K'])
    return results


def run_cv_sequential(k_range, n_folds=5, seed=42, device='cuda:0'):
    """Sequential fallback — single GPU."""
    G_np = read_bed_tensor(BNAME, device='cpu')[0].numpy()
    results = []
    for K in k_range:
        q = type('Q', (), {'put': lambda self, x: results.append(x)})()
        cv_one_k(0, K, G_np, n_folds, seed, q)
    return sorted(results, key=lambda x: x['K'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k-min',  type=int, default=2)
    p.add_argument('--k-max',  type=int, default=10)
    p.add_argument('--folds',  type=int, default=5)
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--sequential', action='store_true')
    args = p.parse_args()

    k_range = range(args.k_min, args.k_max + 1)
    t0 = time.time()

    if args.sequential or torch.cuda.device_count() == 1:
        results = run_cv_sequential(k_range, args.folds, args.seed)
    else:
        results = run_cv_parallel(k_range, args.folds, args.seed)

    elapsed = time.time() - t0

    print("\n" + "="*60)
    print(f"5-Fold CV Results  (elapsed: {elapsed:.1f}s)")
    print("="*60)
    print(f"{'K':>3}  {'Mean CV LL':>15}  {'Fold LLs (summary)'}")
    print("-"*60)
    best_k  = max(results, key=lambda x: x['cv_ll'])['K']
    for r in results:
        marker = " ← best" if r['K'] == best_k else ""
        folds  = r['fold_lls']
        print(f"{r['K']:>3}  {r['cv_ll']:>15.0f}  "
              f"std={np.std(folds):.0f}{marker}")

    print(f"\nCV-optimal K: {best_k}")

    out = os.path.join(ROOT, 'results', 'cv_results.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'best_k': best_k,
                   'elapsed_sec': elapsed}, f, indent=2)
    print(f"Saved: {out}")
