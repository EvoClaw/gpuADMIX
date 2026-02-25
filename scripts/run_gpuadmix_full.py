#!/usr/bin/env python3
"""
Phase 4b: Run gpuADMIX for K=2..10 × 5 seeds in parallel across 8 GPUs.
Assigns GPU i to K=(i+2) round-robin.
"""
import subprocess, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed

BFILE   = '1kGP_200k_ldpruned'
K_LIST  = list(range(2, 11))     # K=2..10
SEEDS   = [42, 1, 2, 3, 7]
N_GPUS  = 8
OUT_DIR = 'results/gpuadmix'
LOG_DIR = 'logs'


def run_one(K, seed, gpu_id):
    log_path = os.path.join(LOG_DIR, f'gpuadmix_K{K}_s{seed}.log')
    cmd = [
        sys.executable, 'gpuadmix.py',
        '--bfile', BFILE,
        '--K', str(K),
        '--seed', str(seed),
        '--out', OUT_DIR,
        '--device', f'cuda:{gpu_id}',
    ]
    t0 = time.time()
    with open(log_path, 'w') as fh:
        ret = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elapsed = time.time() - t0
    return K, seed, gpu_id, ret.returncode, elapsed, log_path


def main():
    jobs = []
    for i, K in enumerate(K_LIST):
        for j, seed in enumerate(SEEDS):
            # Spread jobs: first all K for seed=42 on separate GPUs, then others
            gpu_id = (i + j) % N_GPUS
            jobs.append((K, seed, gpu_id))

    print(f"Launching {len(jobs)} jobs across {N_GPUS} GPUs...")
    print(f"K={K_LIST}, Seeds={SEEDS}")

    # Limit concurrency to N_GPUS to avoid VRAM conflicts
    results = []
    with ProcessPoolExecutor(max_workers=N_GPUS) as ex:
        futures = {ex.submit(run_one, K, seed, gpu): (K, seed) for K, seed, gpu in jobs}
        for fut in as_completed(futures):
            K, seed, gpu_id, rc, elapsed, log = fut.result()
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f"  K={K} seed={seed} gpu={gpu_id}  {elapsed:.1f}s  [{status}]  {log}")
            results.append({'K': K, 'seed': seed, 'status': status, 'time': elapsed})

    # Summary
    ok = sum(1 for r in results if r['status'] == 'OK')
    print(f"\nDone: {ok}/{len(results)} succeeded")
    with open(os.path.join(LOG_DIR, 'gpuadmix_full_run_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
