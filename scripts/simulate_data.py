#!/usr/bin/env python3
"""
Phase 4b Simulation: generate synthetic admixed genotypes for benchmarking.
Uses the ADMIXTURE generative model (exactly what the tool assumes):
  1. Ancestral allele frequencies from Beta(0.5, 0.5)
  2. Per-population drift: P_k ~ Beta_drift(P_anc, Fst) 
  3. Genotypes: G_ij ~ Binomial(2, sum_k Q_ik * P_jk)

Three datasets:
  SIM_K3:    K=3, N=2000,  M=100K  (basic validation)
  SIM_K5:    K=5, N=3000,  M=200K  (mirrors 1kGP scale)
  SIM_LARGE: K=5, N=30000, M=200K  (scaling benchmark)
"""
import numpy as np, os, struct
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SIMDIR = ROOT / 'simdata'
SIMDIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation core: exact ADMIXTURE generative model
# ─────────────────────────────────────────────────────────────────────────────

def simulate_admixture_model(K: int, N: int, M_target: int,
                              Fst: float = 0.12, admix_frac: float = 0.25,
                              seed: int = 42) -> tuple:
    """
    Simulate genotypes and true admixture proportions.

    Returns:
        G (N, M) int8  — A1 (minor) allele counts, 0/1/2 (no missing)
        Q (N, K) float — true admixture fractions (rows sum to 1)
        P (M, K) float — true ancestral allele frequencies
    """
    rng = np.random.default_rng(seed)

    # 1. Ancestral allele frequencies (U-shaped, as in real SNP arrays)
    p_anc = rng.beta(0.5, 0.5, size=M_target * 2)  # oversample, then filter

    # 2. Per-pop drift: Beta parameterized by Fst
    #    alpha = p_anc * (1 - Fst) / Fst,  beta = (1-p_anc) * (1-Fst) / Fst
    eff = (1 - Fst) / Fst
    P_full = np.zeros((M_target * 2, K))
    for k in range(K):
        alpha = p_anc * eff
        beta  = (1 - p_anc) * eff
        P_full[:, k] = rng.beta(alpha, beta)

    P_full = np.clip(P_full, 1e-4, 1 - 1e-4)

    # 3. MAF filter: keep SNPs with mean MAF >= 0.05
    mean_freq = P_full.mean(axis=1)
    maf = np.minimum(mean_freq, 1 - mean_freq)
    ok = maf >= 0.05
    P = P_full[ok][:M_target]
    M = P.shape[0]
    print(f"  SNPs after MAF filter: {ok.sum()}/{M_target*2} → using {M}")

    # 4. True Q matrix
    n_pure  = int(N * (1 - admix_frac))
    n_admix = N - n_pure
    n_per_k = n_pure // K

    Q = np.zeros((N, K))
    ind = 0
    for k in range(K):
        n_k = n_per_k + (1 if k < n_pure % K else 0)
        Q[ind:ind + n_k, k] = 1.0
        ind += n_k

    # Admixed: Dirichlet with sparse concentration (realistic)
    alpha_dir = rng.exponential(scale=1.0, size=(n_admix, K))
    alpha_dir = np.clip(alpha_dir, 0.1, None)
    Q_admix = alpha_dir / alpha_dir.sum(axis=1, keepdims=True)
    Q[ind:ind + n_admix] = Q_admix

    # 5. Genotypes: G[i,j] ~ Binomial(2, sum_k Q[i,k] * P[j,k])
    mix_freq = (Q @ P.T).clip(1e-6, 1 - 1e-6)  # (N, M)

    # Sample in batches for memory efficiency
    batch = 5000
    G_list = []
    for b in range(0, N, batch):
        G_list.append(rng.binomial(2, mix_freq[b:b+batch]).astype(np.int8))
    G = np.concatenate(G_list, axis=0)

    return G, Q, P


# ─────────────────────────────────────────────────────────────────────────────
# Write PLINK BED / BIM / FAM
# ─────────────────────────────────────────────────────────────────────────────

def write_bed(G: np.ndarray, prefix: str):
    """
    Write (N, M) genotype matrix (A1 allele counts 0/1/2) to PLINK SNP-major BED.
    PLINK encoding: 00=hom_A1=2, 01=missing, 10=het=1, 11=hom_A2=0.
    """
    N, M = G.shape
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else '.', exist_ok=True)

    # Vectorised packing: each SNP → ceil(N/4) bytes
    # Map: g=2→00, g=1→10, g=0→11, g=-1→01
    code_map = np.array([3, 2, 0, 1], dtype=np.uint8)   # index by (2-g)
    G_code   = code_map[2 - G.clip(-1, 2)]               # (N, M) uint8, values 0-3

    # Pad to multiple of 4 rows
    pad = (-N) % 4
    if pad:
        G_code = np.vstack([G_code, np.ones((pad, M), dtype=np.uint8) * 1])  # missing

    # Pack 4 samples per byte, SNP-major
    G_rs  = G_code.reshape(-1, 4, M)            # (N_pad/4, 4, M)
    packed = (G_rs[:, 0, :] |
              (G_rs[:, 1, :] << 2) |
              (G_rs[:, 2, :] << 4) |
              (G_rs[:, 3, :] << 6)).astype(np.uint8)  # (N_pad/4, M)
    packed_T = packed.T.copy()                         # (M, N_pad/4)

    with open(prefix + '.bed', 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(packed_T.tobytes())

    with open(prefix + '.bim', 'w') as f:
        for j in range(M):
            f.write(f"1\tsnp{j+1}\t0\t{j+1}\tA\tG\n")

    with open(prefix + '.fam', 'w') as f:
        for i in range(N):
            f.write(f"FAM{i+1}\tind{i+1}\t0\t0\t0\t-9\n")

    print(f"  Written BED: {prefix}.bed (N={N}, M={M})")


def run_simulation(name: str, K: int, N: int, M: int,
                   Fst: float = 0.12, admix_frac: float = 0.25, seed: int = 42):
    print(f"\n=== {name}: K={K}, N={N}, M={M}, Fst={Fst} ===")
    prefix = str(SIMDIR / name)
    G, Q, P = simulate_admixture_model(K=K, N=N, M_target=M, Fst=Fst,
                                        admix_frac=admix_frac, seed=seed)
    write_bed(G, prefix)
    np.savetxt(prefix + '.Q_true', Q, fmt='%.6f')
    np.savetxt(prefix + '.P_true', P, fmt='%.6f')
    print(f"  Q_true: {prefix}.Q_true  (pure: {int(N*(1-admix_frac))}, admixed: {int(N*admix_frac)})")


if __name__ == '__main__':
    import time
    t0 = time.time()
    print("Generating simulation datasets...")
    run_simulation('SIM_K3', K=3, N=2000,  M=100000, Fst=0.12, admix_frac=0.25, seed=42)
    run_simulation('SIM_K5', K=5, N=3000,  M=200000, Fst=0.12, admix_frac=0.25, seed=42)
    run_simulation('SIM_LARGE', K=5, N=30000, M=200000, Fst=0.10, admix_frac=0.20, seed=42)
    print(f"\nAll simulations complete in {time.time()-t0:.1f}s.")
