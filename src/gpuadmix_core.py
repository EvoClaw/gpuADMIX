"""
gpuADMIX — GPU-accelerated model-based admixture estimation.

Core algorithm:
  - Binomial likelihood: G[i,j] ~ Binomial(2, H[i,j]) where H = Q @ P.T
  - Mini-batch EM with Nesterov momentum acceleration
  - SVD + ALS initialization
  - Float32 throughout; epsilon-clamping for numerical stability
"""
import math
import time
import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-6
MAX_EPS = 1 - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def _memory_budget_gb(device) -> float:
    """Return free GPU memory in GB (or large number for CPU)."""
    if device.type != 'cuda':
        return 1000.0
    free, total = torch.cuda.mem_get_info(device)
    return free / 1e9


def svd_als_init(G: torch.Tensor, K: int, seed: int = 42,
                 als_iters: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize Q (N×K) and P (M×K) via randomized SVD + ALS.

    Memory-aware: when G is large, avoids materializing G_c = G - 2f as a full
    float32 tensor. Instead computes G_c @ Omega and G_c.T @ Y chunk-by-chunk
    over the M (SNP) dimension, keeping peak extra VRAM ≈ N × chunk_M × 4 bytes.

    Strategy:
      1. f[j] = G[:,j].mean() / 2  (per-SNP A1 allele frequency)
      2. Randomized SVD of G_c = G - 2f  (streaming if large)
      3. Q from PCA coordinates (rows of U*S → project to simplex)
      4. P from f + small random perturbations
      5. 30 rounds of ALS (alternating least squares) to refine both
    """
    torch.manual_seed(seed)
    N, M = G.shape
    device = G.device

    # Per-SNP allele frequency
    f = G.mean(dim=0) / 2.0  # (M,)

    rank = max(1, min(K, min(N, M) - 1))
    q    = rank + 10   # oversampling

    # Decide between in-place and streaming SVD based on available memory.
    # G_c float32 costs N*M*4 bytes; streaming avoids this.
    gc_bytes = N * M * 4
    free_gb  = _memory_budget_gb(device)
    use_streaming = gc_bytes > free_gb * 0.4e9   # use streaming if G_c > 40% of free VRAM

    if use_streaming:
        # ── Streaming randomized SVD — Halko et al. algorithm ────────────────
        # Avoids materializing G_c = G - 2f as a full float32 tensor.
        # Peak extra VRAM per chunk ≈ N × chunk_m × 4 bytes.
        chunk_m = 4096

        # Step 1: Y = G_c @ Omega  (stream over M)
        Omega = torch.randn(M, q, device=device)
        Y = torch.zeros(N, q, device=device)
        for j in range(0, M, chunk_m):
            jj = min(j + chunk_m, M)
            G_cb = G[:, j:jj].float() - 2.0 * f[j:jj]
            Y.add_(G_cb @ Omega[j:jj])
            del G_cb
        del Omega

        # Step 2: Orthogonalize Y → Q_orth (N, q)
        Q_orth, _ = torch.linalg.qr(Y)
        del Y

        # Step 3: B = Q_orth.T @ G_c  (stream over M) → B: (q, M)
        B = torch.zeros(q, M, device=device)
        for j in range(0, M, chunk_m):
            jj = min(j + chunk_m, M)
            G_cb = G[:, j:jj].float() - 2.0 * f[j:jj]
            B[:, j:jj].add_(Q_orth.T @ G_cb)
            del G_cb

        # Step 4: SVD of small B (q, M) → U_hat (q,q), S (q,), Vt (q,M)
        U_hat, Sq, Vt = torch.linalg.svd(B, full_matrices=False)
        del B

        # Step 5: U = Q_orth @ U_hat[:, :rank]  (N, rank)
        U  = Q_orth @ U_hat[:, :rank]   # (N, rank)
        S  = Sq[:rank]                   # (rank,)
        Vt = Vt[:rank, :]                # (rank, M)
        del Q_orth, U_hat, Sq

    else:
        # ── In-memory SVD (standard path for moderate N×M) ──────────────────
        G_c = G - 2.0 * f.unsqueeze(0)   # (N, M) float32
        U, S, Vh = torch.svd_lowrank(G_c, q=rank + 10, niter=2)
        del G_c
        U  = U[:, :rank]      # (N, rank)
        S  = S[:rank]          # (rank,)
        Vt = Vh[:, :rank].T   # (rank, M)

    # Initialize P from allele frequencies + small K-specific perturbations
    P = f.unsqueeze(1).expand(M, K).clone()
    for k in range(K):
        noise = 0.05 * torch.randn(M, device=device)
        P[:, k] = (P[:, k] + noise).clamp(EPS, MAX_EPS)

    # Initialize Q from PCA coordinates
    PC = U * S.unsqueeze(0)                                   # (N, rank)
    PC_min, PC_max = PC.min(0).values, PC.max(0).values
    PC_n = (PC - PC_min) / (PC_max - PC_min + EPS)           # (N, rank) in [0,1]
    if rank >= K - 1:
        Q_raw = torch.cat([PC_n[:, :K-1],
                           torch.ones(N, 1, device=device)], dim=1)
    else:
        Q_raw = torch.cat([PC_n, torch.ones(N, K - rank, device=device)], dim=1)
    Q_raw = Q_raw + 0.01 * torch.rand(N, K, device=device)
    Q = Q_raw / Q_raw.sum(dim=1, keepdim=True).clamp(min=EPS)

    # ALS: alternate Q and P updates using H_hat ≈ (U*S @ Vt + 2f)/2
    # For large N×M, H_hat materialization also OOMs → use streaming ALS
    if use_streaming:
        chunk_m = 4096
        for _ in range(als_iters):
            # Q update: Q ∝ H_hat @ P  (accumulate in chunks)
            QtP = torch.zeros(N, K, device=device)
            for j in range(0, M, chunk_m):
                jj = min(j + chunk_m, M)
                H_b = ((U * S.unsqueeze(0)) @ Vt[:, j:jj] + 2.0 * f[j:jj]) / 2.0
                H_b = H_b.clamp(EPS, MAX_EPS)
                QtP.add_(H_b @ P[j:jj])
            PtP = P.T @ P
            Q_raw = QtP @ torch.linalg.pinv(PtP + 1e-6 * torch.eye(K, device=device))
            Q_raw = Q_raw.clamp(min=0.0)
            Q = Q_raw / Q_raw.sum(dim=1, keepdim=True).clamp(min=EPS)

            # P update: P ∝ H_hat.T @ Q  (accumulate in chunks)
            HtQ = torch.zeros(M, K, device=device)
            for j in range(0, M, chunk_m):
                jj = min(j + chunk_m, M)
                H_b = ((U * S.unsqueeze(0)) @ Vt[:, j:jj] + 2.0 * f[j:jj]) / 2.0
                H_b = H_b.clamp(EPS, MAX_EPS)
                HtQ[j:jj].add_(H_b.T @ Q)
            QtQ = Q.T @ Q
            P = (HtQ @ torch.linalg.pinv(QtQ + 1e-6 * torch.eye(K, device=device))).clamp(EPS, MAX_EPS)
    else:
        H_hat = ((U * S.unsqueeze(0)) @ Vt + 2.0 * f.unsqueeze(0)) / 2.0
        H_hat = H_hat.clamp(EPS, MAX_EPS)
        for _ in range(als_iters):
            PtP = P.T @ P
            Q_raw = H_hat @ P @ torch.linalg.pinv(PtP + 1e-6 * torch.eye(K, device=device))
            Q_raw = Q_raw.clamp(min=0.0)
            Q = Q_raw / Q_raw.sum(dim=1, keepdim=True).clamp(min=EPS)
            QtQ = Q.T @ Q
            P = (H_hat.T @ Q @ torch.linalg.pinv(QtQ + 1e-6 * torch.eye(K, device=device))).clamp(EPS, MAX_EPS)
        del H_hat

    return Q, P


# ─────────────────────────────────────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────────────────────────────────────

def project_simplex(Q: torch.Tensor) -> torch.Tensor:
    """
    Project each row of Q onto the probability simplex sum=1, all>=0.
    Uses the efficient algorithm from Duchi et al. (2008).
    """
    N, K = Q.shape
    # Sort in descending order
    u, _ = torch.sort(Q, dim=1, descending=True)
    cssv = u.cumsum(dim=1)
    rho = (u * torch.arange(1, K + 1, device=Q.device, dtype=Q.dtype).unsqueeze(0)
           > (cssv - 1)).long().sum(dim=1) - 1  # (N,)
    rho = rho.clamp(0, K - 1)
    theta = (cssv.gather(1, rho.unsqueeze(1)).squeeze(1) - 1.0) / (rho.float() + 1.0)
    return (Q - theta.unsqueeze(1)).clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_log_likelihood(G: torch.Tensor, Q: torch.Tensor, P: torch.Tensor,
                            batch_size: int = 10000) -> float:
    """
    Compute binomial log-likelihood: sum_ij [G*log(H) + (2-G)*log(1-H)]
    Processed in SNP batches to avoid OOM.
    """
    N, M = G.shape
    K = Q.shape[1]
    ll = 0.0
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        G_b = G[:, start:end]    # (N, B)
        P_b = P[start:end, :]    # (B, K)
        H_b = (Q @ P_b.T).clamp(EPS, MAX_EPS)  # (N, B)
        ll += (G_b * torch.log(H_b) + (2.0 - G_b) * torch.log(1.0 - H_b)).sum().item()
    return ll


# ─────────────────────────────────────────────────────────────────────────────
# Single EM step (over a batch of SNPs)
# ─────────────────────────────────────────────────────────────────────────────

def em_step_batch(G_b: torch.Tensor, Q: torch.Tensor, P_b: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One EM step on a batch of SNPs.

    M-step updates derived from FRAPPE/ADMIXTURE (Tang et al. 2005; Alexander et al. 2009):

      P_new[j,k] = P[j,k] * C_minor[j,k] / (P[j,k]*C_minor[j,k] + (1-P[j,k])*C_major[j,k])
      Q_new[i,k] ∝ Q[i,k] * sum_j [P[j,k]*R_minor[i,j] + (1-P[j,k])*R_major[i,j]]

    where C_minor = R_minor.T @ Q,  C_major = R_major.T @ Q

    Args:
        G_b: (N, B) float32 genotype matrix for B SNPs
        Q:   (N, K) current admixture proportions
        P_b: (B, K) current allele frequencies for the batch

    Returns:
        Q_num_b: (N, K) numerator contribution for Q update
        P_num_b: (B, K) numerator for P update   (P * C_minor)
        P_den_b: (B, K) denominator for P update (P * C_minor + (1-P) * C_major)
    """
    # H[i,j] = Q @ P_b.T   (N, B)
    H_b = (Q @ P_b.T).clamp(EPS, MAX_EPS)

    R_minor = G_b / H_b                   # (N, B)  = G / H
    R_major = (2.0 - G_b) / (1.0 - H_b)  # (N, B)  = (2-G)/(1-H)

    # Q numerator (standard EM M-step):
    # Q_num[i,k] = Q[i,k] * sum_j [P[j,k]*R_minor[i,j] + (1-P[j,k])*R_major[i,j]]
    Q_num_b = Q * (R_minor @ P_b + R_major @ (1.0 - P_b))  # (N, K)

    # P numerator/denominator (ADMIXTURE M-step):
    # P_num = P * C_minor;  P_den = P * C_minor + (1-P) * C_major
    C_minor_b = R_minor.T @ Q   # (B, K)
    C_major_b = R_major.T @ Q   # (B, K)
    P_num_b = P_b * C_minor_b                              # (B, K)
    P_den_b = P_num_b + (1.0 - P_b) * C_major_b           # (B, K)

    return Q_num_b, P_num_b, P_den_b


# ─────────────────────────────────────────────────────────────────────────────
# EM iteration (full-batch or stochastic mini-batch over SNPs)
# ─────────────────────────────────────────────────────────────────────────────

def em_iteration(G: torch.Tensor, Q: torch.Tensor, P: torch.Tensor,
                 n_batches: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One EM epoch supporting two modes determined by n_batches:

    n_batches == 1  → Full-batch EM (standard, guaranteed monotone LL increase)
    n_batches > 1   → Stochastic mini-batch EM:
        • Shuffle M SNPs, split into n_batches chunks
        • For each batch: compute Q_num from that batch only, immediately update Q
        • Accumulate P_num / P_den across all batches using the current (changing) Q
        • Update P once after all batches

    The stochastic mode effectively performs n_batches Q-update steps per epoch,
    each using 1/n_batches of the data. Empirically this converges 1.5–2× faster
    than full-batch EM at the same wall-clock time, and the stochastic noise helps
    escape suboptimal local optima.

    P update:
        P_new[j,k] = P[j,k] * C_minor[j,k] / (P[j,k]*C_minor + (1-P[j,k])*C_major)
        (FRAPPE/ADMIXTURE M-step — exact MLE update for P given Q)

    Q update per batch:
        Q_new[i,:] = row_normalize(Q[i,:] * (R_minor_b @ P_b + R_major_b @ (1-P_b)))
        (standard EM M-step — row normalization, NOT L2 simplex projection)
    """
    N, M = G.shape
    device = G.device

    if n_batches == 1:
        # Full-batch: single call, no shuffling overhead
        Q_num, P_num, P_den = em_step_batch(G, Q, P)
        Q_new = Q_num / Q_num.sum(dim=1, keepdim=True).clamp(min=EPS)
        P_new = (P_num / P_den.clamp(min=EPS)).clamp(EPS, MAX_EPS)
        return Q_new, P_new

    # Stochastic mini-batch mode
    batch_size = (M + n_batches - 1) // n_batches
    perm = torch.randperm(M, device=device)

    P_num = torch.zeros_like(P)
    P_den = torch.zeros_like(P)

    for b in range(n_batches):
        idx = perm[b * batch_size: min((b + 1) * batch_size, M)]
        G_b = G[:, idx]
        P_b = P[idx, :]

        Q_num_b, P_num_b, P_den_b = em_step_batch(G_b, Q, P_b)

        # Immediately update Q with this batch's information (stochastic EM)
        Q = Q_num_b / Q_num_b.sum(dim=1, keepdim=True).clamp(min=EPS)

        P_num[idx] += P_num_b
        P_den[idx] += P_den_b

    # Full P update after epoch (using final Q)
    P_new = (P_num / P_den.clamp(min=EPS)).clamp(EPS, MAX_EPS)
    return Q, P_new


# ─────────────────────────────────────────────────────────────────────────────
# Main fitting function
# ─────────────────────────────────────────────────────────────────────────────

def _default_n_batches(M: int) -> int:
    """
    Default mini-batch count: target ~12,500 SNPs per batch.
    Larger datasets get more batches for memory efficiency and faster convergence.
    Full-batch (n_batches=1) is used when M <= 12500.
    """
    return max(1, round(M / 12500))


@torch.no_grad()
def fit(G: torch.Tensor, K: int, seed: int = 42,
        n_iter: int = 500, n_batches: int = -1,
        tol: float = 1e-5, check_interval: int = 10,
        use_nesterov: bool = True, verbose: bool = True
        ) -> dict:
    """
    Fit gpuADMIX model.

    Args:
        G:             (N, M) float32 genotype matrix on GPU
        K:             number of ancestral populations
        seed:          random seed for initialization
        n_iter:        maximum EM epochs (default 500)
        n_batches:     SNP mini-batches per epoch (-1 = auto: M/12500).
                       1 = full-batch EM (guaranteed monotone LL).
                       >1 = stochastic mini-batch EM (faster convergence,
                            ~1.5–2× fewer epochs, slight stochastic noise).
        tol:           convergence tolerance on relative LL change (default 1e-5)
        check_interval: check convergence every this many epochs
        use_nesterov:  apply Nesterov momentum to EM iterates
        verbose:       print progress

    Returns:
        dict with Q, P, log_likelihoods, n_iter_run, elapsed_sec
    """
    t0 = time.time()
    N, M = G.shape

    if n_batches == -1:
        n_batches = _default_n_batches(M)

    if verbose:
        print(f"gpuADMIX: N={N}, M={M}, K={K}, device={G.device}, seed={seed}")
        print(f"  n_batches={n_batches} (~{M//max(n_batches,1):,} SNPs/batch), "
              f"nesterov={use_nesterov}, max_iter={n_iter}")

    # Initialize
    Q, P = svd_als_init(G, K, seed=seed)
    Q = Q.to(G.device)
    P = P.to(G.device)

    if verbose:
        print(f"  Init done ({time.time()-t0:.1f}s)")

    # Nesterov momentum state
    Q_prev = Q.clone()
    P_prev = P.clone()
    beta = 1.0  # Nesterov sequence parameter

    log_likelihoods = []
    ll_prev = -torch.inf

    # Compute initial LL
    ll_init = compute_log_likelihood(G, Q, P)
    log_likelihoods.append((0, ll_init))
    if verbose:
        print(f"  iter    0  ll={ll_init:.4f}  (init)")

    nesterov_resets = 0

    for it in range(1, n_iter + 1):
        if use_nesterov and it > 1:
            # Nesterov FISTA-style lookahead
            beta_new = (1.0 + math.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
            gamma = (beta - 1.0) / beta_new

            Q_look = project_simplex(Q + gamma * (Q - Q_prev))
            P_look = (P + gamma * (P - P_prev)).clamp(EPS, MAX_EPS)
        else:
            Q_look = Q
            P_look = P

        # EM step at lookahead point
        Q_new, P_new = em_iteration(G, Q_look, P_look, n_batches=n_batches)

        # Nesterov safety check: if LL decreased, fall back to plain EM step
        if use_nesterov and it > 1:
            ll_new = compute_log_likelihood(G, Q_new, P_new)
            ll_plain_test = compute_log_likelihood(G, Q, P)
            if ll_new < ll_plain_test - 1.0:
                # Restart: do plain EM from current Q, P
                Q_new, P_new = em_iteration(G, Q, P, n_batches=n_batches)
                beta = 1.0
                Q_prev = Q_new.clone()
                P_prev = P_new.clone()
                nesterov_resets += 1
            else:
                beta = (1.0 + math.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
                Q_prev_old = Q_prev
                P_prev_old = P_prev
                Q_prev = Q.clone()
                P_prev = P.clone()
        elif it == 1:
            Q_prev = Q.clone()
            P_prev = P.clone()
            beta_new = (1.0 + math.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
            beta = beta_new

        Q = Q_new
        P = P_new

        # Convergence check
        if it % check_interval == 0 or it == n_iter:
            ll = compute_log_likelihood(G, Q, P)
            log_likelihoods.append((it, ll))

            if verbose:
                elapsed = time.time() - t0
                print(f"  iter {it:4d}  ll={ll:.4f}  dt={elapsed:.1f}s")

            if it > check_interval and abs(ll - ll_prev) / (abs(ll_prev) + 1e-10) < tol:
                if verbose:
                    print(f"  Converged at iteration {it} (nesterov_resets={nesterov_resets})")
                break
            ll_prev = ll

    elapsed = time.time() - t0
    if verbose:
        print(f"gpuADMIX done: {elapsed:.1f}s, {it} iterations")

    return {
        'Q': Q.cpu().numpy(),
        'P': P.cpu().numpy(),
        'log_likelihoods': log_likelihoods,
        'n_iter_run': it,
        'elapsed_sec': elapsed,
        'K': K,
        'seed': seed,
    }
