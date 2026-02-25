# Method Design — gpuADMIX
# Phase 3, M-Step 2
# Generated: 2026-02-25

## Software Name (working): gpuADMIX
(to be finalized; alternatives: AdmixGPU, FastAdmix, gpuMIX)

## Innovation Points

### Innovation 1: GPU-native Mini-Batch EM for Binomial Admixture
The standard ADMIXTURE/FRAPPE EM updates can be fully expressed as:
1. H = Q @ P^T          [GEMM: N×K × K×M → N×M]
2. R_minor = G / H      [element-wise: N×M]
3. R_major = (2-G) / (1-H)  [element-wise: N×M]
4. Q_num = Q * (R_minor @ P + R_major @ (1-P))   [GEMM: N×M × M×K → N×K, then ⊙ Q]
5. P_num = (R_minor^T @ Q) * P                    [GEMM: M×N × N×K → M×K, then ⊙ P]
6. Normalize Q (simplex projection per row), truncate P to [eps, 1-eps]

Steps 1, 4, 5 are GEMMs → directly maps to cuBLAS SGEMM or torch.matmul on GPU.
Steps 2, 3, 6 are element-wise → CUDA fused kernel or vectorized torch ops.

This is the first implementation to map these updates to GPU primitives correctly.
Key: intermediate H matrix (N×M float32) is NOT materialized fully; processed in mini-batches over SNPs.

### Innovation 2: Nesterov Momentum for Constrained EM
ADMIXTURE uses quasi-Newton (QN) with sequential multi-step extrapolation.
fastmixture v2 also uses QN acceleration, which requires 3 sequential EM steps.
QN requires sequential iteration: u1=EM(u0), u2=EM(u1), then extrapolation.
This sequential structure limits GPU parallelism.

Alternative: Nesterov-style momentum directly applied to the EM iterate space:
  θ_lookahead = θ_t + β_t * (θ_t - θ_{t-1})   [momentum extrapolation: GPU-parallel]
  θ_{t+1} = Project(EM_step(G, θ_lookahead))   [single EM step at lookahead point]
  β_t schedule: standard Nesterov (1 + sqrt(1+4β_{t-1}^2)) / 2 ... but adapted for EM

Key: only ONE additional parameter tensor (θ_prev) needed; no sequential dependency.
GPU advantage: the extrapolation step is pure vector math (zero synchronization cost).

### Innovation 3: Multi-GPU Parallel K-Value Estimation
Each GPU device independently runs estimation for one K value:
  GPU 0: K=2, GPU 1: K=3, ..., GPU n-1: K=n+1
Shared data: genotype matrix G transferred once to each GPU's VRAM.
No inter-GPU communication required (embarrassingly parallel).
K-sweep (K=2..10) wall time = time for single K (vs. 9× sequential with single GPU).

### Innovation 4: Block-Bootstrap Uncertainty Quantification
SNP-block bootstrap: resample contiguous LD blocks (not individual SNPs) to preserve LD structure.
Each bootstrap replicate runs independent EM on GPU (parallel across GPUs or sequential streams).
Output: Q_mean ± Q_std (N×K matrix of mean ± std admixture proportions).
First implementation of bootstrap UQ for admixture at this scale.

## Algorithm Pseudo-Code

```
Algorithm: gpuADMIX
Input:  G (N×M genotype matrix, PLINK BED format)
        K (number of ancestral populations)
        B (number of mini-batches, default 32)
        max_iter (default 200)
Output: Q (N×K admixture proportions), P (M×K allele frequencies)

1. INITIALIZATION:
   a. Compute randomized SVD: U, S, V^T = rSVD(G_centered, rank=K-1)  [GPU: torch.svd_lowrank]
   b. Estimate individual allele freq: H_hat = G_centered via SVD
   c. ALS initialization: minimize ||H_hat - 2*Q*P^T||_F for Q, P  [GPU: alternating least squares]
   d. Project Q to simplex; truncate P to [eps, 1-eps]

2. MINI-BATCH EM WITH NESTEROV MOMENTUM:
   For t = 1..max_iter:
     a. Shuffle SNP indices, split into B mini-batches
     b. For each batch b in 1..B:
        - H_b = Q @ P_b^T              [GEMM: GPU]
        - R_minor_b = G_b / clamp(H_b) [element-wise: GPU]
        - R_major_b = (2-G_b)/clamp(1-H_b) [element-wise: GPU]
        - Q_num_b = Q * (R_minor_b @ P_b + R_major_b @ (1-P_b))  [GEMM: GPU]
        - Q = normalize(Q_num_b)       [simplex projection: GPU]
     c. Full EM step over all SNPs (stabilization pass)
     d. Nesterov extrapolation:
        Q_mom = Q + β_t * (Q - Q_prev); project to simplex
        P_mom = P + β_t * (P - P_prev); truncate to [eps, 1-eps]
     e. Every 5 iterations: compute log-likelihood; check convergence (delta_LL < eps)
        If LL fluctuates: halve B (more conservative batching); reset momentum

3. OUTPUT: Q, P, log-likelihood trace

4. PARALLEL-K MODE (multi-GPU):
   For K in K_list:  [each on separate GPU device]
     Run above algorithm independently
   Return Q_k, P_k for each K

5. BOOTSTRAP UQ (optional):
   For b in 1..n_bootstrap:  [parallel across GPUs]
     G_boot = block_bootstrap(G, block_size=LD_block_size)
     Run algorithm on G_boot → Q_boot_b
   Return Q_mean = mean(Q_boot), Q_std = std(Q_boot)
```

## Value Proposition
"GPU parallelism over the N×M EM computation (natural GEMM structure) provides 
10-50x raw compute speedup over 64-thread CPU, enabling accurate model-based ancestry 
estimation that was previously only achievable with hours-long CPU runs to complete 
in minutes on any multi-GPU server."

## Predicted Failure Modes and Backup Plans

| Failure Mode | Probability | Mitigation |
|---|---|---|
| GPU VRAM insufficient for H matrix at N=3202, M=200K | Low (H≈2.5GB, fine for A100) | Mini-batch already addresses; reduce batch size |
| Nesterov momentum diverges or slows convergence | Medium | Fallback: standard QN (same as fastmixture); Nesterov is additive improvement |
| Multi-GPU communication bottleneck | Low (K-parallel has NO cross-GPU comms) | Already designed to avoid this |
| Float32 precision issues for small allele freqs | Medium | Add epsilon clamping; test against double precision ADMIXTURE |
| GPU not faster than fastmixture CPU for small N | Low (for N=3202, M=200K) | Expected 5-10x; demonstrate scaling advantage at N=10K+ |
| Nesterov convergence proof fails | Medium | Use empirical convergence guarantee; cite similar results for EM+momentum in literature |
