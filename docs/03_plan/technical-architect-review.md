# Technical Architect Review: gpuADMIX Implementation Design
**Reviewer Role:** Technical Architect for Bioinformatics journal submission  
**Date:** 2026-02-25  
**Scope:** GPU PyTorch implementation design for binomial admixture EM

---

## 1. IMPLEMENTABILITY

**Verdict: CLEAN to code with minor ambiguities**

### Strengths
- **Well-defined GEMM mapping:** Steps 1, 4, 5 map directly to `torch.matmul`; steps 2, 3, 6 to element-wise ops. No obscure control flow.
- **Standard PyTorch primitives:** `torch.svd_lowrank`, `torch.matmul`, `clamp`, `normalize` — all readily available.
- **Multi-GPU design is simple:** No inter-GPU communication; each device runs independent K.

### Ambiguous Design Choices

| Item | Ambiguity | Recommendation |
|------|-----------|----------------|
| **ALS initialization** | "ALS to get Q, P" — unclear: ALS for NMF vs. admixture-constrained ALS? | Specify: non-negative ALS with simplex constraint on Q, `[eps,1-eps]` on P. Use `torch.linalg.lstsq` or projected gradient. |
| **Stabilization pass** | Step 2c "Full EM step over all SNPs" — does this update both Q and P, or only P? | **Must update both Q and P.** Current design implies Q updates B times (batch loop) then P updates once (full pass). Clarify: full pass = one complete Q,P EM update over full G. |
| **Nesterov order** | Step 2d: extrapolate before or after the stabilization pass? | **Extrapolate after** full EM: θ_{t+1} = Project(EM(θ_{t+0.5})) where θ_{t+0.5} = θ_t + β(θ_t - θ_{t-1}). |
| **Bootstrap block size** | "LD_block_size" — not specified. | Use 10 cM (ADMIXTURE default) or LD-based blocks from PLINK; document in config. |

---

## 2. KEY TECHNICAL CHALLENGES

### 2.1 GEMM-Based EM Formulation — Mathematical Correctness

**Question:** Is the formula `Q_num = Q * (R_minor @ P + R_major @ (1-P))` correct, including the element-wise multiply by Q BEFORE normalization?

**Answer: YES**

The formula matches the standard FRAPPE/ADMIXTURE binomial EM update. Derivation:

- Model: G_ij ~ Bin(2, H_ij), H_ij = Σ_k q_ik p_jk = (Q @ P^T)[i,j]
- E-step: posterior over latent ancestry assignments
- M-step: q_ik^{new} ∝ q_ik^{old} × Σ_j [ (G_ij/H_ij)·p_jk + ((2-G_ij)/(1-H_ij))·(1-p_jk) ]

In matrix form:
- R_minor = G / H, R_major = (2-G) / (1-H)
- (R_minor @ P)[i,k] = Σ_j (G_ij/H_ij) p_jk
- (R_major @ (1-P))[i,k] = Σ_j ((2-G_ij)/(1-H_ij)) (1-p_jk)
- Q_num[i,k] = q_ik × (sum of above) = Q * (R_minor @ P + R_major @ (1-P))[i,k]

The multiplicative factor Q is correct — it is the current iterate used in the EM multiplicative update. Normalize Q_num row-wise to obtain Q_new.

**P update:** P_num = (R_minor^T @ Q) * P, then truncate to [eps, 1-eps] and optionally renormalize per SNP if using a different normalization.

### 2.2 Nesterov Extrapolation with Simplex Constraints

**Problem:** θ_ahead = θ + β(θ - θ_prev) can produce points outside the simplex (Q) or [0,1] (P).

**Risks:**
- **Simplex:** Linear combination of two simplex points is not on the simplex. E.g. (0.5, 0.5) + 0.5×((0.5,0.5)-(0.8,0.2)) = (0.35, 0.65) — OK. But (0.9, 0.1) + 2×((0.9,0.1)-(0.1,0.9)) = (2.5, -1.5) — invalid.
- **Projection:** Clipping negatives to 0 and renormalizing can collapse to a vertex (all mass on one component) if extrapolation is extreme.
- **EM at invalid point:** The design says "EM at θ_ahead" — but θ_ahead is projected before EM. So the flow is: θ_ahead = Project(θ + β(θ-θ_prev)), then θ_new = Project(EM(θ_ahead)). Projection before EM is correct.

**Mitigation:**
1. **Clamp β:** Use β ∈ [0, 1] or adaptive β that reduces when ||θ_ahead - θ|| is large.
2. **Reject bad extrapolations:** If Project(θ_ahead) differs significantly from θ_ahead (e.g. L1 norm of projection residual > threshold), skip momentum and use plain EM.
3. **Fallback:** If LL decreases after 2 consecutive momentum steps, reset β = 0 and continue with plain EM.

### 2.3 Batch-P Update Consistency

**Problem:** Q updates B times per cycle (once per batch); P updates once per full cycle. Is this statistically sound?

**Analysis:**
- **fastmixture:** Uses the same pattern (lit review: "Q updated B times per cycle; P updated once per full cycle"). Empirically validated.
- **Interpretation:** This is block coordinate descent with Q as the "fast" block and P as the "slow" block. Q is updated with partial data (B mini-batches); P is updated with full data. The stabilization pass (full EM) corrects any bias.
- **Risk:** Q can drift toward a local optimum that suits the stale P. If the full pass is done correctly, both get one proper full-data update per cycle.

**Recommendation:** Ensure the full EM step (2c) updates **both** Q and P over the full G. The batch loop is an acceleration heuristic; the full pass is the "correct" EM step. Document this as "stochastic block coordinate descent with full correction."

---

## 3. STABILITY

### Float32 vs Float16

| Precision | Recommendation |
|-----------|----------------|
| **float32** | **Required** for core EM. H, R_minor, R_major, Q_num, P_num can have values near 0 or 1; float16 has ~3 decimal digits and will underflow/overflow. |
| **float16** | Avoid for EM. Optional for G storage if using 2-bit decompressed to int8; never use float16 for H or division. |
| **float64** | Optional for log-likelihood accumulation to avoid rounding error in long sums. |

### Numerical Issues

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **H near 0** | R_minor = G/H → inf when H→0, G>0 | `clamp(H, eps, 1-eps)` with eps ≈ 1e-6 |
| **H near 1** | R_major = (2-G)/(1-H) → inf when H→1, G<2 | Same clamp |
| **1-H underflow** | When H≈1, 1-H can underflow in float32 | Use `clamp(1-H, eps, 1)` or equivalently clamp H |
| **Q_num = 0** | If Q[i,k]=0 and numerator=0, row stays 0 | Normalize only non-zero rows; or add eps to Q_num before normalize |

**Recommended epsilon:** `eps = 1e-6` for H and (1-H). Test against ADMIXTURE (float64) to ensure no systematic bias.

---

## 4. RESOURCE FIT

### Memory Analysis (float32)

| Component | N=3202, M=200K | N=10K, M=200K |
|-----------|----------------|---------------|
| G (int8) | 3202×200K×1 = 640 MB | 10K×200K×1 = 2 GB |
| G (2-bit BED) | 3202×200K/4 = 160 MB | 500 MB |
| H_b (batch) | 3202×6250×4 = 80 MB | 10K×6250×4 = 250 MB |
| Q | 3202×10×4 = 128 KB | 400 KB |
| P | 200K×10×4 = 8 MB | 8 MB |
| R_minor_b, R_major_b | 2×80 MB = 160 MB | 500 MB |
| **Peak (batch)** | ~1 GB | ~3.5 GB |
| **A100 40GB** | ✓ Comfortable | ✓ |
| **V100 80GB** | ✓ | ✓ |

**N=10K scaling:** Batch size 6250 keeps H_b = 250MB. If 4× larger N, consider reducing batch size (e.g. 3125 SNPs) to keep H_b < 500MB per batch for multi-GPU memory headroom.

### GEMM Sizes

- Q @ P_b.T: (3202×10) × (10×6250) = 3202×6250 — moderate GEMM
- R_minor_b @ P_b: (3202×6250) × (6250×10) — larger GEMM, good for GPU
- R_minor^T @ Q: (200K×3202) × (3202×10) — very large for full pass; must be done in chunks or batch

**Note:** Full P update requires R_minor^T @ Q over full M. For M=200K, this is 200K×3202 × 3202×10. The first matrix is 200K×3202×4 = 2.5GB. This may exceed GPU memory if materialized. **Solution:** Compute P update in SNP batches: for each batch of SNPs, compute R_minor_b^T @ Q, update P_b. Same batching as Q update.

---

## 5. FAILURE MODE ANALYSIS

| Rank | Failure Mode | Probability | Mitigation |
|------|--------------|-------------|------------|
| **1** | **Numerical instability (H near 0/1)** | High | Robust clamping: eps=1e-6; validate against ADMIXTURE on same data; add unit tests for edge cases (G=0,1,2; H near boundaries). |
| **2** | **Nesterov divergence or oscillation** | Medium | Implement rejection: if LL decreases, skip momentum. Fallback to plain EM. Ablation: compare with/without Nesterov. |
| **3** | **Mini-batch Q drift / stale P** | Medium | Ensure full EM pass updates both Q and P. Consider B=16 or B=8 for stability if B=32 causes LL fluctuation. Adaptive B (halve when fluctuating) is already in design. |

**Additional:** Bootstrap on 100 replicates × 4–8 GPUs = 12–25 replicates per GPU. Ensure sufficient GPU memory for multiple streams if running bootstrap in parallel.

---

## 6. SUGGESTED IMPROVEMENTS

### 6.1 Cleaner EM Implementation (Addresses Numerical Issues)

**Option A: Stabilized multiplicative update (recommended)**

```python
eps = 1e-6
H_safe = H.clamp(eps, 1 - eps)
R_minor = (G + eps) / (H_safe + eps)  # Avoid 0/0 when G=0
R_major = (2 - G + eps) / (1 - H_safe + eps)
Q_num = Q * (R_minor @ P + R_major @ (1 - P))
Q_num = Q_num.clamp(eps, 1 - eps)  # Before normalize, avoid exact zeros
Q_new = Q_num / Q_num.sum(dim=1, keepdim=True)
```

**Option B: Log-domain (for extreme stability)**

Work in log-space for the ratio to avoid overflow: `log(R_minor) = log(G+eps) - log(H+eps)`. Then `R_minor @ P` becomes a log-sum-exp style computation. More complex; consider only if Option A fails.

**Option C: Sufficient statistics accumulation**

For full-batch update, accumulate:
- numerator_Q += R_minor @ P + R_major @ (1-P)  [per batch]
- denominator handled by normalization

This avoids materializing full R_minor, R_major. For mini-batch Q-only updates, keep current design; for full P update, accumulate over batches.

### 6.2 P Update Batching

The full P update `P_num = (R_minor^T @ Q) * P` requires R_minor^T @ Q. R_minor is N×M — too large. Use:
```
For each SNP batch b:
  P_num_b = (R_minor_b.T @ Q) * P_b
  P_b = clamp(P_num_b / denom_b, eps, 1-eps)
```
Denom for P: typically column sum of Q-weighted contributions. For binomial model, the denominator for P[j,k] is Σ_i q_ik (expected count). So: `P_new = P_num / (Q^T @ ones)` or similar. Check ADMIXTURE/FRAPPE for exact P normalization.

### 6.3 Initialization Robustness

SVD on centered G: `G_centered = G - 2*row_mean` or `G - 2*Q_init@P_init`? Standard is center by mean allele frequency per SNP. `G_centered = G - 2 * (G.sum(0)/(2*N))` — column means. Use `torch.svd_lowrank(G_centered)` with rank=K-1.

---

## 7. VERDICT

### CONDITIONAL PASS

**Conditions for approval:**

1. **Clarify and implement:** (a) Full EM stabilization pass updates both Q and P; (b) Nesterov extrapolation order (extrapolate → project → EM → project); (c) P update batching for full pass.
2. **Numerical robustness:** Implement Option A (stabilized multiplicative update) with eps=1e-6; validate against ADMIXTURE on 1kGP_200K.
3. **Fallback:** Implement plain EM (no Nesterov) as default or flag; Nesterov as optional acceleration. Document ablation in supplement.

**Strengths:** GEMM formulation is correct; design is implementable; multi-GPU and bootstrap are straightforward; resource fit is good for target data.

**Risks:** Numerical stability and Nesterov behavior need empirical validation. Mitigations are clear and low-cost.

---

## 8. SPECIFIC ANSWER: Cleaner EM Implementation

**Is there a cleaner way to implement the EM updates on GPU that avoids potential numerical issues?**

**Yes.** Use the **stabilized multiplicative update** (Option A above):

1. **Clamp H and (1-H):** `H_safe = H.clamp(1e-6, 1-1e-6)` before any division.
2. **Stabilize numerator:** Add small eps to G and (2-G) in the numerator of R_minor, R_major to avoid 0/0 when G=0 or G=2.
3. **Clamp Q_num before normalize:** `Q_num.clamp(1e-8, 1e8)` to avoid inf/nan; then normalize.
4. **Use float32 throughout:** Do not use float16 for H, R, or division.
5. **Fused kernel (optional):** A single CUDA kernel that computes H_b, R_minor_b, R_major_b, and Q_num_b in one pass avoids materializing intermediate R matrices in memory — but PyTorch's element-wise ops are already efficient; only consider if profiling shows memory bottleneck.

The algebraic structure is correct; the main improvement is numerical safeguarding via clamping and small additive constants. This is standard practice in numerical EM implementations (e.g. scikit-learn's GMM).
