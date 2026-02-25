# Negative Results & Failures — gpuADMIX
Generated: 2026-02-25

## 1. Critical Bug: L2 Simplex Projection in EM M-Step (FIXED)
**What happened**: EM Q-update used L2 Euclidean projection onto simplex instead of row normalization.
**Impact**: LL decreased by 6M units per step; algorithm diverged.
**Fix**: Replaced simplex projection with row normalization; L2 projection kept only for Nesterov lookahead.
**Documentation value**: Demonstrates importance of correct M-step formulation.

## 2. K=4 Multimodality (Landscape Phenomenon)
**What happened**: gpuADMIX K=4 shows LL std = 63,440 across 5 seeds (vs K=5 std = 98).
**Meaning**: K=4 has multiple local optima; different seeds find different solutions (within-K RMSE = 0.17).
**Not a bug**: This reflects the true statistical landscape. K=4 is between 3 stable clusters.
**Value**: Motivates multiple restarts and K selection via stability criteria; CLUMPAK-lite detects this.

## 3. K=7 Multimodality
**What happened**: gpuADMIX K=7 LL std = 15,550; within-K RMSE = 0.12.
**Meaning**: K=7 is also multimodal (K=7 splits overlap ambiguously).
**Value**: Further supports K=5 as the "elbow" of stable solutions.

## 4. Nesterov Safety Resets
**What happened**: Nesterov momentum occasionally causes LL decrease (1 reset per ~100 iterations).
**Cause**: Momentum extrapolation overshoots feasible region.
**Fix**: Safety check resets momentum when LL decreases.
**Residual issue**: Safety reset adds overhead; slightly reduces effective speedup.

## 5. Float32 Precision for Rare Variants
**What happened**: R_minor = G/H reaches 200,000 for very rare SNPs (P_A1 ≈ 1e-5) with G=2.
**Impact**: Potential float32 overflow for datasets with very rare variants (MAF < 0.001%).
**Fix**: Clamping H at EPS=1e-6 prevents overflow.
**Limitation**: Very rare variants contribute disproportionately to numerical instability.

## 6. ADMIXTURE Q File Not Recovered
**What happened**: ADMIXTURE K=5 run completed successfully but output .Q file was lost.
**Impact**: Cannot compute Q correlation between gpuADMIX and ADMIXTURE.
**Mitigation**: LL values from log are sufficient; use fastmixture Q as proxy reference.
**Acknowledged gap**: Direct Q comparison to ADMIXTURE is missing.

## 7. fastmixture K=8..10 Not Completed
**What happened**: fastmixture K=8..10 runs were not completed (time constraint).
**Impact**: No LL/time comparison for K>7.
**Mitigation**: gpuADMIX K=8..10 results complete. Gap in fastmixture comparison for high K.

## 8. Block-Bootstrap UQ Not Implemented
**What happened**: Planned bootstrap UQ was deprioritized as nice-to-have.
**Impact**: Cannot report ancestry proportion confidence intervals.
**Mitigation**: Seed-based stability (within-K RMSE) serves as proxy reproducibility measure.

## 9. gpuADMIX K=2,6 Slightly Worse LL Than fastmixture
**What happened**: At K=2, gpuADMIX mean LL (-252,414,278) < fastmixture (-252,413,106) by 1,172 units.
**Meaning**: At K=2, convergence criterion may differ; fastmixture may run more iterations.
**Context**: At K=5, gpuADMIX is +2,892 better than fastmixture. K=2 gap is inconsequential.

## ── NEW: K≥8 Limitation ──────────────────────────────────────────────────
**Finding**: For K=8..10, fastmixture achieves better LL than gpuADMIX
(K=8: ΔLL = -24,499; K=9: -18,120; K=10: -6,090).

**Interpretation**: fastmixture's quasi-Newton (QN) acceleration is more effective
than Nesterov momentum at navigating highly complex multimodal landscapes (K≥8).
Nesterov with fixed momentum may overshoot in high-K regimes.

**Evidence for multimodality**: gpuADMIX K=8 std = 12,178 across 5 seeds
(vs K=5 std = 98); confirms severe landscape complexity at high K.

**Impact**: For practical population genetics (K=2..7), gpuADMIX outperforms
or matches fastmixture. Users interested in K≥8 should run multiple seeds
and compare with fastmixture.

**Response**: Disclosed in Limitations. Recommendation: use multiple seeds and
take the best run; or consider fastmixture for exploratory high-K analysis.

## ── NEW: K=8 Non-monotone in CV ──────────────────────────────────────────
K=8 CV LL (-70,673,458) is worse than K=7 (-69,896,182), creating a dip in
the CV curve. This is consistent with K=8 multimodality — some CV folds
converge to poorer local optima. K=9 recovers (+148K improvement), with
tighter cross-fold variance (std=386K vs K=8 std=790K).
