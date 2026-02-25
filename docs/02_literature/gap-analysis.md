# Gap Analysis: Admixture Population Structure Inference
Generated: 2026-02-25

## Gap 1: No GPU-accelerated model-based admixture method
Description: All model-based methods (ADMIXTURE, fastmixture) are CPU-only. The only GPU method (Neural ADMIXTURE) is an autoencoder that sacrifices accuracy.
Already solved? NO. fastmixture (2024) is state-of-the-art but explicitly CPU-only with no GPU plans.
Why unsolved? Model-based EM involves sequential per-sample and per-variant updates that are harder to parallelize on GPU than matrix ops.
How to approach: Reformulate EM update as batch matrix operations (GEMM). Use 2-bit packed genotype representation on GPU VRAM. Multi-GPU via splitting N or M dimension.
Confidence: HIGH (confirmed by reading fastmixture v2 paper which says "CPU-based setup only")

## Gap 2: Accuracy-speed tradeoff is unresolved for likelihood-free methods
Description: SCOPE and NMF methods are fast but noisy. This noise worsens with larger K and complex demography.
Already solved? fastmixture partially solves it for CPU. Still open for GPU scale.
Confidence: HIGH

## Gap 3: No method benchmarked beyond ~30x speedup vs ADMIXTURE with model-based accuracy
Description: fastmixture achieves ~30x on 64 CPU cores. No method has achieved 100x+ while maintaining accuracy.
Confidence: HIGH (confirmed by fastmixture paper which explicitly identifies this as limitation)

## Gap 4: K selection (choosing number of ancestral populations) is still slow
Description: Running K=2..20 requires 19 separate runs even with fastmixture. GPU could enable parallel K sweeps.
Already solved? Partially by Neural ADMIXTURE multi-head (but inaccurate). Not solved accurately.
Confidence: HIGH

## Gap 5: Multi-GPU distributed admixture for truly biobank-scale data
Description: fastmixture 74 min for 1kGP (3K samples, 6.8M SNPs). UK Biobank (500K x 1M SNPs) still challenging.
Already solved? SCOPE runs UK Biobank but sacrifices accuracy. No accurate method scales to >100K samples.
Confidence: HIGH

## Gap 6: Reproducibility/initialization robustness
Description: ADMIXTURE noted to find suboptimal solutions due to random initialization (fastmixture v2 shows this for Scenario C). fastmixture SVD initialization improves this.
Already solved? fastmixture improves; GPU could enable multiple restarts quickly.
Confidence: MEDIUM

## Most Important Gap
Gap 1 is the primary research target:
"No existing method achieves GPU-accelerated model-based ancestry estimation with ADMIXTURE-level accuracy"
This is confirmed, novel, and directly actionable given the user's multi-GPU server.
