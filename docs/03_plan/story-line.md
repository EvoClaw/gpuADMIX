# Story Line — gpuADMIX
# Phase 3, M-Step 6
# Generated: 2026-02-25

## Narrative Arc (5-sentence version)

1. MOTIVATION: Accurate ancestry estimation (ADMIXTURE model-based EM) is essential for 
   population genetics and GWAS correction, but ADMIXTURE is prohibitively slow for 
   modern datasets; fastmixture (2024) achieves ~30x CPU speedup but is limited by CPU 
   memory bandwidth saturation, while the only GPU method (Neural ADMIXTURE) has 
   critically poor accuracy due to an algorithmic normalization bug.

2. INSIGHT: The ADMIXTURE/FRAPPE EM updates are dominated by matrix multiplications 
   (GEMM: H = Q@P^T, Q updates, P updates) — exactly the operation GPUs are optimized 
   for; Neural ADMIXTURE's failure was algorithmic (wrong objective + premature 
   convergence bug), NOT a fundamental limitation of GPU hardware for this problem.

3. METHOD: We present gpuADMIX, the first GPU-accelerated model-based admixture tool 
   preserving the exact binomial log-likelihood. Core contributions: (1) GEMM-based GPU 
   mini-batch EM with Nesterov momentum (1 extrapolation step vs QN's 3 sequential 
   steps); (2) multi-GPU parallel K estimation (K=2..10 in single-K wall time); 
   (3) block-bootstrap uncertainty quantification on GPU.

4. EXPECTED RESULTS: gpuADMIX achieves equivalent accuracy to ADMIXTURE and fastmixture 
   (validated by RMSE/JSD vs simulated ground truth), is significantly faster than 
   fastmixture on multi-GPU hardware (5-20x depending on N), and demonstrates that GPU 
   efficiency advantage over CPU threads does not saturate (strong-scaling experiment: 
   fastmixture 64-thread vs gpuADMIX 1-GPU).

5. ANALYSIS PLAN: Ablation study isolates each contribution (Nesterov vs plain EM, 
   SVD init vs random, QN vs Nesterov on GPU, mini-batch vs full-batch); simulation 
   study on SIM_K3 and SIM_K5_ADMIX validates accuracy; 1kGP_200K validates real-data 
   agreement with ADMIXTURE; SIM_LARGE (N=10K) demonstrates scalability advantage; 
   parallel-K experiment shows total K-sweep speedup.

## Paper Sections Outline

1. Introduction (gap, insight, contributions)
2. Methods
   2.1. Admixture model and EM formulation
   2.2. GPU-native EM: GEMM reformulation
   2.3. Nesterov momentum acceleration
   2.4. SVD initialization
   2.5. Multi-GPU parallel K estimation
   2.6. Block-bootstrap uncertainty quantification
   2.7. Implementation (PyTorch, PLINK BED input, output format)
3. Results
   3.1. Accuracy validation (simulation: SIM_K3, SIM_K5_ADMIX)
   3.2. Runtime comparison (all 4 baselines, multiple datasets)
   3.3. Strong-scaling analysis (fastmixture 1-64 threads vs gpuADMIX 1 GPU)
   3.4. Admixture bar plots on 1kGP (K=5, all methods)
   3.5. Parallel K-sweep efficiency
   3.6. Bootstrap UQ example
4. Discussion (contributions, limitations, future work: WGS scale, biobank)
5. Software availability

## Key Claims (locked after G2)

CLAIM 1: "gpuADMIX achieves equivalent accuracy to ADMIXTURE and fastmixture, 
  measured by RMSE and JSD vs. simulated ground truth Q"
EVIDENCE TYPE: simulation experiment with msprime ground truth

CLAIM 2: "gpuADMIX is significantly faster than fastmixture (CPU-64-threads) 
  on multi-GPU hardware"
EVIDENCE TYPE: wall-clock comparison on same machine; reported speedup factor

CLAIM 3: "GPU advantage does not saturate with more CPU threads"
EVIDENCE TYPE: strong-scaling experiment (fastmixture 1/8/16/32/64 threads vs 1 GPU)

CLAIM 4: "Parallel K-sweep reduces total K-selection time by ~K-fold"
EVIDENCE TYPE: parallel vs sequential K timing

CLAIM 5: "Nesterov momentum accelerates convergence vs plain EM on GPU"
EVIDENCE TYPE: ablation: iterations to convergence, wall time
