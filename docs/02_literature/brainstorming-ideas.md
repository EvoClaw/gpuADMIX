# 5 Candidate Research Ideas
Generated: 2026-02-25 (Step 5c)
Source: Synthesis of gap analysis + 12 deep thinking insights

=== IDEA 1: GPU-Accelerated Mini-Batch EM for Model-Based Admixture ===
Core question: Can the model-based EM algorithm for ancestry estimation (as used by ADMIXTURE and fastmixture) be efficiently implemented on multi-GPU hardware to achieve 100-500x speedup over ADMIXTURE while maintaining identical accuracy?
Novelty source: Insight 1 (contradiction: GPU is not inherently incompatible with accuracy) + Insight 11 (trend extrapolation) + Insight 7 (fastmixture limitation-to-opportunity)
Why it matters: Would be the first GPU-accelerated model-based admixture tool with ADMIXTURE-level accuracy. Enables accurate ancestry estimation for biobank-scale datasets (100K-1M samples) that are currently inaccessible to accurate methods.
Key algorithmic contributions:
  1. Reformulate ADMIXTURE/fastmixture EM updates as GPU GEMM operations
  2. Efficient 2-bit genotype encoding on GPU VRAM (PLINK BED format unpacking in CUDA)
  3. Multi-GPU data-parallel EM (split variant dimension across GPUs)
  4. GPU-native QN/Nesterov acceleration adapted for async batch execution
Feasibility: HIGH - algorithms are well-understood; GPU GEMM is standard; user has multi-GPU server
Risk: 
  - Memory bottleneck: N*M float32 matrix may exceed GPU VRAM for large datasets
  - Mitigation: chunked processing, same as fastmixture's randomized SVD chunk approach
  - May need careful numerical precision management (float16 vs float32)
Competition: No known competing work. fastmixture is CPU-only; Neural ADMIXTURE is GPU but broken.
Estimated scope: Journal paper (Bioinformatics or Genome Research)

=== IDEA 2: Natural Gradient Variational Inference for Admixture on GPU ===
Core question: Can natural gradient stochastic variational inference (adapted from Online LDA) applied to the admixture model achieve better accuracy than fastSTRUCTURE while matching or exceeding fastmixture speed on GPU?
Novelty source: Insight 5 (cross-domain transfer from LDA/NLP) + Insight 3 (challenging EM-as-only-option assumption)
Why it matters: Natural gradients exploit the Riemannian geometry of the simplex and Fisher information metric, potentially converging in fewer iterations. Could yield a theoretically principled fast AND accurate method.
Key algorithmic contributions:
  1. Derive natural gradient update for the admixture model's Q and P parameters
  2. Mini-batch natural gradient with GPU acceleration
  3. Amortized inference: learn a shared encoder Q(genotype) -> admixture proportions that CORRECTLY optimizes the binomial log-likelihood (fixing Neural ADMIXTURE's bug)
Feasibility: MEDIUM - requires deriving new updates and validating; more complex than Idea 1
Risk:
  - Natural gradient computation requires Fisher information matrix inverse, which may be expensive
  - Accuracy advantage over simple EM+QN is not guaranteed
Competition: No one has tried natural gradient admixture; fastSTRUCTURE uses variational Bayes but not natural gradients
Estimated scope: Journal paper (PLOS Genetics or Bioinformatics)

=== IDEA 3: GPU-Admixture with Parallel K Selection and Uncertainty Quantification ===
Core question: Can a single multi-GPU run simultaneously estimate admixture proportions for K=2..20 AND provide convergence uncertainty, replacing the current need for 20 separate sequential runs?
Novelty source: Insight 4 (challenging sequential K assumption) + Idea 1 base (GPU EM) + Gap 4 (K selection)
Why it matters: Current workflow requires N_K separate ADMIXTURE runs for cross-validation. With GPU, could parallelize across K values in a single job. Additionally, running multiple restarts per K (currently infeasible for large datasets) would enable proper uncertainty quantification on Q.
Key algorithmic contributions:
  1. Multi-stream GPU execution: different K values as separate CUDA streams on different GPU devices
  2. Shared genotype loading: genotype matrix loaded once to GPU, shared across K computations
  3. Cross-validation log-likelihood computation fully on GPU
  4. Bootstrap uncertainty estimation for admixture proportions
Feasibility: HIGH (builds on Idea 1; multi-stream GPU execution is standard)
Risk:
  - GPU memory may limit simultaneous K values
  - Mitigation: process K values sequentially but cache genotype on GPU
Competition: Neural ADMIXTURE multi-head (but inaccurate); no accurate multi-K GPU method
Estimated scope: Journal paper (Bioinformatics; could be combined with Idea 1)

=== IDEA 4: GPU-Admixture with Whole-Genome Sequencing Data (100M+ SNPs) ===
Core question: Can GPU-accelerated model-based admixture handle whole-genome sequencing (WGS) data with 10-100M SNPs per sample, a scale completely inaccessible to current accurate methods?
Novelty source: Insight 12 (trend extrapolation: biobank sizes growing) + Insight 7 + Gap 5
Why it matters: WGS data is standard for UK Biobank Phase 2, gnomAD, All of Us. Current accurate methods (fastmixture) need 74 min for 3K samples x 7M SNPs. For 50K samples x 10M SNPs: ~weeks. GPU could reduce to hours.
Key algorithmic contributions:
  1. Out-of-core GPU processing: stream genotype data from disk to GPU in chunks
  2. Two-level mini-batch: outer loop over SNP chunks (streamed from disk), inner loop within GPU VRAM
  3. Convergence-guaranteed streaming EM with theoretical analysis
  4. Demonstration on WGS data (1kGP sequence data or simulated WGS)
Feasibility: MEDIUM-HIGH - streaming EM is well-studied; disk I/O may be bottleneck
Risk:
  - I/O bound: SSD throughput may limit speed for very large WGS datasets
  - Convergence guarantees for streaming EM are more complex to prove
Competition: SCOPE handles large datasets but inaccurate; fastmixture future work mentions this
Estimated scope: Journal paper (Nature Methods or Bioinformatics; higher impact)

=== IDEA 5: Supervised/Semi-supervised GPU Admixture with Reference Panel ===
Core question: Can GPU-accelerated admixture leverage a reference panel (known population allele frequencies) to achieve faster convergence AND better accuracy, while scaling to modern biobanks?
Novelty source: Insight 6 (cross-domain: LDA has supervised variants) + SCOPE's supervised feature (but SCOPE is inaccurate)
Why it matters: In practice, reference panels (1kGP, gnomAD, HGDP) are always available. Using fixed P (allele frequencies) from reference panels reduces the problem to estimating Q only, which is much faster. Currently only SCOPE does this but with ALS (inaccurate). A GPU model-based supervised admixture would be both fast and accurate.
Key algorithmic contributions:
  1. Fix P from reference panel; only optimize Q (N*K parameters vs N*K + M*K)
  2. Reduces EM to M-step for Q only: massive simplification and speedup
  3. Multi-GPU parallel Q estimation: each GPU handles subset of individuals
  4. Semi-supervised extension: iteratively refine P from data + prior reference
Feasibility: HIGH - fixing P reduces complexity dramatically; Q update is embarrassingly parallel across individuals
Risk:
  - Limited to scenarios where reference panel matches population
  - Full unsupervised (no reference) is still needed
Competition: SCOPE supervised (but inaccurate); no accurate GPU supervised method
Estimated scope: Journal paper (Bioinformatics); could be companion paper to Idea 1
