# Ablation Plan — gpuADMIX
# Phase 3, M-Step 4

## Core Ablations

| Component | Ablation | Expected Effect | Why Important |
|---|---|---|---|
| Nesterov momentum | Remove → plain EM | +20-50% more iterations to converge | Validates momentum contribution |
| SVD initialization | Remove → random init | +50% iterations or instability | Validates init contribution |
| Mini-batch (B=32) | B=1 (full-batch) | Slower per iteration; may need fewer iterations | Validates batching strategy |
| GPU | Same code on CPU (torch, no CUDA) | ~10-50x slower | Validates GPU contribution directly |
| Multi-GPU K-parallel | Sequential K on single GPU | K× slower K-sweep | Validates multi-GPU K utility |

## Additional Experiments

| Experiment | What It Shows |
|---|---|
| B sensitivity: B={1, 8, 16, 32, 64} | Optimal batch size for GPU utilization |
| K sensitivity: K={2,3,5,7,10} on 1kGP | Behavior at different K; parallel-K speedup curve |
| N scaling: N={1K, 3K, 10K} | GPU advantage grows with N |
| Multi-GPU scaling: 1 GPU vs 2 GPU vs 4 GPU | Parallel K efficiency |
| Convergence trace: LL vs iteration | GPU vs fastmixture convergence speed comparison |
| Bootstrap UQ: 100 replicates on 1kGP | Q uncertainty visualized on stacked bar plot |
