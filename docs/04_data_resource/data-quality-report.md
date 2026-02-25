# Data Quality Report — gpuADMIX
Generated: 2026-02-25

## Dataset: 1kGP_200k_ldpruned

### File Validation
| File | Status | Details |
|------|--------|---------|
| `1kGP_200k_ldpruned.bed` | ✅ Valid | Magic bytes [0x6c, 0x1b, 0x01] confirmed (SNP-major mode) |
| `1kGP_200k_ldpruned.bim` | ✅ Valid | 200,000 SNPs |
| `1kGP_200k_ldpruned.fam` | ✅ Valid | 3,202 individuals |
| `samples.info` | ✅ Valid | 26 super-populations / populations |

### Data Dimensions
- **Individuals (N)**: 3,202
- **SNPs (M)**: 200,000
- **Populations**: 26 populations (5 super-populations: AFR, AMR, EAS, EUR, SAS)
- **Source**: 1000 Genomes Project (phase 3), LD-pruned to 200K SNPs

### Memory Requirements
| Format | Size | Fits in L20 VRAM (48GB)? |
|--------|------|--------------------------|
| int8 (storage) | 0.64 GB | ✅ Yes |
| float32 (computation) | 2.56 GB | ✅ Yes (trivially) |
| float32 × 8 GPUs | 20.5 GB | ✅ Yes |

### Hardware Confirmed
| Resource | Spec |
|----------|------|
| GPUs | 8× NVIDIA L20 (48 GB VRAM each) |
| CPU | Intel Xeon Platinum 8358P @ 2.60GHz, 128 cores |
| RAM | 1 TB |
| CUDA | 12.5 (driver), 12.1 (PyTorch build) |

### BIM Format Check
- Chromosome: 1-22 (autosomal)
- Positions: physical bp positions present
- Alleles: biallelic SNPs confirmed from LD pruning step
- SNP IDs: `CHR:POS:REF:ALT` format

### FAM Format Check
- Sample IDs present (individual IDs in column 1 and 2)
- No sex coding (-9 = unknown for all)
- No phenotype (-9 for all, unsupervised analysis)

### Leakage Audit
- **N/A**: This is an unsupervised learning task (no train/test split for inference)
- Admixture proportions are estimated jointly across all individuals; no label leakage possible

## G3 Gate Status: ✅ PASSED

All items confirmed:
- [x] Data arrived and validated
- [x] Leakage audit: N/A (unsupervised)
- [x] Resources confirmed (8× L20 GPUs exceed requirements)
- [x] Tool chain ready (PyTorch 2.5.1+cu121, fastmixture 1.3.0, ADMIXTURE 1.3.0)
- [x] Baseline implementation plan confirmed in evaluation-protocol.yaml
