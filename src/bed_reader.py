"""
Fast PLINK BED file reader.
Returns genotype matrix G as (N_individuals × M_snps) float32 tensor on specified device.
Missing genotypes (PLINK code 1) are imputed with the SNP mean (2*p_hat).
"""
import numpy as np
import torch


def read_bed_numpy(prefix: str) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Read PLINK BED file into numpy int8 array.
    Returns:
        G: (N, M) int8 array, -1 = missing
        sample_ids: list of N sample IDs
        snp_ids: list of M SNP IDs
    """
    fam_path = prefix + '.fam'
    bim_path = prefix + '.bim'
    bed_path = prefix + '.bed'

    with open(fam_path) as f:
        lines = f.readlines()
    N = len(lines)
    sample_ids = [l.split()[1] for l in lines]

    with open(bim_path) as f:
        lines = f.readlines()
    M = len(lines)
    snp_ids = [l.split()[1] for l in lines]

    bytes_per_snp = (N + 3) // 4

    with open(bed_path, 'rb') as f:
        magic = f.read(3)
        if list(magic) != [0x6c, 0x1b, 0x01]:
            raise ValueError("Invalid BED file or not SNP-major mode")
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    raw = raw.reshape(M, bytes_per_snp)

    # Vectorized 2-bit unpacking:
    # bits [1:0], [3:2], [5:4], [7:6] → 4 samples per byte
    # PLINK encoding: 00→0(AA), 01→missing, 10→1(Aa), 11→2(aa)
    # PLINK encoding (A1=minor allele convention, matching ADMIXTURE/fastmixture):
    # 00 → 2 (homozygous A1),  01 → missing,  10 → 1 (hetero),  11 → 0 (homozygous A2)
    lookup = np.array([2, -1, 1, 0], dtype=np.int8)

    cols = [lookup[(raw >> shift) & 0b11] for shift in (0, 2, 4, 6)]
    # cols[i]: (M, bytes_per_snp) → sample indices i, i+4, i+8, ...
    # interleave: samples 0,4,8,... | 1,5,9,... | 2,6,10,... | 3,7,11,...
    G_snp_major = np.empty((M, N), dtype=np.int8)
    for offset, col in enumerate(cols):
        indices = np.arange(offset, N, 4)
        valid = indices < N
        G_snp_major[:, indices[valid]] = col[:, :valid.sum()]

    return G_snp_major.T, sample_ids, snp_ids  # (N, M)


def read_bed_tensor(prefix: str, device: str = 'cpu',
                    impute: bool = True) -> tuple[torch.Tensor, list[str], list[str]]:
    """
    Read PLINK BED into float32 torch tensor with optional mean imputation.
    Returns G: (N, M) float32 tensor.
    """
    G_np, sample_ids, snp_ids = read_bed_numpy(prefix)
    G = torch.from_numpy(G_np.astype(np.float32))  # (N, M)

    if impute:
        missing = (G == -1)
        if missing.any():
            # Compute per-SNP mean from non-missing values
            G_masked = G.clone()
            G_masked[missing] = 0.0
            counts = (~missing).float().sum(0)  # (M,)
            means = G_masked.sum(0) / counts.clamp(min=1)  # (M,)
            G[missing] = means.unsqueeze(0).expand_as(G)[missing]

    return G.to(device), sample_ids, snp_ids
