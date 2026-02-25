# gpuADMIX

**GPU-accelerated ancestry estimation with Nesterov-augmented mini-batch EM**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-green.svg)](paper/main.pdf)

> 📄 **[Read the Paper (PDF)](https://github.com/EvoClaw/gpuADMIX/blob/main/paper/main.pdf)**
> — *gpuADMIX: GPU-accelerated ancestry estimation with Nesterov-augmented mini-batch EM*

---

## Overview

gpuADMIX is a GPU-accelerated tool for model-based ancestry estimation from
genome-wide genotype data. It reformulates the ADMIXTURE expectation-maximisation
(EM) algorithm as GPU-native dense matrix multiplications (DGEMM), achieving
**41× and 213× speedups** over fastmixture and ADMIXTURE respectively on the
1000 Genomes Phase 3 dataset — while **matching or exceeding** their accuracy.

Key algorithmic innovations:

| Component | Benefit |
|---|---|
| GPU-native EM (DGEMM) | Full ADMIXTURE likelihood, massively parallel |
| FISTA-style Nesterov momentum | 2.3× fewer iterations, +7,865 LL over plain EM |
| Stochastic mini-batch EM | Better solution quality, lower peak GPU memory |
| Streaming randomised SVD init | Efficient spectral initialisation for large N |
| CLUMPAK-lite post-processor | Label-consistent ancestry bar plots across all K |
| Multi-GPU K dispatcher | K=2–10 scan in ~130 s on 8 GPUs |

### Performance (K=5, 1kGP Phase 3, N=3,202, M=200K SNPs)

| Method | Wall time (s) | Speedup | Log-likelihood | Q r² vs ADMIXTURE |
|---|---|---|---|---|
| ADMIXTURE | 3,583 | 1× | −241,227,839 | 1.000000 |
| fastmixture | 694 ± 40 | 5× | −241,227,643 ± 0.3 | 0.999984 |
| **gpuADMIX** | **16.8 ± 3.4** | **213×** | **−241,224,751 ± 98** | **0.999987** |

Hardware: NVIDIA L20 GPU (gpuADMIX) vs Intel Xeon Platinum 8375C 32-thread CPU (baselines).

---

## Research Assistance

This project was conducted with the assistance of
**[Amplify](https://evoclaw.github.io/amplify/)** — an open-source agentic
research automation framework developed at HKUST(GZ) that turns an AI coding
assistant into an autonomous co-scientist.

Amplify enforced the full 7-phase research workflow for this project:

| Phase | Activity |
|---|---|
| 0 — Domain Anchoring | Field identification, expert persona, resource assessment |
| 1 — Direction Exploration | Literature review of 20+ papers, 6 deep-thinking strategies, multi-agent brainstorming |
| 2 — Problem Validation | Adversarial 3-agent scrutiny of novelty, feasibility, and question specificity |
| 3 — Method Design | GPU-native EM design, evaluation protocol locked (metrics frozen before experiments) |
| 4 — Experiment Execution | Baseline-first, 5 seeds per K, full K=2–10 scan, ablations, cross-validation |
| 5 — Results Integration | 3-agent story deliberation, claim-evidence alignment, publishability check |
| 6 — Paper Writing | Modular LaTeX, per-section 3-agent polishing, reference verification |

Key scientific rigor properties enforced by Amplify:
- **Anti-cherry-picking**: all 5 seeds reported, including failures and variance
- **Metric lock**: evaluation metrics fixed before any experiment ran
- **Claim-evidence alignment**: every number in the paper mapped to a specific experiment log
- **Reference verification**: all citations checked; no fabricated references

**All experimental results are independently reproducible** using the scripts
and instructions in this repository. Amplify orchestrated the research process;
every number in the paper corresponds to actual code execution on real hardware.
See the [Reproducibility](#reproducibility) section below.

---

## Installation

```bash
git clone https://github.com/EvoClaw/gpuADMIX.git
cd gpuADMIX
pip install torch numpy scipy matplotlib pandas scikit-learn
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0 with CUDA support
- PLINK 2.0 (for data preprocessing)
- A CUDA-capable GPU (tested on NVIDIA L20 48 GB; any GPU with ≥ 8 GB VRAM
  works for the 1kGP benchmark at K ≤ 10)

---

## Quick Start

```bash
# Run ancestry estimation (K=5, single GPU)
python gpuadmix.py \
    --bfile data/1kGP_200k_ldpruned \
    --K 5 \
    --n_seeds 5 \
    --out results/my_run

# Scan K=2..10 in parallel across multiple GPUs
python scripts/parallel_k_selection.py \
    --bfile data/1kGP_200k_ldpruned \
    --k_min 2 --k_max 10 \
    --n_seeds 5 \
    --out results/k_scan

# Generate STRUCTURE-style bar plot
python clumpak.py \
    --results_dir results/k_scan \
    --k_min 2 --k_max 7 \
    --fam data/1kGP_200k_ldpruned.fam \
    --out results/structure_plot
```

---

## Data

The benchmark uses the **1000 Genomes Project Phase 3** dataset, LD-pruned to
200,000 autosomal SNPs. The genotype data is not included in this repository
(redistributable from the [1kGP FTP](https://ftp.1000genomes.ebi.ac.uk/)):

```bash
# Download and preprocess (requires PLINK 2.0)
bash ld_prune_1kGP.sh
```

The preprocessing script downloads the 1kGP VCF files, converts to PLINK BED
format, and applies LD pruning (`--indep-pairwise 50 10 0.1`).

---

## Repository Structure

```
gpuADMIX/
├── gpuadmix.py              # Main CLI entry point
├── clumpak.py               # CLUMPAK-lite CLI (label alignment + plotting)
├── ld_prune_1kGP.sh         # Data preprocessing pipeline
│
├── src/
│   ├── gpuadmix_core.py     # Core GPU EM engine (PyTorch)
│   ├── clumpak_lite.py      # Within-K and across-K alignment algorithms
│   ├── structure_plot.py    # Ancestry bar plot generation
│   ├── bed_reader.py        # PLINK BED format reader
│   └── plot_style.py        # Publication-quality plot style
│
├── scripts/
│   ├── parallel_k_selection.py   # Multi-GPU K scan dispatcher
│   ├── cv_for_k.py               # 5-fold SNP hold-out cross-validation
│   ├── run_gpuadmix_full.py      # Full benchmark runner
│   ├── run_fastmixture_full.sh   # fastmixture benchmark runner
│   ├── simulate_data.py          # Simulated data generator
│   ├── make_figures.py           # Publication figure generation
│   └── collect_results.py        # Results aggregation
│
├── results/
│   ├── figures/                  # Publication-quality PDF figures
│   ├── structure_plot_K2-7.pdf   # Main structure plot
│   ├── phase4b_summary.csv       # Benchmark summary table
│   ├── cv_results.json           # Cross-validation results
│   └── gpuadmix_k8_10_fresh.json # K=8–10 benchmark results
│
├── paper/
│   ├── main.tex                  # Main LaTeX file
│   ├── main.pdf                  # Compiled manuscript
│   ├── preamble.tex
│   ├── references.bib
│   ├── sections/                 # Abstract, Intro, Methods, Results, Discussion
│   ├── tables/                   # LaTeX table files
│   └── figures/                  # PDF figures for paper
│
└── docs/
    ├── 03_plan/                  # Method design and evaluation protocol
    ├── 05_execution/             # Experiment logs and results compilation
    └── 06_integration/           # Argument blueprint and paper plan
```

---

## Reproducibility

All results in the paper are reproducible. The following steps reproduce the
main benchmark (Table 1):

```bash
# 1. Preprocess data
bash ld_prune_1kGP.sh

# 2. Run gpuADMIX (5 seeds, K=5)
python scripts/run_gpuadmix_full.py \
    --bfile data/1kGP_200k_ldpruned --K 5 --n_seeds 5

# 3. Run fastmixture baseline (requires fastmixture installed)
bash scripts/run_fastmixture_full.sh

# 4. Run cross-validation (K=2..10)
python scripts/cv_for_k.py \
    --bfile data/1kGP_200k_ldpruned --k_max 10

# 5. Generate figures
python scripts/make_figures.py --results_dir results/ --out results/figures/
```

**Seeds used in paper:** 0, 1, 2, 3, 4 (passed via `--seeds 0 1 2 3 4`).

Pre-computed results (JSON/CSV) are included in `results/` for reference.

---

## Compiling the Paper

The paper uses the Oxford University Press LaTeX template.

```bash
cd paper/

# Download OUP template (one-time setup)
wget https://mirrors.ctan.org/macros/latex/contrib/oup-authoring-template.zip
unzip oup-authoring-template.zip "*.cls" "*.bst" -d .
# Also download algorithmicx (required by OUP class)
wget https://mirrors.ctan.org/macros/latex/contrib/algorithmicx.zip
unzip algorithmicx.zip "algorithmicx/*.sty" -d .
cp algorithmicx/*.sty .

# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Citation

If you use gpuADMIX in your research, please cite:

```bibtex
@article{gpuadmix2026,
  title  = {{gpuADMIX}: {GPU}-accelerated ancestry estimation with
             {N}esterov-augmented mini-batch {EM}},
  author = {Amplify},
  journal = {Bioinformatics},
  year   = {2026},
  note   = {Preprint}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [ADMIXTURE](https://dalexander.github.io/admixture/) (Alexander et al., 2009)
  for the foundational model and benchmark baseline.
- [fastmixture](https://github.com/Rosemeis/fastmixture) (Meisner et al., 2024)
  for the CPU state-of-the-art baseline.
- The 1000 Genomes Project Consortium for the benchmark dataset.
- Research workflow designed and executed with
  **[Amplify](https://evoclaw.github.io/amplify/)** agentic research automation.
