#!/usr/bin/env python3
"""
clumpak.py — CLUMPAK-lite: within-K and across-K label alignment for gpuADMIX.

Usage examples
--------------
# Align K=2..10, seeds 42 1 2 3 7, then draw structure plot
python clumpak.py \\
    --result-dir results/gpuadmix \\
    --bname 1kGP_200k_ldpruned \\
    --K 2 3 4 5 6 7 \\
    --seeds 42 1 2 3 7 \\
    --out-dir results/aligned \\
    --sample-info sample.info \\
    --plot results/structure_plot.pdf

# Quick: single-K alignment only (no --K list defaults to all found)
python clumpak.py --result-dir results/gpuadmix --bname 1kGP_200k_ldpruned \\
    --plot results/structure_k5.png --K 5 --seeds 42 1 2
"""
import argparse, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.clumpak_lite   import run_clumpak_lite
from src.structure_plot import plot_structure_panel


def load_sample_info(path: str) -> dict:
    """
    Load sample → population mapping.
    Expected format: two-column TSV/CSV: sample_id  population
    (matches PLINK .fam column 2 → column 2 or a separate info file)
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_fam(bfile_prefix: str) -> list:
    """Load sample IDs from .fam file (column 2 = individual ID)."""
    fam_path = bfile_prefix + '.fam'
    if not os.path.exists(fam_path):
        return []
    with open(fam_path) as f:
        return [line.split()[1] for line in f if line.strip()]


def auto_detect_params(result_dir: str, bname: str) -> tuple:
    """Auto-detect available K values and seeds from file names."""
    pattern = os.path.join(result_dir, f"{bname}.K*.s*.Q")
    files   = glob.glob(pattern)
    K_set, s_set = set(), set()
    for f in files:
        base = os.path.basename(f)
        try:
            parts = base.replace(bname + '.', '').split('.')
            for p in parts:
                if p.startswith('K'):
                    K_set.add(int(p[1:]))
                elif p.startswith('s'):
                    s_set.add(int(p[1:]))
        except ValueError:
            pass
    return sorted(K_set), sorted(s_set)


def main():
    p = argparse.ArgumentParser(
        description='CLUMPAK-lite: align gpuADMIX Q matrices and draw structure plot.')

    p.add_argument('--result-dir', default='results/gpuadmix',
                   help='Directory containing .K*.s*.Q files')
    p.add_argument('--bname', required=True,
                   help='Dataset basename (e.g. 1kGP_200k_ldpruned)')
    p.add_argument('--K', type=int, nargs='+',
                   help='K values to include (default: auto-detect all found)')
    p.add_argument('--seeds', type=int, nargs='+',
                   help='Seeds to include (default: auto-detect)')
    p.add_argument('--out-dir', default='results/aligned',
                   help='Output directory for aligned Q files')
    p.add_argument('--bfile',
                   help='PLINK bfile prefix (to read sample IDs from .fam)')
    p.add_argument('--sample-info', default='samples.info',
                   help='TSV file: sample_id  population (default: samples.info)')
    p.add_argument('--plot',
                   help='Output path for structure plot (.pdf/.png/.svg)')
    p.add_argument('--no-P', action='store_true',
                   help='Use Q for across-K alignment instead of P matrices')
    p.add_argument('--K-min-plot', type=int, default=None,
                   help='Minimum K to include in plot')
    p.add_argument('--K-max-plot', type=int, default=None,
                   help='Maximum K to include in plot')
    args = p.parse_args()

    # Auto-detect K and seeds if not specified
    K_list_found, seeds_found = auto_detect_params(args.result_dir, args.bname)
    K_list = args.K if args.K else K_list_found
    seeds  = args.seeds if args.seeds else seeds_found

    if not K_list:
        print(f"ERROR: no Q files found in {args.result_dir} for basename '{args.bname}'")
        sys.exit(1)

    print(f"Detected K={K_list}, seeds={seeds}")

    # Run CLUMPAK-lite pipeline
    Q_struct = run_clumpak_lite(
        result_dir   = args.result_dir,
        bname        = args.bname,
        K_list       = K_list,
        seeds        = seeds,
        out_dir      = args.out_dir,
        use_P_for_across_k = not args.no_P,
    )

    if not Q_struct:
        print("No aligned Q matrices produced. Check file paths.")
        sys.exit(1)

    # Structure plot
    if args.plot:
        # Load sample info for population labels
        pop_labels = None
        sample_ids = None

        bfile = args.bfile or args.bname
        fam_ids = load_fam(bfile)

        if args.sample_info and os.path.exists(args.sample_info):
            info = load_sample_info(args.sample_info)
            if fam_ids:
                pop_labels = np.array([info.get(sid, 'UNK') for sid in fam_ids])
                sample_ids = np.array(fam_ids)
                print(f"Loaded {len(info)} population labels from {args.sample_info}")
            else:
                print(f"WARNING: --bfile not provided, cannot map sample IDs to populations")
        elif fam_ids:
            sample_ids = np.array(fam_ids)

        # Filter K range for plot
        plot_K = {K: Q for K, Q in Q_struct.items()}
        if args.K_min_plot:
            plot_K = {K: Q for K, Q in plot_K.items() if K >= args.K_min_plot}
        if args.K_max_plot:
            plot_K = {K: Q for K, Q in plot_K.items() if K <= args.K_max_plot}

        N = next(iter(Q_struct.values())).shape[0]
        suptitle = f"{args.bname}  (N={N}, K={min(plot_K)}..{max(plot_K)})"

        plot_structure_panel(
            Q_by_k     = plot_K,
            pop_labels = pop_labels,
            sample_ids = sample_ids,
            out_path   = args.plot,
            suptitle   = suptitle,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
