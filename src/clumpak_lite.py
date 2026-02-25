"""
clumpak_lite.py — CLUMPAK-like label alignment for gpuADMIX output.

Solves the label-switching problem:
  1. Within-K alignment:  multiple seeds at the same K → find "major mode",
                          align all runs to it via Hungarian algorithm.
  2. Across-K alignment:  K=2..Kmax for structure plot → ensure that the
                          "same" ancestral component keeps the same colour.

Algorithm:
  - Within-K: pairwise Pearson r between Q columns; solve assignment problem
    with scipy.optimize.linear_sum_assignment.
  - Across-K: greedy bottom-up. Start from K=K_min. For each step K→K+1,
    find the permutation of K+1 columns that maximises total correlation with
    the K reference. The one extra component in K+1 is assigned to the last
    position (gets a new colour in the structure plot).
  - Major mode: hierarchical clustering of within-K runs; take the largest
    cluster's centroid as the representative Q.

Reference:
  Kopelman et al. (2015) CLUMPAK: a program for identifying clustering modes
  and packaging population structure inferences. Mol. Ecol. Resour. 15:1179.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Low-level: find best column permutation between two Q matrices
# ─────────────────────────────────────────────────────────────────────────────

def _corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pearson r between each pair of columns (A[:, i], B[:, j]).
    Returns (K_A, K_B) matrix — vectorised via normalised column dot products.
    """
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    nA = np.linalg.norm(A_c, axis=0, keepdims=True)   # (1, K_A)
    nB = np.linalg.norm(B_c, axis=0, keepdims=True)   # (1, K_B)
    A_n = A_c / (nA + 1e-12)   # (N, K_A)
    B_n = B_c / (nB + 1e-12)   # (N, K_B)
    return A_n.T @ B_n          # (K_A, K_B)


def align_to_reference(Q_ref: np.ndarray, Q_test: np.ndarray,
                        allow_partial: bool = False
                        ) -> Tuple[np.ndarray, List[int]]:
    """
    Find the column permutation of Q_test that best matches Q_ref.

    If Q_test has more columns than Q_ref (across-K case, allow_partial=True),
    only the first K_ref columns of the permuted Q_test are matched; the extra
    columns are appended at the end in their original relative order.

    Returns
    -------
    Q_aligned : ndarray  (N, K_test) — Q_test with columns reordered
    perm      : list[int] — permutation applied (Q_aligned = Q_test[:, perm])
    """
    K_ref  = Q_ref.shape[1]
    K_test = Q_test.shape[1]

    C = _corr_matrix(Q_test, Q_ref)   # (K_test, K_ref)

    if K_test == K_ref:
        # Square: standard Hungarian
        row_ind, col_ind = linear_sum_assignment(-C)
        # col_ind[i] = which ref column matches test row i
        # We want perm[j] = which test column goes to position j
        perm = [0] * K_test
        for r, c in zip(row_ind, col_ind):
            perm[c] = r
    elif K_test > K_ref and allow_partial:
        # Rectangular: match first K_ref columns, extras at end
        # Use Hungarian on top K_ref×K_ref sub-problem
        C_sub = C[:, :]   # (K_test, K_ref)
        row_ind, col_ind = linear_sum_assignment(-C_sub)
        # row_ind: which K_test rows were selected (all K_ref of them)
        # col_ind: which K_ref columns they map to (0..K_ref-1)

        matched_test_cols = list(row_ind)   # K_ref test cols matched to K_ref ref cols
        unmatched         = [c for c in range(K_test) if c not in matched_test_cols]

        # Build perm of length K_test
        # Position j (0..K_ref-1) ← the test column matched to ref col j
        perm = [None] * K_test
        for r, c in zip(row_ind, col_ind):
            perm[c] = r   # ref position c ← test column r
        # Fill in unmatched columns at positions K_ref..K_test-1
        tail_pos = K_ref
        for u in unmatched:
            perm[tail_pos] = u
            tail_pos += 1
    else:
        # K_test < K_ref: unusual, just do best-effort square match on K_test cols
        C_sub = C[:, :K_test]
        row_ind, col_ind = linear_sum_assignment(-C_sub)
        perm = list(row_ind)

    Q_aligned = Q_test[:, perm]
    return Q_aligned, perm


# ─────────────────────────────────────────────────────────────────────────────
# Within-K: align multiple runs, find major mode
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_rmse(Qs: List[np.ndarray]) -> np.ndarray:
    """RMSE matrix between runs after optimal alignment."""
    n = len(Qs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Q_j_aligned, _ = align_to_reference(Qs[i], Qs[j])
            rmse = np.sqrt(np.mean((Qs[i] - Q_j_aligned) ** 2))
            D[i, j] = D[j, i] = rmse
    return D


def align_within_k(Q_list: List[np.ndarray],
                   P_list: Optional[List[np.ndarray]] = None,
                   mode: str = 'centroid'
                   ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray,
                               Optional[np.ndarray]]:
    """
    Align multiple Q (and optionally P) matrices from runs with the same K.

    Q and P receive the SAME per-run column permutation so that
    Q_rep[:,k] and P_rep[:,k] describe the same ancestral component.

    Parameters
    ----------
    Q_list : list of (N, K) arrays
    P_list : optional list of (M, K) arrays — aligned with the same perm as Q
    mode   : 'centroid' (default)

    Returns
    -------
    Q_rep    : (N, K)        — centroid Q (component ordering fixed)
    Q_aligned: list (N, K)   — all Q runs aligned to Q_rep
    D        : (n_runs, n_runs) — pairwise RMSE matrix
    P_rep    : (M, K) or None — centroid P (same ordering as Q_rep)
    """
    if len(Q_list) == 1:
        P_rep = P_list[0].copy() if P_list else None
        return Q_list[0].copy(), [Q_list[0].copy()], np.array([[0.0]]), P_rep

    n = len(Q_list)
    D = _pairwise_rmse(Q_list)

    ref_idx = int(np.argmin(D.sum(axis=1)))
    Q_ref   = Q_list[ref_idx].copy()

    # Iterative centroid refinement; track permutations to align P consistently
    aligned_Q = [Q_ref.copy()]
    perms     = [list(range(Q_ref.shape[1]))]   # identity perm for ref

    for i in range(n):
        if i == ref_idx:
            continue
        Q_al, perm = align_to_reference(Q_ref, Q_list[i])
        aligned_Q.append(Q_al)
        perms.append(perm)

    for _ in range(3):
        Q_ref = np.mean(aligned_Q, axis=0)
        Q_ref /= Q_ref.sum(axis=1, keepdims=True)
        aligned_Q, perms = [], []
        for Q in Q_list:
            Q_al, perm = align_to_reference(Q_ref, Q)
            aligned_Q.append(Q_al)
            perms.append(perm)

    Q_rep = np.mean(aligned_Q, axis=0)
    Q_rep /= Q_rep.sum(axis=1, keepdims=True)

    # Align P using the SAME permutations derived from Q alignment
    P_rep = None
    if P_list is not None and len(P_list) == n:
        aligned_P = [P_list[i][:, perms[i]] for i in range(n)]
        P_rep = np.mean(aligned_P, axis=0)

    return Q_rep, aligned_Q, D, P_rep


# ─────────────────────────────────────────────────────────────────────────────
# Across-K: bottom-up greedy alignment for structure plot
# ─────────────────────────────────────────────────────────────────────────────

def align_across_k(Q_by_k: Dict[int, np.ndarray],
                   P_by_k: Optional[Dict[int, np.ndarray]] = None,
                   anchor_k: Optional[int] = None
                   ) -> Dict[int, np.ndarray]:
    """
    Align Q matrices across K values so that 'the same' component keeps the
    same column index (and therefore colour in the structure plot).

    Strategy
    --------
    1. Sort K values.
    2. Use the smallest K as anchor (or user-supplied anchor_k).
    3. For each step K_prev → K_curr:
         Find permutation of K_curr columns that maximises correlation with
         K_prev aligned columns.  The extra column in K_curr gets index K_curr-1
         (rightmost = new colour).

    If P_by_k is supplied, matching uses P matrices (M×K) instead of Q (N×K);
    P-based matching is often more reliable because M >> N, but Q-based is the
    default when P is unavailable.

    Returns
    -------
    Q_aligned_by_k : dict {K: aligned (N, K) Q matrix}
    """
    K_list = sorted(Q_by_k.keys())
    if not K_list:
        return {}

    if anchor_k is None:
        anchor_k = K_list[0]

    # Use P or Q for matching signal
    ref_by_k = P_by_k if P_by_k is not None else Q_by_k

    result = {}
    # Anchor stays unchanged
    result[anchor_k]      = Q_by_k[anchor_k].copy()
    result_ref            = {anchor_k: ref_by_k[anchor_k].copy()}

    # Propagate upward (increasing K)
    for i, K in enumerate(K_list):
        if K <= anchor_k:
            continue
        K_prev = K_list[i - 1]
        Q_curr_aligned, perm = align_to_reference(
            result_ref[K_prev], ref_by_k[K], allow_partial=True)
        result[K]     = Q_by_k[K][:, perm]
        result_ref[K] = ref_by_k[K][:, perm]

    # Propagate downward (decreasing K, if anchor > K_min)
    for i in range(K_list.index(anchor_k) - 1, -1, -1):
        K      = K_list[i]
        K_next = K_list[i + 1]
        # When going from K+1 → K, K has fewer columns → partial match from K+1 side
        Q_curr_aligned, perm = align_to_reference(
            result_ref[K_next][:, :K], ref_by_k[K], allow_partial=False)
        result[K]     = Q_curr_aligned
        result_ref[K] = result_ref[K_next][:, :K]   # use top-K components of K+1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline: load → within-K align → across-K align
# ─────────────────────────────────────────────────────────────────────────────

def run_clumpak_lite(result_dir: str,
                     bname: str,
                     K_list: List[int],
                     seeds: List[int],
                     out_dir: str,
                     use_P_for_across_k: bool = True
                     ) -> Dict[int, np.ndarray]:
    """
    Full CLUMPAK-lite pipeline.

    1. Load Q (and optionally P) files from result_dir.
    2. Within-K alignment: align multiple seeds, save centroid Q.
    3. Across-K alignment: align K_min..K_max, save structure-plot-ready Q.

    Output files written to out_dir:
        {bname}.K{K}.Q_aligned   — within-K centroid (aligned)
        {bname}.K{K}.Q_struct    — across-K structure-plot-ready

    Returns
    -------
    Q_struct : dict {K: (N, K) aligned Q matrix for structure plot}
    """
    import os, json

    os.makedirs(out_dir, exist_ok=True)

    Q_rep_by_k: Dict[int, np.ndarray] = {}
    P_rep_by_k: Dict[int, np.ndarray] = {}
    within_k_stats = {}

    # ── Step 1: Within-K alignment ──────────────────────────────────────────
    print("Within-K alignment:")
    for K in K_list:
        Q_list, P_list = [], []
        for s in seeds:
            qpath = os.path.join(result_dir, f"{bname}.K{K}.s{s}.Q")
            ppath = os.path.join(result_dir, f"{bname}.K{K}.s{s}.P")
            if not os.path.exists(qpath):
                continue
            Q_list.append(np.loadtxt(qpath))
            if os.path.exists(ppath):
                P_list.append(np.loadtxt(ppath))

        if not Q_list:
            print(f"  K={K}: no Q files found, skipping")
            continue

        Q_rep, Q_aligned_list, D, P_rep = align_within_k(Q_list, P_list if P_list else None)
        Q_rep_by_k[K] = Q_rep
        if P_rep is not None:
            P_rep_by_k[K] = P_rep

        # Save within-K centroid
        out_q = os.path.join(out_dir, f"{bname}.K{K}.Q_aligned")
        np.savetxt(out_q, Q_rep, fmt='%.6f')

        mean_rmse = D[D > 0].mean() if D.size > 1 else 0.0
        within_k_stats[K] = {'n_runs': len(Q_list), 'mean_pairwise_rmse': float(mean_rmse)}
        print(f"  K={K}: {len(Q_list)} runs, mean pairwise RMSE={mean_rmse:.4f} → {out_q}")

    if not Q_rep_by_k:
        print("No Q matrices found.")
        return {}

    # ── Step 2: Across-K alignment ──────────────────────────────────────────
    print("\nAcross-K alignment:")
    signal = P_rep_by_k if (use_P_for_across_k and P_rep_by_k) else None
    Q_struct = align_across_k(Q_rep_by_k, P_by_k=signal)

    for K, Q in sorted(Q_struct.items()):
        out_s = os.path.join(out_dir, f"{bname}.K{K}.Q_struct")
        np.savetxt(out_s, Q, fmt='%.6f')
        print(f"  K={K}: saved {out_s}")

    # Save alignment report
    report = {
        'K_list':       K_list,
        'seeds':        seeds,
        'within_k':     within_k_stats,
        'across_k_note': 'P-matrix based' if signal else 'Q-matrix based',
    }
    with open(os.path.join(out_dir, 'alignment_report.json'), 'w') as f:
        import json as _json
        _json.dump(report, f, indent=2)
    print(f"\nAlignment report: {os.path.join(out_dir, 'alignment_report.json')}")

    return Q_struct
