#!/usr/bin/env bash
# ============================================================
# LD pruning pipeline v2 — optimised for speed
#   - Chromosomes already converted to BED: prune+extract from BED
#   - Remaining chromosomes: prune from VCF directly, then
#     extract to small pruned BED (skip writing large intermediate BED)
#   - ALL 22 autosomes run in parallel
# ============================================================
set -euo pipefail

DATADIR="/home/yanlin/admixture/20220422_3202_phased_SNV_INDEL_SV"
OUTDIR="/home/yanlin/admixture/ld_pruned"
TMPDIR="${OUTDIR}/tmp"
LOGDIR="${OUTDIR}/logs"
THREADS=6      # per-chromosome thread count (22 × 6 = 132 ≤ 128+SMT)
MEM=8000       # MB per plink job

mkdir -p "${TMPDIR}" "${LOGDIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── per-chromosome function ───────────────────────────────────
process_chr() {
    local CHR=$1
    local VCF="${DATADIR}/1kGP_high_coverage_Illumina.chr${CHR}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    local BED="${TMPDIR}/chr${CHR}.bed"
    local BIM="${TMPDIR}/chr${CHR}.bim"
    local FAM="${TMPDIR}/chr${CHR}.fam"
    local PRUNE_IN="${TMPDIR}/chr${CHR}_prune.prune.in"
    local PRUNED_BED="${TMPDIR}/chr${CHR}_pruned.bed"

    # Skip if final pruned output already exists
    if [[ -f "${PRUNED_BED}" && -f "${TMPDIR}/chr${CHR}_pruned.bim" ]]; then
        N=$(wc -l < "${TMPDIR}/chr${CHR}_pruned.bim")
        echo "[$(ts)] chr${CHR}: already done (${N} SNPs), skipping"
        return 0
    fi

    # ── Step A: generate prune.in list ───────────────────────
    if [[ -f "${PRUNE_IN}" ]]; then
        echo "[$(ts)] chr${CHR}: prune list exists, skipping LD pruning"
    elif [[ -f "${BED}" && -f "${BIM}" && -f "${FAM}" ]]; then
        # Fast path: existing BED → LD prune
        echo "[$(ts)] chr${CHR}: LD pruning from BED ($(wc -l < ${BIM}) SNPs)..."
        plink \
            --bfile "${TMPDIR}/chr${CHR}" \
            --indep-pairwise 200 25 0.3 \
            --out "${TMPDIR}/chr${CHR}_prune" \
            --threads ${THREADS} \
            --memory ${MEM} \
            > "${LOGDIR}/chr${CHR}_prune.log" 2>&1
    else
        # Direct path: VCF → LD prune (no intermediate BED written)
        echo "[$(ts)] chr${CHR}: LD pruning directly from VCF..."
        plink \
            --vcf "${VCF}" \
            --vcf-half-call missing \
            --double-id \
            --snps-only just-acgt \
            --maf 0.01 \
            --indep-pairwise 200 25 0.3 \
            --out "${TMPDIR}/chr${CHR}_prune" \
            --threads ${THREADS} \
            --memory ${MEM} \
            > "${LOGDIR}/chr${CHR}_prune.log" 2>&1
    fi

    local NPRUNE
    NPRUNE=$(wc -l < "${PRUNE_IN}")
    echo "[$(ts)] chr${CHR}: ${NPRUNE} SNPs in prune-in list"

    # ── Step B: extract pruned SNPs → small final BED ────────
    if [[ -f "${BED}" && -f "${BIM}" && -f "${FAM}" ]]; then
        # Extract from existing BED (fast)
        echo "[$(ts)] chr${CHR}: extracting from BED..."
        plink \
            --bfile "${TMPDIR}/chr${CHR}" \
            --extract "${PRUNE_IN}" \
            --make-bed \
            --out "${TMPDIR}/chr${CHR}_pruned" \
            --threads ${THREADS} \
            --memory ${MEM} \
            > "${LOGDIR}/chr${CHR}_extract.log" 2>&1
    else
        # Extract from VCF (second VCF read, but only writes small BED)
        echo "[$(ts)] chr${CHR}: extracting from VCF..."
        plink \
            --vcf "${VCF}" \
            --vcf-half-call missing \
            --double-id \
            --snps-only just-acgt \
            --maf 0.01 \
            --extract "${PRUNE_IN}" \
            --make-bed \
            --out "${TMPDIR}/chr${CHR}_pruned" \
            --threads ${THREADS} \
            --memory ${MEM} \
            > "${LOGDIR}/chr${CHR}_extract.log" 2>&1
    fi

    local NFINAL
    NFINAL=$(wc -l < "${TMPDIR}/chr${CHR}_pruned.bim")
    echo "[$(ts)] chr${CHR}: DONE — ${NFINAL} SNPs after LD pruning"
}

export -f process_chr ts
export DATADIR TMPDIR LOGDIR THREADS MEM

# ── Step 1: all 22 chromosomes in parallel ───────────────────
echo "=================================================="
echo " Step 1: LD pruning — all 22 autosomes in parallel"
echo "  Window=200 SNPs, Step=25, r2<0.3, MAF>=0.01    "
echo "=================================================="

PIDS=()
for CHR in $(seq 1 22); do
    process_chr "${CHR}" &
    PIDS+=($!)
done
echo "[$(ts)] Launched 22 parallel jobs, waiting..."
for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "WARNING: job ${PID} failed"; }
done
echo "[$(ts)] All chromosomes pruned."

# ── Step 2: verify all pruned BEDs exist ─────────────────────
echo ""
echo "Per-chromosome SNP counts after LD pruning:"
TOTAL=0
ALL_OK=1
for CHR in $(seq 1 22); do
    F="${TMPDIR}/chr${CHR}_pruned.bim"
    if [[ -f "${F}" ]]; then
        N=$(wc -l < "${F}")
        echo "  chr${CHR}: ${N}"
        TOTAL=$((TOTAL + N))
    else
        echo "  chr${CHR}: MISSING — check ${LOGDIR}/chr${CHR}_prune.log"
        ALL_OK=0
    fi
done
echo "  TOTAL: ${TOTAL}"
if [[ ${ALL_OK} -eq 0 ]]; then
    echo "ERROR: some chromosomes failed. Aborting merge." >&2
    exit 1
fi

# ── Step 3: merge all pruned chromosomes ─────────────────────
echo ""
echo "================================="
echo " Step 2: Merge all 22 autosomes  "
echo "================================="

> "${TMPDIR}/merge_list.txt"
for CHR in $(seq 2 22); do
    echo "${TMPDIR}/chr${CHR}_pruned"
done >> "${TMPDIR}/merge_list.txt"

plink \
    --bfile "${TMPDIR}/chr1_pruned" \
    --merge-list "${TMPDIR}/merge_list.txt" \
    --make-bed \
    --out "${OUTDIR}/all_autosomes_pruned" \
    --threads 32 \
    --memory 64000 \
    2>&1 | tee "${LOGDIR}/merge.log" | tail -10

MERGED=$(wc -l < "${OUTDIR}/all_autosomes_pruned.bim")
SAMPLES=$(wc -l < "${OUTDIR}/all_autosomes_pruned.fam")
echo "[$(ts)] Merged: ${MERGED} SNPs, ${SAMPLES} samples"

# ── Step 4: thin to exactly 1,000,000 SNPs ───────────────────
echo ""
echo "======================================="
echo " Step 3: Thin to exactly 1,000,000 SNPs"
echo "======================================="

FINAL="${OUTDIR}/1kGP_1M_ldpruned"

if [[ ${MERGED} -gt 1000000 ]]; then
    echo "[$(ts)] ${MERGED} > 1M — thinning..."
    plink \
        --bfile "${OUTDIR}/all_autosomes_pruned" \
        --thin-count 1000000 \
        --make-bed \
        --out "${FINAL}" \
        --threads 32 \
        --memory 64000 \
        2>&1 | tee "${LOGDIR}/thin.log" | tail -5
else
    echo "[$(ts)] ${MERGED} SNPs — keeping all (below 1M target)."
    cp "${OUTDIR}/all_autosomes_pruned.bed" "${FINAL}.bed"
    cp "${OUTDIR}/all_autosomes_pruned.bim" "${FINAL}.bim"
    cp "${OUTDIR}/all_autosomes_pruned.fam" "${FINAL}.fam"
fi

echo ""
echo "========================================================"
echo " DONE"
printf "  Final BED : %s.bed\n" "${FINAL}"
printf "  SNPs      : %d\n"  "$(wc -l < ${FINAL}.bim)"
printf "  Samples   : %d\n"  "$(wc -l < ${FINAL}.fam)"
echo "========================================================"
