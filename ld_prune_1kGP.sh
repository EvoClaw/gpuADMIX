#!/usr/bin/env bash
# ============================================================
# LD pruning pipeline for 1000 Genomes Project high-coverage data
# Goal: ~1,000,000 biallelic autosomal SNPs (MAF >= 0.01)
#       after genome-wide LD pruning (r2 < 0.3)
# ============================================================
set -euo pipefail

DATADIR="/home/yanlin/admixture/20220422_3202_phased_SNV_INDEL_SV"
OUTDIR="/home/yanlin/admixture/ld_pruned"
TMPDIR="${OUTDIR}/tmp"
LOGDIR="${OUTDIR}/logs"

# Parallelism: N chromosomes at a time, THREADS threads each
PARALLEL=4
THREADS=16

mkdir -p "${TMPDIR}" "${LOGDIR}"

# ── lockfile guard (prevent multiple simultaneous runs) ──────
LOCKFILE="${OUTDIR}/.pipeline.lock"
exec 200>"${LOCKFILE}"
if ! flock -n 200; then
    echo "ERROR: Another instance of this pipeline is already running."
    echo "       Remove ${LOCKFILE} if this is a stale lock."
    exit 1
fi
trap "flock -u 200; rm -f ${LOCKFILE}" EXIT

# ── helper: timestamp ─────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── step 1: per-chromosome conversion + MAF filter ───────────
echo "=============================="
echo " Step 1: Convert VCFs to BED "
echo "=============================="

convert_chr() {
    local CHR=$1
    local VCF="${DATADIR}/1kGP_high_coverage_Illumina.chr${CHR}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    local OUT="${TMPDIR}/chr${CHR}"
    local LOG="${LOGDIR}/chr${CHR}_convert.log"

    # Skip if already successfully converted
    if [[ -f "${OUT}.bed" && -f "${OUT}.bim" && -f "${OUT}.fam" ]]; then
        local N
        N=$(wc -l < "${OUT}.bim")
        echo "[$(ts)] chr${CHR}: already done (${N} SNPs), skipping"
        return 0
    fi

    echo "[$(ts)] chr${CHR}: starting bcftools → plink conversion"

    # bcftools: biallelic SNPs only; plink: MAF >= 0.01
    bcftools view \
        --min-alleles 2 --max-alleles 2 \
        --type snps \
        --threads 4 \
        "${VCF}" \
    | plink \
        --vcf /dev/stdin \
        --vcf-half-call missing \
        --double-id \
        --maf 0.01 \
        --make-bed \
        --out "${OUT}" \
        --threads ${THREADS} \
        --memory 20000 \
        > "${LOG}" 2>&1

    local N
    N=$(wc -l < "${OUT}.bim")
    echo "[$(ts)] chr${CHR}: done — ${N} biallelic SNPs (MAF>=0.01)"
}

# Run in parallel batches
PIDS=()
for CHR in $(seq 1 22); do
    # Skip chr22 already done
    convert_chr "${CHR}" &
    PIDS+=($!)
    if [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done
# Wait for remaining
for PID in "${PIDS[@]}"; do
    wait "$PID"
done
echo "[$(ts)] All chromosomes converted."

# ── step 2: per-chromosome LD pruning ────────────────────────
echo ""
echo "========================================"
echo " Step 2: LD pruning per chromosome      "
echo "  Window=200 SNPs, Step=25, r2<0.3      "
echo "========================================"

PIDS=()
for CHR in $(seq 1 22); do
    (
        PRUNE_OUT="${TMPDIR}/chr${CHR}_prune"
        LOG="${LOGDIR}/chr${CHR}_prune.log"

        if [[ -f "${PRUNE_OUT}.prune.in" ]]; then
            N=$(wc -l < "${PRUNE_OUT}.prune.in")
            echo "[$(ts)] chr${CHR}: prune list already exists (${N} SNPs), skipping"
        else
            echo "[$(ts)] chr${CHR}: LD pruning..."
            plink \
                --bfile "${TMPDIR}/chr${CHR}" \
                --indep-pairwise 200 25 0.3 \
                --out "${PRUNE_OUT}" \
                --threads ${THREADS} \
                --memory 20000 \
                > "${LOG}" 2>&1
            N=$(wc -l < "${PRUNE_OUT}.prune.in")
            echo "[$(ts)] chr${CHR}: ${N} SNPs retained after LD pruning"
        fi
    ) &
    PIDS+=($!)
    if [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done
for PID in "${PIDS[@]}"; do
    wait "$PID"
done
echo "[$(ts)] LD pruning complete for all chromosomes."

# ── step 3: extract pruned SNPs per chromosome ───────────────
echo ""
echo "========================================"
echo " Step 3: Extract pruned SNPs per chr    "
echo "========================================"

PIDS=()
for CHR in $(seq 1 22); do
    (
        OUT="${TMPDIR}/chr${CHR}_pruned"
        LOG="${LOGDIR}/chr${CHR}_extract.log"

        if [[ -f "${OUT}.bed" ]]; then
            echo "[$(ts)] chr${CHR}: already extracted, skipping"
        else
            plink \
                --bfile "${TMPDIR}/chr${CHR}" \
                --extract "${TMPDIR}/chr${CHR}_prune.prune.in" \
                --make-bed \
                --out "${OUT}" \
                --threads 4 \
                --memory 8000 \
                > "${LOG}" 2>&1
            echo "[$(ts)] chr${CHR}: extraction done ($(wc -l < ${OUT}.bim) SNPs)"
        fi
    ) &
    PIDS+=($!)
    if [[ ${#PIDS[@]} -ge 22 ]]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo ""
echo "Per-chromosome SNP counts after LD pruning:"
TOTAL_PRUNED=0
for CHR in $(seq 1 22); do
    N=$(wc -l < "${TMPDIR}/chr${CHR}_pruned.bim")
    echo "  chr${CHR}: ${N}"
    TOTAL_PRUNED=$((TOTAL_PRUNED + N))
done
echo "  TOTAL: ${TOTAL_PRUNED}"

# ── step 4: merge all chromosomes ────────────────────────────
echo ""
echo "========================================"
echo " Step 4: Merge all autosomes            "
echo "========================================"

> "${TMPDIR}/merge_list.txt"
for CHR in $(seq 2 22); do
    echo "${TMPDIR}/chr${CHR}_pruned"
done >> "${TMPDIR}/merge_list.txt"

plink \
    --bfile "${TMPDIR}/chr1_pruned" \
    --merge-list "${TMPDIR}/merge_list.txt" \
    --make-bed \
    --out "${OUTDIR}/all_autosomes_pruned" \
    --threads ${THREADS} \
    --memory 128000 \
    2>&1 | tee "${LOGDIR}/merge.log" | tail -10

MERGED=$(wc -l < "${OUTDIR}/all_autosomes_pruned.bim")
echo "[$(ts)] Merged dataset: ${MERGED} SNPs, $(wc -l < ${OUTDIR}/all_autosomes_pruned.fam) samples"

# ── step 5: thin to exactly 1,000,000 SNPs ───────────────────
echo ""
echo "========================================"
echo " Step 5: Thin to 1,000,000 SNPs         "
echo "========================================"

FINAL_OUT="${OUTDIR}/1kGP_1M_ldpruned"

if [[ ${MERGED} -gt 1000000 ]]; then
    echo "[$(ts)] ${MERGED} > 1M — thinning to exactly 1,000,000 SNPs..."
    plink \
        --bfile "${OUTDIR}/all_autosomes_pruned" \
        --thin-count 1000000 \
        --make-bed \
        --out "${FINAL_OUT}" \
        --threads ${THREADS} \
        --memory 128000 \
        2>&1 | tee "${LOGDIR}/thin.log" | tail -5
else
    echo "[$(ts)] ${MERGED} <= 1M — using all pruned SNPs."
    cp "${OUTDIR}/all_autosomes_pruned.bed" "${FINAL_OUT}.bed"
    cp "${OUTDIR}/all_autosomes_pruned.bim" "${FINAL_OUT}.bim"
    cp "${OUTDIR}/all_autosomes_pruned.fam" "${FINAL_OUT}.fam"
fi

echo ""
echo "============================================================"
echo " DONE"
echo "  Final BED : ${FINAL_OUT}.bed"
echo "  SNP count : $(wc -l < ${FINAL_OUT}.bim)"
echo "  Samples   : $(wc -l < ${FINAL_OUT}.fam)"
echo "============================================================"
