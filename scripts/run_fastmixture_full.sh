#!/usr/bin/env bash
# Phase 4b: Run fastmixture for K=2..10 × 3 seeds in parallel.
# Each job uses 32 threads. Up to 4 jobs run simultaneously.

BFILE="1kGP_200k_ldpruned"
OUTDIR="results/fastmixture"
LOGDIR="logs"
SEEDS=(42 1 2)
K_LIST=(2 3 4 5 6 7 8 9 10)
THREADS=32

mkdir -p "$OUTDIR" "$LOGDIR"

pids=()
run_count=0
max_parallel=4

for K in "${K_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        logfile="${LOGDIR}/fastmixture_K${K}_s${seed}.log"
        outprefix="${OUTDIR}/${BFILE}"

        # fastmixture writes: PREFIX.K{K}.s{seed}.Q / .P / .log
        fastmixture \
            --bfile "$BFILE" \
            --K "$K" \
            --seed "$seed" \
            --out "$outprefix" \
            --threads "$THREADS" \
            > "$logfile" 2>&1 &

        pids+=($!)
        run_count=$((run_count + 1))
        echo "Started K=${K} seed=${seed} (pid=${!}) -> ${logfile}"

        # Throttle: wait for a slot when max_parallel jobs are running
        if (( run_count % max_parallel == 0 )); then
            echo "--- waiting for batch of ${max_parallel} to finish ---"
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
            pids=()
        fi
    done
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== fastmixture full run complete ==="
