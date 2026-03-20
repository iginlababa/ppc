#!/usr/bin/env bash
# Run E7 N-Body benchmarks.
# Usage: ./scripts/run/run_nbody.sh [--platform nvidia_rtx5060] [--reps 30]
#
# Loops: abstraction × kernel_variant × problem_size
# Emits raw CSVs to data/raw/nbody_{abs}_{platform}_{date}.csv
#
# CSV columns (raw):
#   timestamp, experiment_id, kernel, abstraction, platform, problem_size,
#   n_atoms, n_nbrs_mean, n_nbrs_max, actual_flops, run_id,
#   execution_time_ms, throughput_gflops, hw_state_verified
#
# Note: throughput_gflops = actual_flops / time_s / 1e9
#       For tile kernel: actual_flops = N*(N-1)*FLOPS_PER_PAIR (all-pairs)
#       For notile:      actual_flops = total_neighbor_pairs * FLOPS_PER_PAIR

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin/nbody"
DATA_RAW="${REPO_ROOT}/data/raw"
mkdir -p "${DATA_RAW}"

PLATFORM="nvidia_rtx5060"
REPS=30
ABSTRACTION="all"
DATE=$(date +%Y%m%d)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        *) echo "[run_nbody] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Abstractions and sizes ────────────────────────────────────────────────────
ABSTRACTIONS=("native" "julia")
KERNEL_VARIANTS=("notile" "tile")  # native only; julia has notile only
declare -A SIZES=([small]=4000 [medium]=32000 [large]=256000)
SIZE_ORDER=("small" "medium" "large")

CSV_HEADER="timestamp,experiment_id,kernel,abstraction,platform,problem_size,n_atoms,n_nbrs_mean,n_nbrs_max,actual_flops,run_id,execution_time_ms,throughput_gflops,hw_state_verified"

# ── Parse NBODY_STATS line ─────────────────────────────────────────────────────
# NBODY_STATS n_atoms=N n_nbrs_mean=M n_nbrs_min=n n_nbrs_max=X n_nbrs_std=S total_nbrs=T
parse_nbody_stats() {
    local line="$1"
    STATS_N_NBRS_MEAN=$(echo "${line}" | grep -oP 'n_nbrs_mean=\K[0-9.]+')
    STATS_N_NBRS_MAX=$(echo  "${line}" | grep -oP 'n_nbrs_max=\K[0-9]+')
    STATS_TOTAL_NBRS=$(echo  "${line}" | grep -oP 'total_nbrs=\K[0-9]+')
}
STATS_N_NBRS_MEAN="0"
STATS_N_NBRS_MAX="0"
STATS_TOTAL_NBRS="0"

# ── Parse output and write CSV ─────────────────────────────────────────────────
# NBODY_RUN run=R kernel=K n_atoms=N size=S time_ms=T throughput_gflops=G
# NBODY_HW_STATE state=H
write_csv_from_output() {
    local output="$1"
    local abs_name="$2"
    local size_label="$3"
    local csv_file="$4"

    local hw_state=1
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    while IFS= read -r line; do
        if [[ "${line}" =~ ^NBODY_STATS ]]; then
            parse_nbody_stats "${line}"
            continue
        fi

        if [[ "${line}" =~ ^NBODY_HW_STATE ]]; then
            hw_state=$(echo "${line}" | grep -oP 'state=\K[01]')
            continue
        fi

        if [[ "${line}" =~ ^NBODY_RUN ]]; then
            local run kernel n_atoms size t_ms gflops
            run=$(    echo "${line}" | grep -oP 'run=\K[0-9]+')
            kernel=$( echo "${line}" | grep -oP 'kernel=\K[a-z0-9_]+')
            n_atoms=$(echo "${line}" | grep -oP 'n_atoms=\K[0-9]+')
            size=$(   echo "${line}" | grep -oP 'size=\K[a-z]+')
            t_ms=$(   echo "${line}" | grep -oP 'time_ms=\K[0-9.]+')
            gflops=$( echo "${line}" | grep -oP 'throughput_gflops=\K[0-9.]+')

            # actual_flops for notile = total_nbrs * 20; tile = N*(N-1)*20
            local actual_flops
            if [[ "${kernel}" == "tile" ]]; then
                actual_flops=$(( n_atoms * (n_atoms - 1) * 20 ))
            else
                actual_flops=$(( STATS_TOTAL_NBRS * 20 ))
            fi

            local exp_id="nbody_${abs_name}_${kernel}_${PLATFORM}_${size_label}_n${n_atoms}_$(printf '%03d' "${run}")"
            echo "${ts},${exp_id},${kernel},${abs_name},${PLATFORM},${size_label},${n_atoms},${STATS_N_NBRS_MEAN},${STATS_N_NBRS_MAX},${actual_flops},${run},${t_ms},${gflops},${hw_state}" \
                >> "${csv_file}"
        fi
    done <<< "${output}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs_name in "${ABSTRACTIONS[@]}"; do
    if [[ "${ABSTRACTION}" != "all" && "${ABSTRACTION}" != "${abs_name}" ]]; then
        continue
    fi

    BIN="${BIN_DIR}/nbody-${abs_name}"
    if [[ ! -x "${BIN}" ]]; then
        echo "[run_nbody] SKIP ${abs_name}: binary not found at ${BIN}"
        continue
    fi

    CSV_FILE="${DATA_RAW}/nbody_${abs_name}_${PLATFORM}_${DATE}.csv"
    [[ ! -f "${CSV_FILE}" ]] && echo "${CSV_HEADER}" > "${CSV_FILE}"

    # Determine which kernel variants to run
    if [[ "${abs_name}" == "julia" ]]; then
        RUN_VARIANTS=("notile")   # julia: notile only
    else
        RUN_VARIANTS=("${KERNEL_VARIANTS[@]}")
    fi

    for size_label in "${SIZE_ORDER[@]}"; do
        N="${SIZES[${size_label}]}"

        for kv in "${RUN_VARIANTS[@]}"; do
            echo "[run_nbody] Running ${abs_name} kernel=${kv} size=${size_label} n=${N} reps=${REPS}"

            OUTPUT=$("${BIN}" \
                --n     "${N}" \
                --kernel "${kv}" \
                --size  "${size_label}" \
                --reps  "${REPS}" \
                --platform "${PLATFORM}" \
                2>/dev/null)

            write_csv_from_output "${OUTPUT}" "${abs_name}" "${size_label}" "${CSV_FILE}"
            echo "[run_nbody] Done ${abs_name} ${kv} ${size_label}"
        done
    done
    echo "[run_nbody] ${abs_name} complete → ${CSV_FILE}"
done

echo "[run_nbody] All done.  Raw CSVs:"
ls -lh "${DATA_RAW}"/nbody_*"${PLATFORM}"*"${DATE}"* 2>/dev/null || true
