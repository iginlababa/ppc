#!/usr/bin/env bash
# Run E6 BFS benchmarks.
# Usage: ./scripts/run/run_bfs.sh [--platform nvidia_rtx5060_laptop] [--reps 30]
#
# Loops: abstraction × graph_type × problem_size
# Emits raw CSVs to data/raw/bfs_{abs}_{platform}_{date}.csv
# Also emits frontier profile to data/raw/bfs_profile_{platform}_{date}.csv
#
# CSV columns (raw):
#   timestamp, experiment_id, kernel, abstraction, platform, graph_type,
#   problem_size, n_vertices, n_edges, n_levels, max_frontier_width,
#   min_frontier_width, peak_frontier_fraction, run_id, execution_time_ms,
#   throughput_gflops, hw_state_verified
#
# Note: throughput_gflops stores GTEPS = n_edges / time_s / 1e9 for BFS.
#       "gflops" naming is for CSV schema consistency with E2-E5.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin/bfs"
DATA_RAW="${REPO_ROOT}/data/raw"
mkdir -p "${DATA_RAW}"

PLATFORM="nvidia_rtx5060_laptop"
REPS=30
ABSTRACTION="all"
DATE=$(date +%Y%m%d)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        *) echo "[run_bfs] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Abstractions and sizes ─────────────────────────────────────────────────────
ABSTRACTIONS=("native" "kokkos" "raja" "julia")
GRAPH_TYPES=("erdos_renyi" "2d_grid")
declare -A SIZES=([small]=1024 [medium]=16384 [large]=65536)
SIZE_ORDER=("small" "medium" "large")

# ── CSV header ────────────────────────────────────────────────────────────────
CSV_HEADER="timestamp,experiment_id,kernel,abstraction,platform,graph_type,problem_size,n_vertices,n_edges,n_levels,max_frontier_width,min_frontier_width,peak_frontier_fraction,run_id,execution_time_ms,throughput_gflops,hw_state_verified"

PROFILE_HEADER="timestamp,platform,graph_type,problem_size,n_vertices,n_levels,frontier_widths"

# ── Parse BFS_RUN output line into CSV ────────────────────────────────────────
# BFS_RUN run=N n_vertices=V n_edges=E n_levels=L max_fw=M min_fw=m
#         peak_ff=F graph=TYPE size=S time_ms=T throughput_gflops=G
# BFS_HW_STATE state=H
write_csv_from_output() {
    local output="$1"
    local abs_name="$2"
    local graph_type="$3"
    local size_label="$4"
    local csv_file="$5"
    local profile_file="$6"

    local hw_state=1
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local profile_written=false

    while IFS= read -r line; do
        if [[ "${line}" =~ ^BFS_PROFILE ]]; then
            if [[ "${profile_written}" == "false" ]]; then
                local nl widths
                nl=$(echo "${line}" | grep -oP 'n_levels=\K[0-9]+')
                widths=$(echo "${line}" | grep -oP 'widths=\K[^[:space:]]+')
                local n_v="${SIZES[${size_label}]}"
                echo "${ts},${PLATFORM},${graph_type},${size_label},${n_v},${nl},${widths}" \
                    >> "${profile_file}"
                profile_written=true
            fi
            continue
        fi

        if [[ "${line}" =~ ^BFS_HW_STATE ]]; then
            hw_state=$(echo "${line}" | grep -oP 'state=\K[01]')
            continue
        fi

        if [[ "${line}" =~ ^BFS_RUN ]]; then
            local run n_v n_e nl max_fw min_fw peak_ff t_ms gteps
            run=$(    echo "${line}" | grep -oP 'run=\K[0-9]+')
            n_v=$(    echo "${line}" | grep -oP 'n_vertices=\K[0-9]+')
            n_e=$(    echo "${line}" | grep -oP 'n_edges=\K[0-9]+')
            nl=$(     echo "${line}" | grep -oP 'n_levels=\K[0-9]+')
            max_fw=$( echo "${line}" | grep -oP 'max_fw=\K[0-9]+')
            min_fw=$( echo "${line}" | grep -oP 'min_fw=\K[0-9]+')
            peak_ff=$(echo "${line}" | grep -oP 'peak_ff=\K[0-9.]+')
            t_ms=$(   echo "${line}" | grep -oP 'time_ms=\K[0-9.]+')
            gteps=$(  echo "${line}" | grep -oP 'throughput_gflops=\K[0-9.]+')

            local exp_id="bfs_${abs_name}_${PLATFORM}_${graph_type}_${size_label}_n${n_v}_$(printf '%03d' "${run}")"
            echo "${ts},${exp_id},bfs,${abs_name},${PLATFORM},${graph_type},${size_label},${n_v},${n_e},${nl},${max_fw},${min_fw},${peak_ff},${run},${t_ms},${gteps},${hw_state}" \
                >> "${csv_file}"
        fi
    done <<< "${output}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs_name in "${ABSTRACTIONS[@]}"; do
    if [[ "${ABSTRACTION}" != "all" && "${ABSTRACTION}" != "${abs_name}" ]]; then
        continue
    fi

    BIN="${BIN_DIR}/bfs-${abs_name}"
    if [[ ! -x "${BIN}" ]]; then
        echo "[run_bfs] SKIP ${abs_name}: binary not found at ${BIN}"
        continue
    fi

    CSV_FILE="${DATA_RAW}/bfs_${abs_name}_${PLATFORM}_${DATE}.csv"
    PROFILE_FILE="${DATA_RAW}/bfs_profile_${PLATFORM}_${DATE}.csv"

    # Write headers if files are new
    [[ ! -f "${CSV_FILE}" ]]     && echo "${CSV_HEADER}"     > "${CSV_FILE}"
    [[ ! -f "${PROFILE_FILE}" ]] && echo "${PROFILE_HEADER}" > "${PROFILE_FILE}"

    for graph_type in "${GRAPH_TYPES[@]}"; do
        for size_label in "${SIZE_ORDER[@]}"; do
            N="${SIZES[${size_label}]}"
            echo "[run_bfs] Running ${abs_name} graph=${graph_type} size=${size_label} n=${N} reps=${REPS}"

            OUTPUT=$("${BIN}" \
                --graph "${graph_type}" \
                --n "${N}" \
                --reps "${REPS}" \
                --platform "${PLATFORM}" \
                2>/dev/null)

            write_csv_from_output \
                "${OUTPUT}" "${abs_name}" "${graph_type}" "${size_label}" \
                "${CSV_FILE}" "${PROFILE_FILE}"

            echo "[run_bfs] Done ${abs_name} ${graph_type} ${size_label}"
        done
    done
    echo "[run_bfs] ${abs_name} complete → ${CSV_FILE}"
done

echo "[run_bfs] All done.  Raw CSVs:"
ls -lh "${DATA_RAW}"/bfs_*"${PLATFORM}"*"${DATE}"* 2>/dev/null || true
