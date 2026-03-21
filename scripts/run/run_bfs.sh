#!/usr/bin/env bash
# Run E6 BFS benchmarks.
# Usage: ./scripts/run/run_bfs.sh [--platform nvidia_rtx5060] [--reps 30]
#        ./scripts/run/run_bfs.sh [--platform amd_mi300x] [--abstraction kokkos] [--reps 30]
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
        *) echo "[run_bfs] Unknown argument: $1"; exit 1 ;;
    esac
done

VENDOR="${PLATFORM%%_*}"   # "nvidia" or "amd"

# ── Abstractions and sizes ─────────────────────────────────────────────────────
GRAPH_TYPES=("erdos_renyi" "2d_grid")
declare -A SIZES=([small]=1024 [medium]=16384 [large]=65536)
SIZE_ORDER=("small" "medium" "large")

# ── Platform-specific registry ────────────────────────────────────────────────
declare -A BINARY_NAME
declare -A BINARY_DIR

if [[ "${VENDOR}" == "amd" ]]; then
    BINARY_NAME[native]="bfs-hip"
    BINARY_DIR[native]="hip"
    ALL_ABSTRACTIONS=(native kokkos raja sycl julia)
    export JULIA_GPU_BACKEND="amdgpu"
else
    BINARY_NAME[native]="bfs-native"
    BINARY_DIR[native]="bin/bfs"   # flat structure for NVIDIA
    ALL_ABSTRACTIONS=(native kokkos raja julia)
    # numba: UNSUPPORTED_CC120; sycl: NO_COMPILER on RTX 5060
fi

BINARY_NAME[kokkos]="bfs-kokkos"; BINARY_DIR[kokkos]="kokkos"
BINARY_NAME[raja]="bfs-raja";     BINARY_DIR[raja]="raja"
BINARY_NAME[sycl]="bfs-sycl";     BINARY_DIR[sycl]="sycl"
BINARY_NAME[julia]="bfs-julia";   BINARY_DIR[julia]="julia"

# ── Binary finder ─────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]}"

    if [[ "${VENDOR}" == "amd" ]]; then
        # AMD: binaries in build/bfs/{dir}_{PLATFORM}/
        local dir_stem="${BINARY_DIR[$abs]:-${abs}}"
        local p="${REPO_ROOT}/build/bfs/${dir_stem}_${PLATFORM}/${bin_name}"
        [[ -x "${p}" ]] && { echo "${p}"; return 0; }
        # fallback: search all build subdirs
        for d in "${REPO_ROOT}/build/bfs/"/*/; do
            if [[ -x "${d}${bin_name}" ]]; then
                [[ "${d}" == *"${PLATFORM}"* ]] && { echo "${d}${bin_name}"; return 0; }
            fi
        done
        echo ""
    else
        # NVIDIA: flat bin/bfs/ directory
        local p="${REPO_ROOT}/bin/bfs/${bin_name}"
        [[ -x "${p}" ]] && { echo "${p}"; return 0; }
        echo ""
    fi
}

# ── CSV header ────────────────────────────────────────────────────────────────
CSV_HEADER="timestamp,experiment_id,kernel,abstraction,platform,graph_type,problem_size,n_vertices,n_edges,n_levels,max_frontier_width,min_frontier_width,peak_frontier_fraction,run_id,execution_time_ms,throughput_gflops,hw_state_verified"
PROFILE_HEADER="timestamp,platform,graph_type,problem_size,n_vertices,n_levels,frontier_widths"

# ── Parse BFS_RUN output line into CSV ────────────────────────────────────────
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
                echo "${ts},${PLATFORM},${graph_type},${size_label},${n_v},${nl},\"${widths}\"" \
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

# ── Build abstraction list ─────────────────────────────────────────────────────
if [[ "${ABSTRACTION}" == "all" ]]; then
    run_abstractions=("${ALL_ABSTRACTIONS[@]}")
else
    run_abstractions=("${ABSTRACTION}")
fi

echo "[run_bfs] ================================================================"
echo "[run_bfs] Platform:      ${PLATFORM}"
echo "[run_bfs] Reps:          ${REPS}"
echo "[run_bfs] Abstractions:  ${run_abstractions[*]}"
echo "[run_bfs] Graph types:   ${GRAPH_TYPES[*]}"
echo "[run_bfs] Sizes:         ${SIZE_ORDER[*]}"
if [[ "${VENDOR}" == "amd" ]]; then
    echo "[run_bfs] Note:          numba=SKIP (numba-hip experimental)"
    echo "[run_bfs] Note:          JULIA_GPU_BACKEND=amdgpu"
else
    echo "[run_bfs] Note:          numba=UNSUPPORTED_CC120, sycl=NO_COMPILER"
fi
echo "[run_bfs] ================================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs_name in "${run_abstractions[@]}"; do
    BIN="$(find_binary "${abs_name}")"
    if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
        echo "[run_bfs] SKIP ${abs_name}: binary not found (run build_bfs.sh first)"
        continue
    fi

    CSV_FILE="${DATA_RAW}/bfs_${abs_name}_${PLATFORM}_${DATE}.csv"
    PROFILE_FILE="${DATA_RAW}/bfs_profile_${PLATFORM}_${DATE}.csv"

    [[ ! -f "${CSV_FILE}" ]]     && echo "${CSV_HEADER}"     > "${CSV_FILE}"
    [[ ! -f "${PROFILE_FILE}" ]] && echo "${PROFILE_HEADER}" > "${PROFILE_FILE}"

    for graph_type in "${GRAPH_TYPES[@]}"; do
        for size_label in "${SIZE_ORDER[@]}"; do
            N="${SIZES[${size_label}]}"
            echo ""
            echo "[run_bfs] ── ${abs_name} / ${graph_type} / ${size_label} (N=${N}) ──────────────"
            echo "[run_bfs]    binary: ${BIN}"
            echo "[run_bfs]    csv:    ${CSV_FILE}"

            OUTPUT=$("${BIN}" \
                --graph "${graph_type}" \
                --n "${N}" \
                --reps "${REPS}" \
                --platform "${PLATFORM}" \
                2>/dev/null)

            write_csv_from_output \
                "${OUTPUT}" "${abs_name}" "${graph_type}" "${size_label}" \
                "${CSV_FILE}" "${PROFILE_FILE}"

            echo "[run_bfs]    → rows appended to ${CSV_FILE}"
        done
    done
    echo "[run_bfs] ${abs_name} complete → ${CSV_FILE}"
done

echo ""
echo "[run_bfs] ================================================================"
echo "[run_bfs] Done. CSV files:"
ls -lh "${DATA_RAW}"/bfs_*"${PLATFORM}"*"${DATE}"* 2>/dev/null || true
echo "[run_bfs] ================================================================"
echo "[run_bfs] Next step: python3 scripts/analysis/process_e6.py"
