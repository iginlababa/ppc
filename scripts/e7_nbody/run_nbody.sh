#!/usr/bin/env bash
# Run all E7 N-Body benchmarks and write raw CSV to results/e7_nbody/raw/.
#
# Abstractions × kernels:
#   native_notile, native_tile, kokkos, raja, julia
# Sizes: small (4K), medium (32K), large (256K)
# Reps:  30 timed runs, 50 warmup (not timed, in binary)
#
# Usage:
#   ./scripts/e7_nbody/run_nbody.sh [--platform nvidia_rtx5060_laptop] [--reps 30]
#                                   [--sizes "small medium large"]
#                                   [--abstractions "native_notile native_tile kokkos raja julia"]
#                                   [--verify]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/e7_nbody"
RESULTS_DIR="${REPO_ROOT}/results/e7_nbody/raw"
mkdir -p "${RESULTS_DIR}"

PLATFORM="nvidia_rtx5060_laptop"
REPS=30
SIZES="small medium large"
ABSTRACTIONS="native_notile native_tile kokkos raja julia"
VERIFY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)     PLATFORM="$2"; shift 2 ;;
        --reps)         REPS="$2"; shift 2 ;;
        --sizes)        SIZES="$2"; shift 2 ;;
        --abstractions) ABSTRACTIONS="$2"; shift 2 ;;
        --verify)       VERIFY=true; shift ;;
        *) echo "[run_nbody] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DATE_STR="$(date +%Y%m%d)"
RAW_CSV="${RESULTS_DIR}/e7_nbody_${PLATFORM}_${DATE_STR}.csv"

# CSV header
echo "platform,abstraction,kernel,problem_size,n_atoms,run_id,n_nbrs_total,max_nbrs_per_atom,mean_nbrs_per_atom,actual_flops,time_ms,throughput_gflops,hw_state_verified" \
    > "${RAW_CSV}"

echo "[run_nbody] Platform: ${PLATFORM}"
echo "[run_nbody] Output:   ${RAW_CSV}"
echo "[run_nbody] Reps:     ${REPS}"
echo "[run_nbody] Abstractions: ${ABSTRACTIONS}"
echo "[run_nbody] Sizes:    ${SIZES}"
echo ""

# ── Helper: run one abstraction × size combo ──────────────────────────────────
run_one() {
    local abs="$1"
    local sz="$2"
    local bin="$3"
    local kernel_flag="$4"   # --kernel notile or --kernel tile or ""

    echo "[run_nbody] Running ${abs} / ${sz} ..."
    local verify_flag=""
    [[ "${VERIFY}" == "true" ]] && verify_flag="--verify"

    # Build invocation
    local cmd_args="--size ${sz} --reps ${REPS} --platform ${PLATFORM} ${verify_flag}"
    [[ -n "${kernel_flag}" ]] && cmd_args="${cmd_args} ${kernel_flag}"

    # Capture output; parse NBODY_META, NBODY_RUN, NBODY_HW_STATE lines
    local output
    output="$("${bin}" ${cmd_args} 2>&1)" || { echo "[run_nbody] ERROR: ${abs}/${sz} failed"; return 1; }

    # Extract metadata
    local n_atoms n_nbrs_total max_nbrs mean_nbrs hw_state
    n_atoms=$(echo "${output}" | grep '^NBODY_META ' | sed 's/.*n=\([0-9]*\).*/\1/')
    n_nbrs_total=$(echo "${output}" | grep '^NBODY_META ' | sed 's/.*n_nbrs_total=\([0-9]*\).*/\1/')
    max_nbrs=$(echo "${output}" | grep '^NBODY_META ' | sed 's/.*max_nbrs_per_atom=\([0-9]*\).*/\1/')
    mean_nbrs=$(echo "${output}" | grep '^NBODY_META ' | sed 's/.*mean_nbrs=\([0-9.]*\).*/\1/')
    hw_state=$(echo "${output}" | grep '^NBODY_HW_STATE ' | sed 's/.*state=\([01]\).*/\1/')
    hw_state="${hw_state:-0}"

    # Determine kernel label for CSV
    local kernel_col
    case "${abs}" in
        native_notile) kernel_col="notile" ;;
        native_tile)   kernel_col="tile" ;;
        *)             kernel_col="notile" ;;
    esac

    # Parse each NBODY_RUN line
    while IFS= read -r line; do
        [[ "${line}" != NBODY_RUN* ]] && continue
        local run_id actual_flops time_ms gflops
        run_id=$(echo "${line}" | sed 's/.*run=\([0-9]*\).*/\1/')
        actual_flops=$(echo "${line}" | sed 's/.*actual_flops=\([0-9.]*\).*/\1/')
        time_ms=$(echo "${line}" | sed 's/.*time_ms=\([0-9.]*\).*/\1/')
        gflops=$(echo "${line}" | sed 's/.*throughput_gflops=\([0-9.]*\).*/\1/')
        echo "${PLATFORM},${abs},${kernel_col},${sz},${n_atoms},${run_id},${n_nbrs_total},${max_nbrs},${mean_nbrs},${actual_flops},${time_ms},${gflops},${hw_state}" \
            >> "${RAW_CSV}"
    done <<< "${output}"

    # Show verify result if present (grep may exit 1 when no --verify flag — suppress)
    echo "${output}" | { grep '^NBODY_VERIFY' || true; } | while IFS= read -r vline; do
        echo "[run_nbody]   ${vline}"
    done

    echo "[run_nbody]   ${abs}/${sz} done"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for ABS in ${ABSTRACTIONS}; do
    for SZ in ${SIZES}; do

        case "${ABS}" in
            native_notile)
                BIN="${BUILD_BASE}/native_${PLATFORM}/nbody-native"
                KFLAG="--kernel notile"
                ;;
            native_tile)
                BIN="${BUILD_BASE}/native_${PLATFORM}/nbody-native"
                KFLAG="--kernel tile"
                ;;
            kokkos)
                BIN="${BUILD_BASE}/kokkos_${PLATFORM}/nbody-kokkos"
                KFLAG=""
                ;;
            raja)
                BIN="${BUILD_BASE}/raja_${PLATFORM}/nbody-raja"
                KFLAG=""
                ;;
            julia)
                BIN="${BUILD_BASE}/julia_${PLATFORM}/nbody-julia"
                KFLAG=""
                ;;
            *)
                echo "[run_nbody] Unknown abstraction: ${ABS}" >&2
                continue
                ;;
        esac

        if [[ ! -x "${BIN}" ]]; then
            echo "[run_nbody] SKIP ${ABS}/${SZ}: binary not found (${BIN})"
            continue
        fi

        run_one "${ABS}" "${SZ}" "${BIN}" "${KFLAG}"
    done
done

echo ""
echo "[run_nbody] All runs complete."
echo "[run_nbody] Raw CSV: ${RAW_CSV}"
