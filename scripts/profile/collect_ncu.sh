#!/usr/bin/env bash
# Collect NVIDIA Nsight Compute kernel-level counters for a flagged configuration.
# Captures: occupancy, memory throughput, L2 hit rate, warp divergence, register usage.
#
# Usage:
#   ./scripts/profile/collect_ncu.sh \
#       --kernel stream --abstraction kokkos --size large --platform nvidia_a100

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

KERNEL=""; ABSTRACTION=""; SIZE="large"; PLATFORM="nvidia_a100"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)      KERNEL="$2";      shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --platform)    PLATFORM="$2";    shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "${KERNEL}" || -z "${ABSTRACTION}" ]] && { echo "ERROR: --kernel and --abstraction required"; exit 1; }

EXE="${REPO_ROOT}/build/${KERNEL}/${ABSTRACTION}_${PLATFORM}/${KERNEL}-${ABSTRACTION}"
OUT_DIR="${REPO_ROOT}/results/${PLATFORM}/${KERNEL}/profiles"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/ncu_${ABSTRACTION}_${SIZE}"

echo "[collect_ncu] Profiling ${KERNEL}/${ABSTRACTION}/${SIZE}..."
ncu \
    --set full \
    --export "${REPORT}" \
    --force-overwrite \
    --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        lts__t_sectors.avg.pct_of_peak_sustained_elapsed,\
        smsp__sass_average_branch_targets_threads_uniform.pct,\
        sm__warps_active.avg.pct_of_peak_sustained_active \
    "${EXE}" --size "${SIZE}"

echo "[collect_ncu] Report: ${REPORT}.ncu-rep"

# Export CSV for data/profiling_metrics.csv ingestion
ncu --import "${REPORT}.ncu-rep" \
    --csv \
    --page details \
    > "${REPORT}.csv" 2>/dev/null || true
echo "[collect_ncu] CSV: ${REPORT}.csv"
