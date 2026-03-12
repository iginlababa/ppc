#!/usr/bin/env bash
# Collect AMD rocprof profiling data for a flagged configuration.
#
# Usage:
#   ./scripts/profile/collect_rocprof.sh \
#       --kernel stream --abstraction kokkos --size large --platform amd_mi250x

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

KERNEL=""; ABSTRACTION=""; SIZE="large"; PLATFORM="amd_mi250x"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)      KERNEL="$2";      shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --platform)    PLATFORM="$2";    shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

EXE="${REPO_ROOT}/build/${KERNEL}/${ABSTRACTION}_${PLATFORM}/${KERNEL}-${ABSTRACTION}"
OUT_DIR="${REPO_ROOT}/results/${PLATFORM}/${KERNEL}/profiles"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/rocprof_${ABSTRACTION}_${SIZE}"

# Counters: memory bandwidth, L2 hit, wavefront divergence, occupancy
COUNTERS_FILE="${REPO_ROOT}/scripts/profile/rocprof_counters.txt"
if [[ ! -f "${COUNTERS_FILE}" ]]; then
    cat > "${COUNTERS_FILE}" <<'EOF'
pmc: FETCH_SIZE WRITE_SIZE L2CacheHit
pmc: WAVES_DISPATCHED INACTIVE_CUS
pmc: SQ_WAVES SQ_BUSY_CYCLES
EOF
fi

echo "[collect_rocprof] Profiling ${KERNEL}/${ABSTRACTION}/${SIZE} on ${PLATFORM}..."
rocprof \
    --input "${COUNTERS_FILE}" \
    --output-dir "${OUT_DIR}" \
    --basenames off \
    "${EXE}" --size "${SIZE}"

echo "[collect_rocprof] Results in ${OUT_DIR}/results.csv"
