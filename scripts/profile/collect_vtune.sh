#!/usr/bin/env bash
# Collect Intel VTune GPU offload analysis for Intel PVC.
#
# Usage:
#   ./scripts/profile/collect_vtune.sh \
#       --kernel stream --abstraction sycl --size large --platform intel_pvc

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

KERNEL=""; ABSTRACTION=""; SIZE="large"; PLATFORM="intel_pvc"

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
RESULT_DIR="${OUT_DIR}/vtune_${ABSTRACTION}_${SIZE}"

echo "[collect_vtune] Profiling ${KERNEL}/${ABSTRACTION}/${SIZE} on ${PLATFORM}..."
vtune \
    -collect gpu-offload \
    -result-dir "${RESULT_DIR}" \
    -app-working-dir "${REPO_ROOT}" \
    -- "${EXE}" --size "${SIZE}"

# Export CSV
vtune \
    -report summary \
    -result-dir "${RESULT_DIR}" \
    -format csv \
    -report-output "${RESULT_DIR}/summary.csv"

echo "[collect_vtune] Result: ${RESULT_DIR}"
