#!/usr/bin/env bash
# Collect AMD Omniperf deep memory hierarchy analysis.
# Requires: omniperf >= 1.0, sudo or CAP_SYS_ADMIN
#
# Usage:
#   ./scripts/profile/collect_omniperf.sh \
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
OUT_DIR="${REPO_ROOT}/results/${PLATFORM}/${KERNEL}/profiles/omniperf_${ABSTRACTION}_${SIZE}"
mkdir -p "${OUT_DIR}"

echo "[collect_omniperf] Profiling ${KERNEL}/${ABSTRACTION}/${SIZE}..."
omniperf profile \
    --name "${KERNEL}_${ABSTRACTION}_${SIZE}" \
    --path "${OUT_DIR}" \
    -- "${EXE}" --size "${SIZE}"

echo "[collect_omniperf] To analyze: omniperf analyze --path ${OUT_DIR}"
