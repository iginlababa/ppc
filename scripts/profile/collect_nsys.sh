#!/usr/bin/env bash
# Collect NVIDIA Nsight Systems timeline profile for a flagged configuration.
# Triggered automatically when abstraction < 0.85 * native (§9.5).
#
# Usage:
#   ./scripts/profile/collect_nsys.sh \
#       --kernel stream --abstraction kokkos --size large \
#       --platform nvidia_a100

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
if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: Executable not found: ${EXE}"; exit 1
fi

OUT_DIR="${REPO_ROOT}/results/${PLATFORM}/${KERNEL}/profiles"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/nsys_${ABSTRACTION}_${SIZE}"

echo "[collect_nsys] Profiling ${KERNEL}/${ABSTRACTION}/${SIZE} on ${PLATFORM}..."
nsys profile \
    --output="${REPORT}" \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    "${EXE}" --size "${SIZE}"

echo "[collect_nsys] Report: ${REPORT}.nsys-rep"
echo "[collect_nsys] To view: nsys-ui ${REPORT}.nsys-rep"
echo "[collect_nsys] To export: nsys stats --report=gputrace ${REPORT}.nsys-rep"
