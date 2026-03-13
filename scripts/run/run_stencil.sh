#!/usr/bin/env bash
# Run 3D Stencil (E3).
# Usage: ./scripts/run/run_stencil.sh --platform nvidia_a100 --reps 30
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/stencil"
RESULTS_BASE="${REPO_ROOT}/results"

PLATFORM="nvidia_rtx5060_laptop"; ABSTRACTION="all"; SIZE="all"; REPS=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

declare -A SIZES; SIZES[small]="64 64 64"; SIZES[medium]="256 256 256"; SIZES[large]="512 512 512"
ABSTRACTIONS=(native kokkos raja sycl julia)
[[ "${ABSTRACTION}" != "all" ]] && ABSTRACTIONS=("${ABSTRACTION}")
SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/stencil"
mkdir -p "${OUT_DIR}"

echo "[run_stencil] Platform=${PLATFORM}, reps=${REPS}"
for abs in "${ABSTRACTIONS[@]}"; do
    for sz in "${SIZE_LIST[@]}"; do
        dims="${SIZES[$sz]}"
        outfile="${OUT_DIR}/${abs}_${sz}.out"
        echo "  TODO: run ${abs}/${sz} (dims=${dims}) → ${outfile}"
    done
done
echo "[run_stencil] Placeholder — implement executable invocation."
