#!/usr/bin/env bash
# Run SpMV (E4).
# Usage: ./scripts/run/run_spmv.sh --platform nvidia_a100 --reps 30
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/spmv"
mkdir -p "${OUT_DIR}"

ABSTRACTIONS=(native kokkos raja sycl julia)
[[ "${ABSTRACTION}" != "all" ]] && ABSTRACTIONS=("${ABSTRACTION}")
MATRIX_VARIANTS=(structured power_law random)

echo "[run_spmv] Platform=${PLATFORM}, reps=${REPS}"
for abs in "${ABSTRACTIONS[@]}"; do
    for variant in "${MATRIX_VARIANTS[@]}"; do
        outfile="${OUT_DIR}/${abs}_${variant}.out"
        echo "  TODO: run ${abs}/${variant} → ${outfile}"
    done
done
echo "[run_spmv] Placeholder — implement matrix loading and kernel invocation."
