#!/usr/bin/env bash
# Run DGEMM (E2). See run_stream.sh for pattern.
# Usage: ./scripts/run/run_dgemm.sh --platform nvidia_a100 --reps 30

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/dgemm"
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

declare -A SIZES; SIZES[small]=1024; SIZES[medium]=4096; SIZES[large]=16384
ABSTRACTIONS=(native kokkos raja sycl julia)
[[ "${ABSTRACTION}" != "all" ]] && ABSTRACTIONS=("${ABSTRACTION}")
SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/dgemm"
mkdir -p "${OUT_DIR}"

echo "[run_dgemm] Platform=${PLATFORM}, reps=${REPS}"
for abs in "${ABSTRACTIONS[@]}"; do
    for sz in "${SIZE_LIST[@]}"; do
        n="${SIZES[$sz]}"
        exe="${BUILD_BASE}/${abs}_${PLATFORM}/rajaperf"
        outfile="${OUT_DIR}/${abs}_${sz}.out"
        if [[ ! -x "${exe}" ]]; then
            echo "  WARNING: ${exe} not found — skipping"; continue
        fi
        echo "  Running ${abs}/${sz} (N=${n})..."
        # Warm-up
        for _ in $(seq 1 10); do "${exe}" -k POLYBENCH_GEMM -n "${n}" &>/dev/null; done
        > "${outfile}"
        for i in $(seq 1 "${REPS}"); do "${exe}" -k POLYBENCH_GEMM -n "${n}" >> "${outfile}" 2>&1; done
        echo "  → ${outfile}"
    done
done
echo "[run_dgemm] Done."
