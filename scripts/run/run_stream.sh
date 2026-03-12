#!/usr/bin/env bash
# Run STREAM Triad (E1) — all or one abstraction, one or all sizes.
#
# Usage:
#   ./scripts/run/run_stream.sh --platform nvidia_a100 --reps 30
#   ./scripts/run/run_stream.sh --platform nvidia_a100 --abstraction kokkos --size large --reps 30

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/benchmarks/stream/config.yaml"
BUILD_BASE="${REPO_ROOT}/build/stream"
RESULTS_BASE="${REPO_ROOT}/results"

PLATFORM="nvidia_a100"
ABSTRACTION="all"
SIZE="all"
REPS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

declare -A SIZES
SIZES[small]=1048576
SIZES[medium]=67108864
SIZES[large]=268435456

ABSTRACTIONS=(native kokkos raja sycl julia)
[[ "${ABSTRACTION}" != "all" ]] && ABSTRACTIONS=("${ABSTRACTION}")

SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/stream"
mkdir -p "${OUT_DIR}"

run_one() {
    local abs="$1" sz="$2"
    local n="${SIZES[$sz]}"
    local exe="${BUILD_BASE}/${abs}_${PLATFORM}/stream-${abs}"
    local outfile="${OUT_DIR}/${abs}_${sz}.out"

    if [[ ! -x "${exe}" ]]; then
        echo "  WARNING: Executable not found: ${exe} — skipping"
        return
    fi

    echo "  Running ${abs}/${sz} (N=${n}, reps=${REPS})..."
    # Warm-up (10 runs, not recorded)
    for _ in $(seq 1 10); do "${exe}" --arraysize "${n}" --numtimes 1 &>/dev/null; done
    # Timed runs (30 runs, each logged)
    > "${outfile}"
    for i in $(seq 1 "${REPS}"); do
        "${exe}" --arraysize "${n}" --numtimes 1 >> "${outfile}" 2>&1
    done
    echo "  → ${outfile}"
}

echo "[run_stream] Platform=${PLATFORM}, reps=${REPS}"
for abs in "${ABSTRACTIONS[@]}"; do
    for sz in "${SIZE_LIST[@]}"; do
        run_one "${abs}" "${sz}"
    done
done
echo "[run_stream] Done. Parse with: python scripts/parse/parse_results.py --results-dir ${OUT_DIR%/stream}"
