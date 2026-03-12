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

# Map abstraction name → binary name (built by kernels/stream/CMakeLists.txt).
# "native" on NVIDIA = CUDA binary; on AMD = HIP binary (future).
declare -A BINARY_NAME
BINARY_NAME[native]="stream-cuda"   # NVIDIA; will be stream-hip on AMD
BINARY_NAME[kokkos]="stream-kokkos"
BINARY_NAME[raja]="stream-raja"
BINARY_NAME[sycl]="stream-sycl"
BINARY_NAME[julia]="stream-julia"   # Actually a Julia script — handled separately

ABSTRACTIONS=(native kokkos raja sycl julia)
[[ "${ABSTRACTION}" != "all" ]] && ABSTRACTIONS=("${ABSTRACTION}")

# Override native binary for AMD
if [[ "${PLATFORM}" == "amd_mi250x" ]]; then
    BINARY_NAME[native]="stream-hip"
fi

SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/stream"
mkdir -p "${OUT_DIR}"

run_one() {
    local abs="$1" sz="$2"
    local n="${SIZES[$sz]}"
    local bin_name="${BINARY_NAME[$abs]:-stream-${abs}}"

    # Build directory layout: build/stream/<binary>_<platform>/
    local build_dir="${BUILD_BASE}/${bin_name#stream-}_${PLATFORM}"
    local exe="${build_dir}/${bin_name}"
    local outfile="${OUT_DIR}/${abs}_${sz}.out"

    if [[ ! -x "${exe}" ]]; then
        echo "  WARNING: Executable not found: ${exe} — skipping ${abs}/${sz}"
        echo "  Build with: ./scripts/build/build_stream.sh --platform ${PLATFORM}"
        return
    fi

    echo "  Running ${abs}/${sz} (N=${n}, reps=${REPS})..."
    # Warmup and timed runs are handled inside the binary via --warmup / --numtimes.
    # We run the binary ONCE — it internally does 10 warmup + REPS timed iterations
    # and outputs one STREAM_RUN line per timed iteration.
    "${exe}" \
        --arraysize "${n}" \
        --numtimes  "${REPS}" \
        --warmup    10 \
        > "${outfile}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "  ERROR: ${abs}/${sz} exited with rc=${rc} — see ${outfile}"
    else
        echo "  → ${outfile}"
    fi
}

echo "[run_stream] Platform=${PLATFORM}, reps=${REPS}"
echo "[run_stream] Abstractions: ${ABSTRACTIONS[*]}"
echo "[run_stream] Sizes: ${SIZE_LIST[*]}"

# Run abstractions in declared order (native is first by default).
# Native must always run before abstraction variants — it is the PPC reference.
for abs in "${ABSTRACTIONS[@]}"; do
    for sz in "${SIZE_LIST[@]}"; do
        run_one "${abs}" "${sz}"
    done
done

echo "[run_stream] Done."
echo "[run_stream] Parse with:"
echo "  python scripts/parse/parse_results.py \\"
echo "    --results-dir results/${PLATFORM}/ \\"
echo "    --kernel stream \\"
echo "    --output data/performance.csv"
