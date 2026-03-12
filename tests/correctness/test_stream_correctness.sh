#!/usr/bin/env bash
# Correctness gate for E1 STREAM Triad across all abstraction implementations.
#
# Rules (project_spec.md §7):
#   - Each abstraction must produce output matching the CUDA native reference
#     within 1e-6 relative error (STREAM_CORRECT_TOL in stream_common.h).
#   - The test uses the Small problem size (2^20) to run quickly.
#   - MUST PASS before any timing run is accepted.
#
# Usage:
#   ./tests/correctness/test_stream_correctness.sh --platform nvidia_a100
#   pytest tests/correctness/ -v --experiment E1    (if invoked via pytest wrapper)
#
# Exit codes: 0 = all pass, 1 = one or more failures

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/stream"

PLATFORM="nvidia_a100"
ARRAYSIZE=1048576    # Small: 2^20 — fast correctness run
PASS_COUNT=0
FAIL_COUNT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --arraysize) ARRAYSIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

check_abstraction() {
    local abs="$1"
    local bin_name="$2"
    local exe="${BUILD_BASE}/${bin_name#stream-}_${PLATFORM}/${bin_name}"

    printf "  %-12s " "${abs}"

    if [[ ! -x "${exe}" ]]; then
        printf "${RED}SKIP${NC} (not built: %s)\n" "${exe}"
        return
    fi

    # Run with small array — correctness check is embedded in the binary
    local tmpout
    tmpout=$(mktemp)
    "${exe}" --arraysize "${ARRAYSIZE}" --numtimes 1 --warmup 1 \
             --all-kernels > "${tmpout}" 2>&1 || true

    if grep -q "STREAM_CORRECT PASS" "${tmpout}"; then
        local max_err
        max_err=$(grep "STREAM_CORRECT PASS" "${tmpout}" \
                  | grep -oP 'max_err_a=\K[\d.eE+\-]+' | head -1)
        printf "${GREEN}PASS${NC} (max_err_a=%s)\n" "${max_err:-?}"
        (( PASS_COUNT++ )) || true
    elif grep -q "STREAM_CORRECT FAIL" "${tmpout}"; then
        printf "${RED}FAIL${NC}\n"
        grep "STREAM_CORRECT" "${tmpout}" | sed 's/^/    /'
        (( FAIL_COUNT++ )) || true
    else
        printf "${RED}ERROR${NC} (no STREAM_CORRECT line in output)\n"
        tail -5 "${tmpout}" | sed 's/^/    /'
        (( FAIL_COUNT++ )) || true
    fi
    rm -f "${tmpout}"
}

echo "============================================================"
echo " E1 STREAM Triad — Correctness Gate"
echo " Platform: ${PLATFORM}   Array size: ${ARRAYSIZE} (2^20)"
echo " Tolerance: 1e-6 relative error"
echo "============================================================"

# Map abstraction → binary name (mirrors run_stream.sh)
declare -A BINARY_NAME
BINARY_NAME[native]="stream-cuda"
BINARY_NAME[kokkos]="stream-kokkos"
BINARY_NAME[raja]="stream-raja"
BINARY_NAME[sycl]="stream-sycl"
[[ "${PLATFORM}" == "amd_mi250x" ]] && BINARY_NAME[native]="stream-hip"

for abs in native kokkos raja sycl; do
    check_abstraction "${abs}" "${BINARY_NAME[$abs]}"
done

# Julia and Numba have separate correctness scripts (Python/Julia invocation)
echo ""
echo "Note: Julia and Numba correctness tested via separate scripts (TODO)"
echo "------------------------------------------------------------"

if [[ ${FAIL_COUNT} -eq 0 && ${PASS_COUNT} -gt 0 ]]; then
    echo -e "${GREEN}ALL TESTS PASSED${NC} (${PASS_COUNT} passed, 0 failed)"
    exit 0
elif [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo -e "${RED}${FAIL_COUNT} TEST(S) FAILED${NC} (${PASS_COUNT} passed)"
    exit 1
else
    echo "No tests ran — check that binaries are built"
    exit 1
fi
