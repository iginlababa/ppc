#!/usr/bin/env bash
# Run E1 STREAM Triad — all abstractions, all problem sizes, 30 timed reps.
#
# Outputs:
#   data/raw/stream_<platform>_<YYYYMMDD>.csv   — one row per timed iteration
#   results/<platform>/stream/<abs>_<sz>.out    — raw binary stdout (provenance)
#
# Usage:
#   ./scripts/run/run_stream.sh --platform nvidia_a100
#   ./scripts/run/run_stream.sh --platform nvidia_a100 --abstraction kokkos --size large
#   ./scripts/run/run_stream.sh --platform amd_mi250x  --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/stream"
RESULTS_BASE="${REPO_ROOT}/results"
DATA_RAW="${REPO_ROOT}/data/raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060_laptop"
ABSTRACTION="all"
SIZE="all"
REPS=30
WARMUP=10
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --dry-run)     DRY_RUN=1;        shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Problem sizes (elements) ──────────────────────────────────────────────────
declare -A SIZES
SIZES[small]=1048576     # 2^20  ~  24 MB  — launch overhead / cache regime
SIZES[medium]=67108864   # 2^26  ~ 1.6 GB  — realistic working set
SIZES[large]=268435456   # 2^28  ~ 6.4 GB  — primary E1 result; forces DRAM saturation

# ── Abstraction → binary name ─────────────────────────────────────────────────
# "native" is always first: it is the PPC efficiency denominator.
# On NVIDIA the native binary is stream-cuda; on AMD it is stream-hip.
declare -A BINARY_NAME
BINARY_NAME[native]="stream-cuda"
BINARY_NAME[kokkos]="stream-kokkos"
BINARY_NAME[raja]="stream-raja"
BINARY_NAME[sycl]="stream-sycl"
BINARY_NAME[julia]="stream-julia"
BINARY_NAME[numba]="stream-numba"

# Platform-specific overrides
case "${PLATFORM}" in
    amd_mi250x|amd_mi300x)
        BINARY_NAME[native]="stream-hip"
        ;;
    intel_pvc|intel_*)
        # SYCL is the native baseline on Intel PVC
        BINARY_NAME[native]="stream-sycl"
        ;;
esac

# Ordered abstraction list — native MUST come first
ALL_ABSTRACTIONS=(native kokkos raja sycl julia numba)

if [[ "${ABSTRACTION}" == "all" ]]; then
    ABSTRACTIONS=("${ALL_ABSTRACTIONS[@]}")
else
    # Validate requested abstraction
    valid=0
    for a in "${ALL_ABSTRACTIONS[@]}"; do
        [[ "$a" == "${ABSTRACTION}" ]] && valid=1
    done
    if [[ ${valid} -eq 0 ]]; then
        echo "Unknown abstraction '${ABSTRACTION}'. Valid: ${ALL_ABSTRACTIONS[*]}" >&2
        exit 1
    fi
    # Always include native so efficiency can be computed; add requested after
    if [[ "${ABSTRACTION}" == "native" ]]; then
        ABSTRACTIONS=(native)
    else
        ABSTRACTIONS=(native "${ABSTRACTION}")
    fi
fi

SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

# ── Output paths ──────────────────────────────────────────────────────────────
TODAY="$(date -u +%Y%m%d)"
CSV_FILE="${DATA_RAW}/stream_${PLATFORM}_${TODAY}.csv"
RAW_OUT_DIR="${RESULTS_BASE}/${PLATFORM}/stream"

if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${DATA_RAW}" "${RAW_OUT_DIR}"
fi

# CSV header — written only if file is new or empty
CSV_HEADER="timestamp,experiment_id,kernel,abstraction,platform,problem_size,problem_size_n,run_id,execution_time_ms,throughput,efficiency,hardware_state_verified,compiler_version,framework_version"
if [[ ${DRY_RUN} -eq 0 ]] && [[ ! -s "${CSV_FILE}" ]]; then
    echo "${CSV_HEADER}" > "${CSV_FILE}"
    echo "[run_stream] Created ${CSV_FILE}"
fi

# ── Binary locator ────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]:-stream-${abs}}"
    local prefix="${bin_name#stream-}"   # strip "stream-" to get build dir prefix

    # Canonical: build/stream/<prefix>_<platform>/<binary>
    local p="${BUILD_BASE}/${prefix}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }

    # Flat layout fallback: build/stream/<binary>
    local p2="${BUILD_BASE}/${bin_name}"
    [[ -x "${p2}" ]] && { echo "${p2}"; return 0; }

    # One-level wildcard: any child directory of BUILD_BASE containing the binary
    for d in "${BUILD_BASE}"/*/; do
        [[ -x "${d}${bin_name}" ]] && { echo "${d}${bin_name}"; return 0; }
    done

    echo ""   # not found
}

# ── CSV row writer ────────────────────────────────────────────────────────────
# Parses STREAM_RUN kernel=triad lines from an .out file and appends CSV rows.
append_csv_rows() {
    local abs="$1" sz="$2" outfile="$3" ts="$4"

    awk \
        -v ts="${ts}" \
        -v abs="${abs}" \
        -v plat="${PLATFORM}" \
        -v sz="${sz}" \
        'BEGIN { OFS="," }
        /^STREAM_RUN/ {
            # parse all key=value tokens on this line
            delete d
            for (i = 2; i <= NF; i++) {
                split($i, kv, "=")
                if (length(kv) == 2) d[kv[1]] = kv[2]
            }
            if (d["kernel"] != "triad") next

            run_id = int(d["run"])
            exp_id = "stream_" abs "_" plat "_" sz "_" sprintf("%03d", run_id)

            print ts, exp_id, "stream", abs, plat, sz, \
                  d["n"], run_id, d["time_ms"], d["bw_gbs"], \
                  "", "1", "", ""
        }' "${outfile}" >> "${CSV_FILE}"
}

# ── Per-run executor ──────────────────────────────────────────────────────────
run_one() {
    local abs="$1" sz="$2"
    local n="${SIZES[$sz]}"
    local bin_name="${BINARY_NAME[$abs]:-stream-${abs}}"
    local outfile="${RAW_OUT_DIR}/${abs}_${sz}.out"

    # In dry-run mode show the expected command path without requiring the binary to exist
    if [[ ${DRY_RUN} -eq 1 ]]; then
        local prefix="${bin_name#stream-}"
        local expected_exe="${BUILD_BASE}/${prefix}_${PLATFORM}/${bin_name}"
        echo "  [DRY] ${expected_exe} --arraysize ${n} --numtimes ${REPS} --warmup ${WARMUP}  >  ${outfile}"
        return 0
    fi

    local exe
    exe="$(find_binary "${abs}")"

    if [[ -z "${exe}" ]]; then
        echo "  SKIP ${abs}/${sz}: binary '${bin_name}' not found under ${BUILD_BASE}"
        echo "        Build with: ./scripts/build/build_stream.sh --platform ${PLATFORM}"
        return 0
    fi

    local cmd=("${exe}" --arraysize "${n}" --numtimes "${REPS}" --warmup "${WARMUP}")

    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "  RUN  ${abs}/${sz}  n=${n}  reps=${REPS}  warmup=${WARMUP}"
    echo "       binary: ${exe}"

    "${cmd[@]}" > "${outfile}" 2>&1
    local rc=$?

    if [[ ${rc} -ne 0 ]]; then
        echo "  ERROR ${abs}/${sz}: exit code ${rc} — see ${outfile}"
        return 0   # non-fatal: continue with remaining abstractions
    fi

    # Extract embedded correctness result
    local correct_line
    correct_line="$(grep -m1 '^STREAM_CORRECT' "${outfile}" || true)"
    if [[ -z "${correct_line}" ]]; then
        echo "  WARN  ${abs}/${sz}: no STREAM_CORRECT line in output"
    elif echo "${correct_line}" | grep -q 'FAIL'; then
        echo "  FAIL  ${abs}/${sz}: correctness check FAILED — skipping CSV write"
        echo "        ${correct_line}"
        return 0
    fi

    # Count timed runs emitted
    local n_runs
    n_runs="$(grep -c '^STREAM_RUN' "${outfile}" || true)"

    # Append rows to CSV
    append_csv_rows "${abs}" "${sz}" "${outfile}" "${ts}"

    # Print median from STREAM_SUMMARY if present
    local summary_line
    summary_line="$(grep '^STREAM_SUMMARY.*kernel=triad' "${outfile}" || true)"
    if [[ -n "${summary_line}" ]]; then
        local median_bw
        median_bw="$(echo "${summary_line}" | grep -oP 'median_bw_gbs=\K[\d.]+')"
        echo "       median ${median_bw} GB/s  (${n_runs} runs → ${outfile})"
    else
        echo "       ${n_runs} STREAM_RUN lines → ${outfile}"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "[run_stream] ================================================================"
echo "[run_stream] Platform:     ${PLATFORM}"
echo "[run_stream] Abstractions: ${ABSTRACTIONS[*]}"
echo "[run_stream] Sizes:        ${SIZE_LIST[*]}"
echo "[run_stream] Reps:         ${REPS}  Warmup: ${WARMUP}"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[run_stream] DRY RUN — no binaries will be executed"
fi
echo "[run_stream] CSV output:   ${CSV_FILE}"
echo "[run_stream] ================================================================"

for abs in "${ABSTRACTIONS[@]}"; do
    echo ""
    echo "── ${abs} ──"
    for sz in "${SIZE_LIST[@]}"; do
        run_one "${abs}" "${sz}"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[run_stream] ================================================================"
if [[ ${DRY_RUN} -eq 0 ]] && [[ -s "${CSV_FILE}" ]]; then
    # Count data rows (subtract 1 for header)
    local_rows=$(( $(wc -l < "${CSV_FILE}") - 1 ))
    echo "[run_stream] Done.  ${local_rows} rows written to ${CSV_FILE}"
    echo ""
    echo "[run_stream] Next steps:"
    echo "  1. Compute PPC:"
    echo "     python analysis/compute_ppc.py \\"
    echo "       --input ${CSV_FILE} \\"
    echo "       --output data/processed/ppc_e1_${PLATFORM}.csv \\"
    echo "       --experiment E1"
    echo ""
    echo "  2. Validate schema:"
    echo "     python scripts/parse/validate_schema.py --input ${CSV_FILE}"
else
    echo "[run_stream] Done."
fi
echo "[run_stream] ================================================================"
