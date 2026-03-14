#!/usr/bin/env bash
# Run E1 STREAM Triad — all abstractions, all problem sizes, 30 timed reps.
#
# Thermal stability measures:
#   - Pre-warmup:  a 10-iteration discard run brings the GPU to steady-state
#                  temperature before the measured run begins.
#   - Clock lock:  nvidia-smi --lock-gpu-clocks locks the GPU to its base
#                  application clock.  Requires nvidia-persistenced or sudo;
#                  gracefully skipped on laptop GPUs that do not support it.
#   - Cooldown:    30-second sleep between abstraction runs (configurable).
#   - Outlier flag: any timed run whose throughput deviates >15% from the
#                  run median is written with hardware_state_verified=0.
#
# Outputs:
#   data/raw/stream_<platform>_<YYYYMMDD>.csv   — one row per timed iteration
#   results/<platform>/stream/<abs>_<sz>.out    — raw binary stdout (provenance)
#
# Usage:
#   ./scripts/run/run_stream.sh --platform nvidia_rtx5060_laptop
#   ./scripts/run/run_stream.sh --platform nvidia_rtx5060_laptop \
#       --abstraction native --size large
#   ./scripts/run/run_stream.sh --platform nvidia_rtx5060_laptop \
#       --cooldown 0 --dry-run

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
PRE_WARMUP_ITERS=10   # separate discard run before each measurement
COOLDOWN=30           # seconds between abstraction runs (0 to skip)
OUTLIER_THRESHOLD=15  # percent deviation from median → thermal outlier
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --cooldown)    COOLDOWN="$2";    shift 2 ;;
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
declare -A BINARY_NAME
BINARY_NAME[native]="stream-cuda"
BINARY_NAME[kokkos]="stream-kokkos"
BINARY_NAME[raja]="stream-raja"
BINARY_NAME[sycl]="stream-sycl"
BINARY_NAME[julia]="stream-julia"
BINARY_NAME[numba]="stream-numba"

case "${PLATFORM}" in
    amd_mi250x|amd_mi300x) BINARY_NAME[native]="stream-hip"  ;;
    intel_pvc|intel_*)     BINARY_NAME[native]="stream-sycl" ;;
esac

ALL_ABSTRACTIONS=(native kokkos raja sycl julia numba)

if [[ "${ABSTRACTION}" == "all" ]]; then
    ABSTRACTIONS=("${ALL_ABSTRACTIONS[@]}")
else
    valid=0
    for a in "${ALL_ABSTRACTIONS[@]}"; do [[ "$a" == "${ABSTRACTION}" ]] && valid=1; done
    if [[ ${valid} -eq 0 ]]; then
        echo "Unknown abstraction '${ABSTRACTION}'. Valid: ${ALL_ABSTRACTIONS[*]}" >&2
        exit 1
    fi
    ABSTRACTIONS=(native)
    [[ "${ABSTRACTION}" != "native" ]] && ABSTRACTIONS+=(${ABSTRACTION})
fi

SIZE_LIST=(small medium large)
[[ "${SIZE}" != "all" ]] && SIZE_LIST=("${SIZE}")

# ── Output paths ──────────────────────────────────────────────────────────────
TODAY="$(date -u +%Y%m%d)"
# CSV_FILE is set per-abstraction inside the main loop:
#   stream_<abstraction>_<platform>_<YYYYMMDD>.csv
RAW_OUT_DIR="${RESULTS_BASE}/${PLATFORM}/stream"
CSV_FILE=""   # set per abstraction in main loop

if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${DATA_RAW}" "${RAW_OUT_DIR}"
fi

# hardware_state_verified: 1 = clean run, 0 = thermal outlier (>OUTLIER_THRESHOLD% from median)
CSV_HEADER="timestamp,experiment_id,kernel,abstraction,platform,problem_size,problem_size_n,run_id,execution_time_ms,throughput,efficiency,hardware_state_verified,compiler_version,framework_version"

# ── Binary locator ────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]:-stream-${abs}}"
    local prefix="${bin_name#stream-}"
    # Try exact platform match first (e.g. raja_nvidia_rtx5060_laptop)
    local p="${BUILD_BASE}/${prefix}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }
    # Try base platform with known run-tags stripped (e.g. _locked, _v2)
    local base_plat="${PLATFORM}"
    base_plat="${base_plat%_locked}"
    if [[ "${base_plat}" != "${PLATFORM}" ]]; then
        local pb="${BUILD_BASE}/${prefix}_${base_plat}/${bin_name}"
        [[ -x "${pb}" ]] && { echo "${pb}"; return 0; }
    fi
    local p2="${BUILD_BASE}/${bin_name}"
    [[ -x "${p2}" ]] && { echo "${p2}"; return 0; }
    # Last resort: find any matching binary (sorted to prefer platform-specific)
    local best=""
    for d in "${BUILD_BASE}"/*/; do
        if [[ -x "${d}${bin_name}" ]]; then
            # Prefer dirs containing the base platform name
            if [[ "${d}" == *"${base_plat}"* ]]; then
                echo "${d}${bin_name}"; return 0
            fi
            [[ -z "${best}" ]] && best="${d}${bin_name}"
        fi
    done
    echo "${best}"
}

# ── GPU clock management ──────────────────────────────────────────────────────
# Queries the base application clock and attempts to lock the GPU to it.
# On laptop GPUs nvidia-smi may report "Not Supported" — this is handled
# gracefully; measurements still proceed with a warning.

GPU_BASE_CLOCK=""   # populated by lock_gpu_clocks(), used for reporting

lock_gpu_clocks() {
    # nvidia-smi --query-gpu=clocks.applications.gr returns e.g. "1890 MHz"
    local raw
    raw="$(nvidia-smi --query-gpu=clocks.applications.gr \
           --format=csv,noheader 2>/dev/null | head -1)"
    GPU_BASE_CLOCK="$(echo "${raw}" | tr -dc '0-9')"

    if [[ -z "${GPU_BASE_CLOCK}" || "${GPU_BASE_CLOCK}" == "0" ]]; then
        echo "  WARN  clock-lock: could not query application clock — skipping"
        GPU_BASE_CLOCK=""
        return
    fi

    # Try without sudo first (works when nvidia-persistenced is running),
    # then fall back to sudo (works on bare-metal with root).
    local locked=0
    if nvidia-smi --lock-gpu-clocks="${GPU_BASE_CLOCK},${GPU_BASE_CLOCK}" \
        > /dev/null 2>&1; then
        locked=1
    elif sudo -n nvidia-smi --lock-gpu-clocks="${GPU_BASE_CLOCK},${GPU_BASE_CLOCK}" \
        > /dev/null 2>&1; then
        locked=1
    fi

    if [[ ${locked} -eq 1 ]]; then
        echo "  GPU   clocks locked at ${GPU_BASE_CLOCK} MHz"
    else
        echo "  WARN  clock-lock: --lock-gpu-clocks not supported on this GPU/driver"
        echo "        (common on laptop GPUs — results may show thermal variance)"
        GPU_BASE_CLOCK=""
    fi
}

reset_gpu_clocks() {
    [[ -z "${GPU_BASE_CLOCK}" ]] && return
    nvidia-smi --reset-gpu-clocks > /dev/null 2>&1 \
        || sudo -n nvidia-smi --reset-gpu-clocks > /dev/null 2>&1 \
        || true
}

# ── Thermal outlier detection ─────────────────────────────────────────────────
# Computes the median of all kernel=triad bw_gbs values in an .out file.
compute_median_bw() {
    local outfile="$1"
    grep '^STREAM_RUN' "${outfile}" \
    | grep 'kernel=triad' \
    | grep -oP 'bw_gbs=\K[0-9.]+' \
    | sort -n \
    | awk 'BEGIN{n=0} {v[n++]=$1}
           END{
               if (n==0) { print 0; exit }
               mid = int(n/2)
               if (n % 2 == 0) printf "%.4f\n", (v[mid-1] + v[mid]) / 2.0
               else             printf "%.4f\n", v[mid]
           }'
}

# ── CSV row writer ────────────────────────────────────────────────────────────
# Two-pass: median is pre-computed, then each row is flagged individually.
# hardware_state_verified = 1  →  clean run (within OUTLIER_THRESHOLD of median)
# hardware_state_verified = 0  →  thermal outlier (flagged, not excluded)
append_csv_rows() {
    local abs="$1" sz="$2" outfile="$3" ts="$4" median_bw="$5"
    local threshold_frac
    threshold_frac="$(echo "${OUTLIER_THRESHOLD}" | awk '{printf "%.6f", $1/100}')"

    awk \
        -v ts="${ts}" \
        -v abs="${abs}" \
        -v plat="${PLATFORM}" \
        -v sz="${sz}" \
        -v median="${median_bw}" \
        -v thr="${threshold_frac}" \
        'BEGIN { OFS="," }
        /^STREAM_RUN/ {
            delete d
            for (i = 2; i <= NF; i++) {
                split($i, kv, "=")
                if (length(kv) == 2) d[kv[1]] = kv[2]
            }
            if (d["kernel"] != "triad") next

            run_id = int(d["run"])
            bw     = d["bw_gbs"] + 0
            exp_id = "stream_" abs "_" plat "_" sz "_" sprintf("%03d", run_id)

            # Flag thermal outliers: |bw - median| / median > threshold
            hw_ok = 1
            if (median > 0) {
                dev = (bw - median)
                if (dev < 0) dev = -dev
                if (dev / median > thr) hw_ok = 0
            }

            print ts, exp_id, "stream", abs, plat, sz, \
                  d["n"], run_id, d["time_ms"], d["bw_gbs"], \
                  "", hw_ok, "", ""
        }' "${outfile}" >> "${CSV_FILE}"
}

# ── Per-run executor ──────────────────────────────────────────────────────────
run_one() {
    local abs="$1" sz="$2"
    local n="${SIZES[$sz]}"
    local bin_name="${BINARY_NAME[$abs]:-stream-${abs}}"
    local outfile="${RAW_OUT_DIR}/${abs}_${sz}.out"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        local prefix="${bin_name#stream-}"
        local expected_exe="${BUILD_BASE}/${prefix}_${PLATFORM}/${bin_name}"
        echo "  [DRY] pre-warmup: ${expected_exe} --arraysize ${n} --numtimes ${PRE_WARMUP_ITERS} --warmup 0"
        echo "  [DRY] clock-lock: nvidia-smi --lock-gpu-clocks=<base>,<base>"
        echo "  [DRY] measure:    ${expected_exe} --arraysize ${n} --numtimes ${REPS} --warmup ${WARMUP}  >  ${outfile}"
        echo "  [DRY] outlier:    flag runs deviating >${OUTLIER_THRESHOLD}% from median"
        return 0
    fi

    local exe
    exe="$(find_binary "${abs}")"
    if [[ -z "${exe}" ]]; then
        echo "  SKIP  ${abs}/${sz}: binary '${bin_name}' not found under ${BUILD_BASE}"
        echo "        Build with: ./scripts/build/build_stream.sh --platform ${PLATFORM}"
        return 0
    fi

    # ── 1. Lock GPU clocks ────────────────────────────────────────────────────
    lock_gpu_clocks

    # ── 2. Pre-warmup run (results discarded) ─────────────────────────────────
    # Runs PRE_WARMUP_ITERS timed iterations with no internal warmup.
    # This brings the GPU to thermal steady-state before the measured run.
    echo "  WARM  ${abs}/${sz}  n=${n}  pre-warmup=${PRE_WARMUP_ITERS} iters (discarded)"
    "${exe}" --arraysize "${n}" \
             --numtimes "${PRE_WARMUP_ITERS}" \
             --warmup 0 \
             > /dev/null 2>&1 || true   # non-fatal; proceed to measured run

    # ── 3. Measured run ───────────────────────────────────────────────────────
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  RUN   ${abs}/${sz}  n=${n}  reps=${REPS}  warmup=${WARMUP}"
    echo "        binary: ${exe}"

    "${exe}" --arraysize "${n}" \
             --numtimes  "${REPS}" \
             --warmup    "${WARMUP}" \
             > "${outfile}" 2>&1
    local rc=$?

    # ── 4. Reset clocks ───────────────────────────────────────────────────────
    reset_gpu_clocks

    if [[ ${rc} -ne 0 ]]; then
        echo "  ERROR ${abs}/${sz}: exit code ${rc} — see ${outfile}"
        return 0
    fi

    # ── 5. Correctness gate ───────────────────────────────────────────────────
    local correct_line
    correct_line="$(grep -m1 '^STREAM_CORRECT' "${outfile}" || true)"
    if [[ -z "${correct_line}" ]]; then
        echo "  WARN  ${abs}/${sz}: no STREAM_CORRECT line in output"
    elif echo "${correct_line}" | grep -q 'FAIL'; then
        echo "  FAIL  ${abs}/${sz}: correctness check FAILED — skipping CSV write"
        echo "        ${correct_line}"
        return 0
    fi

    # ── 6. Compute median and flag outliers ───────────────────────────────────
    local median_bw
    median_bw="$(compute_median_bw "${outfile}")"

    local n_runs outlier_count clean_count
    n_runs="$(grep -c '^STREAM_RUN' "${outfile}" || true)"

    # Count outliers: runs where |bw - median| / median > threshold
    local threshold_frac
    threshold_frac="$(echo "${OUTLIER_THRESHOLD}" | awk '{printf "%.6f", $1/100}')"
    outlier_count="$(grep '^STREAM_RUN' "${outfile}" \
        | grep 'kernel=triad' \
        | grep -oP 'bw_gbs=\K[0-9.]+' \
        | awk -v m="${median_bw}" -v thr="${threshold_frac}" \
              'BEGIN{c=0} {dev=($1-m); if(dev<0)dev=-dev; if(m>0 && dev/m>thr)c++} END{print c}')"
    clean_count=$(( n_runs - outlier_count ))

    # ── 7. Write CSV rows ─────────────────────────────────────────────────────
    append_csv_rows "${abs}" "${sz}" "${outfile}" "${ts}" "${median_bw}"

    # ── 8. Report ─────────────────────────────────────────────────────────────
    local clock_info=""
    [[ -n "${GPU_BASE_CLOCK}" ]] && clock_info="  clock=${GPU_BASE_CLOCK}MHz"
    echo "        median ${median_bw} GB/s  |  ${clean_count}/${n_runs} clean  |  ${outlier_count} outlier(s) flagged${clock_info}"
    echo "        → ${outfile}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "[run_stream] ================================================================"
echo "[run_stream] Platform:          ${PLATFORM}"
echo "[run_stream] Abstractions:      ${ABSTRACTIONS[*]}"
echo "[run_stream] Sizes:             ${SIZE_LIST[*]}"
echo "[run_stream] Reps / Warmup:     ${REPS} / ${WARMUP}"
echo "[run_stream] Pre-warmup iters:  ${PRE_WARMUP_ITERS} (discarded)"
echo "[run_stream] Cooldown:          ${COOLDOWN}s between abstractions"
echo "[run_stream] Outlier threshold: ${OUTLIER_THRESHOLD}% from median"
[[ ${DRY_RUN} -eq 1 ]] && echo "[run_stream] DRY RUN — no binaries will be executed"
echo "[run_stream] CSV pattern:       ${DATA_RAW}/stream_<abstraction>_${PLATFORM}_${TODAY}.csv"
echo "[run_stream] ================================================================"

CREATED_CSVS=()

first_abs=1
for abs in "${ABSTRACTIONS[@]}"; do
    # ── Per-abstraction CSV file ───────────────────────────────────────────────
    CSV_FILE="${DATA_RAW}/stream_${abs}_${PLATFORM}_${TODAY}.csv"
    if [[ ${DRY_RUN} -eq 0 ]] && [[ ! -s "${CSV_FILE}" ]]; then
        echo "${CSV_HEADER}" > "${CSV_FILE}"
        echo "[run_stream] Created ${CSV_FILE}"
    fi

    # Cooldown between abstractions (skip before the first one)
    if [[ ${first_abs} -eq 0 && ${COOLDOWN} -gt 0 && ${DRY_RUN} -eq 0 ]]; then
        echo ""
        echo "  [cooldown] sleeping ${COOLDOWN}s to let GPU temperature stabilise..."
        sleep "${COOLDOWN}"
    fi
    first_abs=0

    echo ""
    echo "── ${abs} ──"
    for sz in "${SIZE_LIST[@]}"; do
        run_one "${abs}" "${sz}"
    done

    [[ -s "${CSV_FILE}" ]] && CREATED_CSVS+=("${CSV_FILE}")
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[run_stream] ================================================================"
if [[ ${DRY_RUN} -eq 0 ]] && [[ ${#CREATED_CSVS[@]} -gt 0 ]]; then
    total_rows=0
    total_outliers=0
    for f in "${CREATED_CSVS[@]}"; do
        rows=$(( $(wc -l < "${f}") - 1 ))
        outliers="$(awk -F',' 'NR>1 && $12=="0"' "${f}" | wc -l)"
        total_rows=$(( total_rows + rows ))
        total_outliers=$(( total_outliers + outliers ))
    done
    echo "[run_stream] Done."
    echo "[run_stream] CSV files written: ${#CREATED_CSVS[@]}"
    for f in "${CREATED_CSVS[@]}"; do
        rows=$(( $(wc -l < "${f}") - 1 ))
        echo "[run_stream]   ${f##*/}  (${rows} rows)"
    done
    echo "[run_stream] Total rows:    ${total_rows}"
    echo "[run_stream] Outliers (hw_state=0): ${total_outliers}"
    echo ""
    echo "[run_stream] Next steps:"
    echo "  1. Compute PPC (provide each CSV or a merged view):"
    echo "     python analysis/compute_ppc.py \\"
    echo "       --input  ${DATA_RAW}/stream_native_${PLATFORM}_${TODAY}.csv \\"
    echo "       --output data/processed/ppc_e1_${PLATFORM}.csv \\"
    echo "       --experiment E1 --problem-size large"
    echo ""
    echo "  2. Validate schema (run per file):"
    echo "     for f in ${DATA_RAW}/stream_*_${PLATFORM}_${TODAY}.csv; do"
    echo "       python scripts/parse/validate_schema.py --input \"\$f\""
    echo "     done"
else
    echo "[run_stream] Done."
fi
echo "[run_stream] ================================================================"
