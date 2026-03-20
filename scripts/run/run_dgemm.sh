#!/usr/bin/env bash
# Run E2 DGEMM — all abstractions, all sizes, write per-run CSV rows.
#
# E2 DESIGN DECISIONS
# [D7] experiment_id: dgemm_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
# [D5] native_cublas: dgemm-cuda invoked with --mode cublas; separate CSV rows.
# [D6] raja_naive: dgemm-raja binary, labeled "raja_naive" in CSV.
# Warmup: 50 (§5.5, §9.1 — dynamic memory clock protocol).
# CSV columns: timestamp,experiment_id,kernel,abstraction,platform,
#              problem_size,run_id,execution_time_ms,throughput_gflops,hw_state_verified
#
# Usage:
#   ./scripts/run/run_dgemm.sh --platform nvidia_rtx5060_locked
#   ./scripts/run/run_dgemm.sh --platform nvidia_rtx5060_locked \
#       --abstraction native --size large --warmup 50 --reps 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/dgemm"
DATA_RAW="${REPO_ROOT}/data/raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060_locked"
ABSTRACTION="all"
SIZE="all"
WARMUP=50       # warmup-50 required for dynamic-clock platforms (§9.1)
REPS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${DATA_RAW}"

# ── Problem sizes (D1: large=8192, not 16384) ─────────────────────────────────
declare -A SIZES
SIZES[small]=1024
SIZES[medium]=4096
SIZES[large]=8192

# ── Abstraction registry ──────────────────────────────────────────────────────
# Each entry: "binary_stem:csv_label:extra_args"
# binary_stem → build/dgemm/{stem}_{PLATFORM}/dgemm-{stem_first}
declare -A BINARY_NAME   # abstraction → binary executable name (stem without path)
declare -A BINARY_MODE   # abstraction → --mode arg (empty if not applicable)
declare -A CSV_LABEL     # abstraction → label written to CSV abstraction column

BINARY_NAME[native]="dgemm-cuda"
BINARY_MODE[native]="native"
CSV_LABEL[native]="native"

BINARY_NAME[native_cublas]="dgemm-cuda"
BINARY_MODE[native_cublas]="cublas"
CSV_LABEL[native_cublas]="native_cublas"

BINARY_NAME[native_rocblas]="dgemm-hip"
BINARY_MODE[native_rocblas]="rocblas"
CSV_LABEL[native_rocblas]="native_rocblas"

BINARY_NAME[kokkos]="dgemm-kokkos"
BINARY_MODE[kokkos]=""
CSV_LABEL[kokkos]="kokkos"

BINARY_NAME[raja_naive]="dgemm-raja"
BINARY_MODE[raja_naive]=""
CSV_LABEL[raja_naive]="raja_naive"

BINARY_NAME[sycl]="dgemm-sycl"
BINARY_MODE[sycl]=""
CSV_LABEL[sycl]="sycl"

BINARY_NAME[julia_naive]="dgemm-julia"
BINARY_MODE[julia_naive]="naive"
CSV_LABEL[julia_naive]="julia_naive"

BINARY_NAME[julia_cublas]="dgemm-julia"
BINARY_MODE[julia_cublas]="cublas"
CSV_LABEL[julia_cublas]="julia_cublas"

BINARY_NAME[julia_rocblas]="dgemm-julia"
BINARY_MODE[julia_rocblas]="rocblas"
CSV_LABEL[julia_rocblas]="julia_rocblas"

BINARY_NAME[numba]="dgemm-numba"
BINARY_MODE[numba]=""
CSV_LABEL[numba]="numba"

ALL_ABSTRACTIONS=(native native_cublas kokkos raja_naive sycl julia_naive julia_cublas numba)

# ── AMD platform overrides ────────────────────────────────────────────────────
# For AMD platforms, the native binary is dgemm-hip and the library ceiling
# abstraction is native_rocblas (not native_cublas).
VENDOR="${PLATFORM%%_*}"
if [[ "${VENDOR}" == "amd" ]]; then
    BINARY_NAME[native]="dgemm-hip"
    BINARY_MODE[native]="native"
    ALL_ABSTRACTIONS=(native native_rocblas kokkos raja_naive sycl julia_naive julia_rocblas)
    # Julia must use AMDGPU backend (matches JULIA_GPU_BACKEND convention)
    export JULIA_GPU_BACKEND="amdgpu"
fi

# ── Binary finder ─────────────────────────────────────────────────────────────
# Strips known run-tags (_locked) from platform when looking up build dirs.
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]}"
    # Determine build directory stem from abstraction
    local dir_stem
    case "${abs}" in
        native|native_cublas)           dir_stem="cuda"   ;;
        native_rocblas)                 dir_stem="hip"    ;;
        kokkos)                         dir_stem="kokkos" ;;
        raja_naive)                     dir_stem="raja"   ;;
        sycl)                           dir_stem="sycl"   ;;
        julia_naive|julia_cublas|julia_rocblas) dir_stem="julia" ;;
        numba)                          dir_stem="numba"  ;;
        *) dir_stem="${abs}"                              ;;
    esac
    # AMD: native uses dgemm-hip, found in the hip build dir
    if [[ "${VENDOR}" == "amd" && "${abs}" == "native" ]]; then
        dir_stem="hip"
    fi

    local p="${BUILD_BASE}/${dir_stem}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }

    # Strip _locked and other run-tags from platform to find build dir
    local base_plat="${PLATFORM}"
    base_plat="${base_plat%_locked}"
    if [[ "${base_plat}" != "${PLATFORM}" ]]; then
        local pb="${BUILD_BASE}/${dir_stem}_${base_plat}/${bin_name}"
        [[ -x "${pb}" ]] && { echo "${pb}"; return 0; }
    fi

    # Glob fallback: prefer dirs containing base_plat
    local best=""
    for d in "${BUILD_BASE}"/*/; do
        if [[ -x "${d}${bin_name}" ]]; then
            [[ "${d}" == *"${base_plat}"* ]] && { echo "${d}${bin_name}"; return 0; }
            [[ -z "${best}" ]] && best="${d}${bin_name}"
        fi
    done
    echo "${best}"
}

# ── CSV writer ────────────────────────────────────────────────────────────────
# Parses DGEMM_RUN / DGEMM_HW_STATE lines from binary stdout, writes CSV rows.
write_csv_from_output() {
    local output_file="$1"   # temp file with binary stdout
    local csv_file="$2"
    local abs_label="$3"
    local size_label="$4"
    local N="$5"
    local ts="$6"            # session timestamp

    # Build lookup: run_id → hw_state
    declare -A hw_map
    while IFS= read -r line; do
        if [[ "${line}" =~ ^DGEMM_HW_STATE\ run=([0-9]+)\ hw_state=([01])$ ]]; then
            hw_map["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done < "${output_file}"

    # Parse DGEMM_RUN lines and emit CSV rows
    while IFS= read -r line; do
        if [[ "${line}" =~ ^DGEMM_RUN\ run=([0-9]+)\ n=([0-9]+)\ time_ms=([0-9.]+)\ gflops=([0-9.]+)$ ]]; then
            local run_id="${BASH_REMATCH[1]}"
            local n_actual="${BASH_REMATCH[2]}"
            local time_ms="${BASH_REMATCH[3]}"
            local gflops="${BASH_REMATCH[4]}"
            local hw="${hw_map[$run_id]:-1}"
            local run_id_padded
            run_id_padded="$(printf '%03d' "${run_id}")"
            local exp_id="dgemm_${abs_label}_${PLATFORM}_${size_label}_n${N}_${run_id_padded}"
            printf '%s,%s,dgemm,%s,%s,%s,%s,%s,%s,%s\n' \
                "${ts}" "${exp_id}" "${abs_label}" "${PLATFORM}" \
                "${size_label}" "${run_id}" "${time_ms}" "${gflops}" "${hw}" \
                >> "${csv_file}"
        fi
    done < "${output_file}"
}

# ── Build abstraction/size lists ──────────────────────────────────────────────
if [[ "${ABSTRACTION}" == "all" ]]; then
    run_abstractions=("${ALL_ABSTRACTIONS[@]}")
else
    run_abstractions=("${ABSTRACTION}")
fi

if [[ "${SIZE}" == "all" ]]; then
    run_sizes=(small medium large)
else
    run_sizes=("${SIZE}")
fi

DATE="$(date +%Y%m%d)"

echo "[run_dgemm] =============================================================="
echo "[run_dgemm] Platform: ${PLATFORM}"
echo "[run_dgemm] Warmup:   ${WARMUP}   (warmup-50 protocol, §9.1)"
echo "[run_dgemm] Reps:     ${REPS}"
echo "[run_dgemm] Sizes:    ${run_sizes[*]}"
echo "[run_dgemm] Abstractions: ${run_abstractions[*]}"
echo "[run_dgemm] =============================================================="

# ── Skip classification ───────────────────────────────────────────────────────
# SKIP              = missing environment dependency (binary not built, missing
#                     package, etc.) — fixable by rebuilding or installing.
# UNSUPPORTED_CC120 = hard platform incompatibility — not fixable with current
#                     tooling. Specifically: Numba 0.64.0 predates Blackwell
#                     CC 12.0; libnvvm generates PTX 9.2 which driver 590.48.01
#                     rejects (max PTX 9.1). No pip-installable fix exists.

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs in "${run_abstractions[@]}"; do
    bin="$(find_binary "${abs}")"
    if [[ -z "${bin}" || ! -x "${bin}" ]]; then
        echo "[run_dgemm] SKIP ${abs}: binary not found (run build_dgemm.sh first)"
        continue
    fi

    csv_file="${DATA_RAW}/dgemm_${CSV_LABEL[$abs]}_${PLATFORM}_${DATE}.csv"

    # Write CSV header if new file
    if [[ ! -f "${csv_file}" ]]; then
        echo "timestamp,experiment_id,kernel,abstraction,platform,problem_size,run_id,execution_time_ms,throughput_gflops,hw_state_verified" \
            > "${csv_file}"
    fi

    for sz in "${run_sizes[@]}"; do
        N="${SIZES[$sz]}"
        ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        echo ""
        echo "[run_dgemm] ── ${abs} / ${sz} (N=${N}) ─────────────────────────"
        echo "[run_dgemm]    binary: ${bin}"
        echo "[run_dgemm]    csv:    ${csv_file}"

        # Build argument list
        local_args=(--n "${N}" --warmup "${WARMUP}" --reps "${REPS}" --platform "${PLATFORM}")
        mode="${BINARY_MODE[$abs]}"
        [[ -n "${mode}" ]] && local_args+=(--mode "${mode}")

        tmp_out="$(mktemp)"
        # Run binary; tee to temp file and terminal
        "${bin}" "${local_args[@]}" 2>&1 | tee "${tmp_out}"

        # Detect hard platform incompatibility from kernel output and relabel.
        # PTX_VERSION_MISMATCH means Numba's libnvvm generated a PTX version
        # the driver cannot load — this is UNSUPPORTED_CC120, not a SKIP.
        if grep -q "PTX_VERSION_MISMATCH" "${tmp_out}" 2>/dev/null; then
            echo "[run_dgemm] UNSUPPORTED_CC120 ${abs}: Numba 0.64.0 does not support" \
                 "CC 12.0 (Blackwell) — PTX 9.2 rejected by driver (max PTX 9.1)." \
                 "No pip-installable fix exists. Mark as platform limitation, not SKIP."
        fi

        # Parse output → append CSV rows
        write_csv_from_output "${tmp_out}" "${csv_file}" \
            "${CSV_LABEL[$abs]}" "${sz}" "${N}" "${ts}"

        rm -f "${tmp_out}"
        echo "[run_dgemm]    → rows appended to ${csv_file}"
    done
done

echo ""
echo "[run_dgemm] =============================================================="
echo "[run_dgemm] Done. CSV files:"
ls -lh "${DATA_RAW}"/dgemm_*_"${PLATFORM}"_"${DATE}".csv 2>/dev/null \
    | awk '{print "  "$NF" ("$5")"}' || true
echo "[run_dgemm] =============================================================="
echo "[run_dgemm] Next step: python scripts/process_e2.py"
