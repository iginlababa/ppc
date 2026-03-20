#!/usr/bin/env bash
# Run E3 3D Stencil — all abstractions, all sizes, write per-run CSV rows.
#
# E3 DESIGN DECISIONS
# [D6] experiment_id: stencil_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
# [D7] Warmup: adaptive (CV < 2% over last 10 timings), max 200 iterations.
#      Kernels handle warmup internally; --warmup passes the max cap.
# CSV columns: timestamp,experiment_id,kernel,abstraction,platform,
#              problem_size,run_id,execution_time_ms,throughput_gbs,hw_state_verified
#
# Usage:
#   ./scripts/run/run_stencil.sh --platform nvidia_rtx5060
#   ./scripts/run/run_stencil.sh --platform nvidia_rtx5060 \
#       --abstraction native --size large --warmup 200 --reps 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/stencil"
DATA_RAW="${REPO_ROOT}/data/raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060"
ABSTRACTION="all"
SIZE="all"
WARMUP=200      # max adaptive warmup iterations (kernel stops early at CV<2%)
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

# ── Problem sizes (D1: sides 32/128/256) ─────────────────────────────────────
declare -A SIZES
SIZES[small]=32
SIZES[medium]=128
SIZES[large]=256

# ── Abstraction registry ──────────────────────────────────────────────────────
declare -A BINARY_NAME   # abstraction → executable stem
declare -A CSV_LABEL     # abstraction → label written to CSV

BINARY_NAME[native]="stencil-cuda"
CSV_LABEL[native]="native"

BINARY_NAME[kokkos]="stencil-kokkos"
CSV_LABEL[kokkos]="kokkos"

BINARY_NAME[raja]="stencil-raja"
CSV_LABEL[raja]="raja"

BINARY_NAME[sycl]="stencil-sycl"
CSV_LABEL[sycl]="sycl"

BINARY_NAME[julia]="stencil-julia"
CSV_LABEL[julia]="julia"

BINARY_NAME[numba]="stencil-numba"
CSV_LABEL[numba]="numba"

ALL_ABSTRACTIONS=(native kokkos raja sycl julia numba)

# ── AMD platform overrides ────────────────────────────────────────────────────
# On AMD, native → stencil-hip; numba is unsupported; set Julia backend.
_VENDOR="${PLATFORM%%_*}"
if [[ "${_VENDOR}" == "amd" ]]; then
    BINARY_NAME[native]="stencil-hip"
    ALL_ABSTRACTIONS=(native kokkos raja sycl julia)
    export JULIA_GPU_BACKEND="amdgpu"
fi

# ── Binary finder ─────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]}"
    local dir_stem
    # On AMD, native binary lives in the hip_* build directory
    if [[ "${_VENDOR}" == "amd" && "${abs}" == "native" ]]; then
        dir_stem="hip"
    else
        case "${abs}" in
            native)  dir_stem="cuda"   ;;
            kokkos)  dir_stem="kokkos" ;;
            raja)    dir_stem="raja"   ;;
            sycl)    dir_stem="sycl"   ;;
            julia)   dir_stem="julia"  ;;
            numba)   dir_stem="numba"  ;;
            *)       dir_stem="${abs}" ;;
        esac
    fi

    local p="${BUILD_BASE}/${dir_stem}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }

    local base_plat="${PLATFORM%_locked}"
    if [[ "${base_plat}" != "${PLATFORM}" ]]; then
        local pb="${BUILD_BASE}/${dir_stem}_${base_plat}/${bin_name}"
        [[ -x "${pb}" ]] && { echo "${pb}"; return 0; }
    fi

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
# Parses STENCIL_RUN / STENCIL_HW_STATE lines from binary stdout → CSV rows.
write_csv_from_output() {
    local output_file="$1"
    local csv_file="$2"
    local abs_label="$3"
    local size_label="$4"
    local N="$5"
    local ts="$6"

    declare -A hw_map
    while IFS= read -r line; do
        if [[ "${line}" =~ ^STENCIL_HW_STATE\ run=([0-9]+)\ hw_state=([01])$ ]]; then
            hw_map["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done < "${output_file}"

    while IFS= read -r line; do
        if [[ "${line}" =~ ^STENCIL_RUN\ run=([0-9]+)\ n=([0-9]+)\ time_ms=([0-9.]+)\ throughput_gbs=([0-9.]+)$ ]]; then
            local run_id="${BASH_REMATCH[1]}"
            local time_ms="${BASH_REMATCH[3]}"
            local gbs="${BASH_REMATCH[4]}"
            local hw="${hw_map[$run_id]:-1}"
            local run_id_padded
            run_id_padded="$(printf '%03d' "${run_id}")"
            local exp_id="stencil_${abs_label}_${PLATFORM}_${size_label}_n${N}_${run_id_padded}"
            printf '%s,%s,stencil,%s,%s,%s,%s,%s,%s,%s\n' \
                "${ts}" "${exp_id}" "${abs_label}" "${PLATFORM}" \
                "${size_label}" "${run_id}" "${time_ms}" "${gbs}" "${hw}" \
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

echo "[run_stencil] =============================================================="
echo "[run_stencil] Platform:      ${PLATFORM}"
echo "[run_stencil] Warmup (max):  ${WARMUP}  (adaptive CV<2% protocol, §9.1 amended)"
echo "[run_stencil] Reps:          ${REPS}"
echo "[run_stencil] Sizes:         ${run_sizes[*]}"
echo "[run_stencil] Abstractions:  ${run_abstractions[*]}"
echo "[run_stencil] =============================================================="

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs in "${run_abstractions[@]}"; do
    bin="$(find_binary "${abs}")"
    if [[ -z "${bin}" || ! -x "${bin}" ]]; then
        echo "[run_stencil] SKIP ${abs}: binary not found (run build_stencil.sh first)"
        continue
    fi

    csv_file="${DATA_RAW}/stencil_${CSV_LABEL[$abs]}_${PLATFORM}_${DATE}.csv"

    if [[ ! -f "${csv_file}" ]]; then
        echo "timestamp,experiment_id,kernel,abstraction,platform,problem_size,run_id,execution_time_ms,throughput_gbs,hw_state_verified" \
            > "${csv_file}"
    fi

    for sz in "${run_sizes[@]}"; do
        N="${SIZES[$sz]}"
        ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        echo ""
        echo "[run_stencil] ── ${abs} / ${sz} (N=${N}) ────────────────────────"
        echo "[run_stencil]    binary: ${bin}"
        echo "[run_stencil]    csv:    ${csv_file}"

        local_args=(--n "${N}" --warmup "${WARMUP}" --reps "${REPS}" --platform "${PLATFORM}")

        tmp_out="$(mktemp)"
        "${bin}" "${local_args[@]}" 2>&1 | tee "${tmp_out}"

        # Detect Numba PTX incompatibility
        if grep -q "PTX_VERSION_MISMATCH" "${tmp_out}" 2>/dev/null; then
            echo "[run_stencil] UNSUPPORTED_CC120 ${abs}: Numba 0.64.0 does not support" \
                 "CC 12.0 (Blackwell) — PTX 9.2 rejected by driver (max PTX 9.1)." \
                 "Platform limitation, not a SKIP."
        fi

        write_csv_from_output "${tmp_out}" "${csv_file}" \
            "${CSV_LABEL[$abs]}" "${sz}" "${N}" "${ts}"

        rm -f "${tmp_out}"
        echo "[run_stencil]    → rows appended to ${csv_file}"
    done
done

echo ""
echo "[run_stencil] =============================================================="
echo "[run_stencil] Done. CSV files:"
ls -lh "${DATA_RAW}"/stencil_*_"${PLATFORM}"_"${DATE}".csv 2>/dev/null \
    | awk '{print "  "$NF" ("$5")"}' || true
echo "[run_stencil] =============================================================="
echo "[run_stencil] Next step: python scripts/process_e3.py"
