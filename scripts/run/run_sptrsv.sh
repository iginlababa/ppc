#!/usr/bin/env bash
# Run E5 SpTRSV — all abstractions, all matrix types, all sizes, write per-run CSV rows.
#
# E5 DESIGN DECISIONS
# [D6] experiment_id: sptrsv_{abstraction}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}
# [D7] Warmup: adaptive (CV < 2% over last 10 timings), max 200 iterations.
#      Kernels handle warmup internally; x is reset inside the warmup loop.
#      Timed region excludes x reset (cudaMemset/deep_copy/CUDA.fill! before timing).
# CSV columns: timestamp,experiment_id,kernel,abstraction,platform,
#              matrix_type,problem_size,n_rows,nnz,
#              n_levels,max_level_width,min_level_width,
#              run_id,execution_time_ms,throughput_gflops,hw_state_verified
#
# Level-set metadata (n_levels, max_level_width, min_level_width) is constant
# per (matrix_type, n_rows) and is emitted by the binary on every SPTRSV_RUN line.
# These columns are the key diagnostic for interpreting efficiency vs. parallelism.
#
# Usage:
#   ./scripts/run/run_sptrsv.sh --platform nvidia_rtx5060_laptop
#   ./scripts/run/run_sptrsv.sh --platform nvidia_rtx5060_laptop \
#       --abstraction native --matrix lower_triangular_laplacian --size large \
#       --warmup 200 --reps 30 --verify

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/sptrsv"
DATA_RAW="${REPO_ROOT}/data/raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060_laptop"
ABSTRACTION="all"
MATRIX="all"
SIZE="all"
WARMUP=200
REPS=30
VERIFY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --matrix)      MATRIX="$2";      shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --verify)      VERIFY=true;      shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${DATA_RAW}"

# ── Problem sizes (D1: small=256, medium=2048, large=8192 rows) ───────────────
declare -A SIZES
SIZES[small]=256
SIZES[medium]=2048
SIZES[large]=8192

# ── Matrix types (D2) ─────────────────────────────────────────────────────────
ALL_MATRICES=(lower_triangular_laplacian lower_triangular_random)

# ── Abstraction registry ──────────────────────────────────────────────────────
declare -A BINARY_NAME
declare -A CSV_LABEL

BINARY_NAME[native]="sptrsv-cuda"
CSV_LABEL[native]="native"

BINARY_NAME[kokkos]="sptrsv-kokkos"
CSV_LABEL[kokkos]="kokkos"

BINARY_NAME[raja]="sptrsv-raja"
CSV_LABEL[raja]="raja"

BINARY_NAME[julia]="sptrsv-julia"
CSV_LABEL[julia]="julia"

ALL_ABSTRACTIONS=(native kokkos raja julia)
# numba: UNSUPPORTED_CC120; sycl: NO_COMPILER — not in run list.

# ── Binary finder ─────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]}"
    local dir_stem
    case "${abs}" in
        native)  dir_stem="cuda"   ;;
        kokkos)  dir_stem="kokkos" ;;
        raja)    dir_stem="raja"   ;;
        julia)   dir_stem="julia"  ;;
        *)       dir_stem="${abs}" ;;
    esac

    local p="${BUILD_BASE}/${dir_stem}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }

    local best=""
    for d in "${BUILD_BASE}"/*/; do
        if [[ -x "${d}${bin_name}" ]]; then
            [[ "${d}" == *"${PLATFORM}"* ]] && { echo "${d}${bin_name}"; return 0; }
            [[ -z "${best}" ]] && best="${d}${bin_name}"
        fi
    done
    echo "${best}"
}

# ── CSV writer ────────────────────────────────────────────────────────────────
# Parses SPTRSV_RUN / SPTRSV_HW_STATE lines from binary stdout → CSV rows.
write_csv_from_output() {
    local output_file="$1"
    local csv_file="$2"
    local abs_label="$3"
    local matrix_type="$4"
    local size_label="$5"
    local ts="$6"

    # Build hw_state map from SPTRSV_HW_STATE lines
    declare -A hw_map
    while IFS= read -r line; do
        if [[ "${line}" =~ ^SPTRSV_HW_STATE\ run=([0-9]+)\ hw_state=([01])$ ]]; then
            hw_map["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done < "${output_file}"

    # Parse SPTRSV_RUN lines and write CSV rows
    # Format: SPTRSV_RUN run=N n_rows=N nnz=N n_levels=N max_lw=N min_lw=N matrix=X time_ms=X throughput_gflops=X
    while IFS= read -r line; do
        if [[ "${line}" =~ ^SPTRSV_RUN\ run=([0-9]+)\ n_rows=([0-9]+)\ nnz=([0-9]+)\ n_levels=([0-9]+)\ max_lw=([0-9]+)\ min_lw=([0-9]+)\ matrix=([a-z_0-9]+)\ time_ms=([0-9.]+)\ throughput_gflops=([0-9.]+)$ ]]; then
            local run_id="${BASH_REMATCH[1]}"
            local n_rows="${BASH_REMATCH[2]}"
            local nnz="${BASH_REMATCH[3]}"
            local n_levels="${BASH_REMATCH[4]}"
            local max_lw="${BASH_REMATCH[5]}"
            local min_lw="${BASH_REMATCH[6]}"
            local time_ms="${BASH_REMATCH[8]}"
            local gflops="${BASH_REMATCH[9]}"
            local hw="${hw_map[$run_id]:-1}"
            local run_id_padded
            run_id_padded="$(printf '%03d' "${run_id}")"
            local exp_id="sptrsv_${abs_label}_${PLATFORM}_${matrix_type}_${size_label}_n${n_rows}_${run_id_padded}"
            printf '%s,%s,sptrsv,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${ts}" "${exp_id}" "${abs_label}" "${PLATFORM}" \
                "${matrix_type}" "${size_label}" "${n_rows}" "${nnz}" \
                "${n_levels}" "${max_lw}" "${min_lw}" \
                "${run_id}" "${time_ms}" "${gflops}" "${hw}" \
                >> "${csv_file}"
        fi
    done < "${output_file}"
}

# ── Build abstraction/matrix/size lists ───────────────────────────────────────
if [[ "${ABSTRACTION}" == "all" ]]; then
    run_abstractions=("${ALL_ABSTRACTIONS[@]}")
else
    run_abstractions=("${ABSTRACTION}")
fi

if [[ "${MATRIX}" == "all" ]]; then
    run_matrices=("${ALL_MATRICES[@]}")
else
    run_matrices=("${MATRIX}")
fi

if [[ "${SIZE}" == "all" ]]; then
    run_sizes=(small medium large)
else
    run_sizes=("${SIZE}")
fi

DATE="$(date +%Y%m%d)"

echo "[run_sptrsv] ================================================================"
echo "[run_sptrsv] Platform:      ${PLATFORM}"
echo "[run_sptrsv] Warmup (max):  ${WARMUP}  (adaptive CV<2% protocol)"
echo "[run_sptrsv] Reps:          ${REPS}"
echo "[run_sptrsv] Sizes:         ${run_sizes[*]}"
echo "[run_sptrsv] Abstractions:  ${run_abstractions[*]}"
echo "[run_sptrsv] Matrix types:  ${run_matrices[*]}"
echo "[run_sptrsv] Note:          numba=UNSUPPORTED_CC120, sycl=NO_COMPILER"
echo "[run_sptrsv] ================================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs in "${run_abstractions[@]}"; do
    bin="$(find_binary "${abs}")"
    if [[ -z "${bin}" || ! -x "${bin}" ]]; then
        echo "[run_sptrsv] SKIP ${abs}: binary not found (run build_sptrsv.sh first)"
        continue
    fi

    csv_file="${DATA_RAW}/sptrsv_${CSV_LABEL[$abs]}_${PLATFORM}_${DATE}.csv"

    if [[ ! -f "${csv_file}" ]]; then
        echo "timestamp,experiment_id,kernel,abstraction,platform,matrix_type,problem_size,n_rows,nnz,n_levels,max_level_width,min_level_width,run_id,execution_time_ms,throughput_gflops,hw_state_verified" \
            > "${csv_file}"
    fi

    for mtype in "${run_matrices[@]}"; do
        for sz in "${run_sizes[@]}"; do
            N="${SIZES[$sz]}"
            ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

            echo ""
            echo "[run_sptrsv] ── ${abs} / ${mtype} / ${sz} (N=${N}) ──────────────"
            echo "[run_sptrsv]    binary: ${bin}"
            echo "[run_sptrsv]    csv:    ${csv_file}"

            local_args=(
                --n "${N}"
                --matrix "${mtype}"
                --warmup "${WARMUP}"
                --reps "${REPS}"
                --platform "${PLATFORM}"
            )
            [[ "${VERIFY}" == "true" ]] && local_args+=(--verify)

            tmp_out="$(mktemp)"
            "${bin}" "${local_args[@]}" 2>&1 | tee "${tmp_out}"

            write_csv_from_output "${tmp_out}" "${csv_file}" \
                "${CSV_LABEL[$abs]}" "${mtype}" "${sz}" "${ts}"

            rm -f "${tmp_out}"
            echo "[run_sptrsv]    → rows appended to ${csv_file}"
        done
    done
done

echo ""
echo "[run_sptrsv] ================================================================"
echo "[run_sptrsv] Done. CSV files:"
ls -lh "${DATA_RAW}"/sptrsv_*_"${PLATFORM}"_"${DATE}".csv 2>/dev/null \
    | awk '{print "  "$NF" ("$5")"}' || true
echo "[run_sptrsv] ================================================================"
echo "[run_sptrsv] Next step: python scripts/analysis/process_e5.py"
