#!/usr/bin/env bash
# Run E4 SpMV — all abstractions, all matrix types, all sizes, write per-run CSV rows.
#
# E4 DESIGN DECISIONS
# [D6] experiment_id: spmv_{abstraction}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}
# [D7] Warmup: adaptive (CV < 2% over last 10 timings), max 200 iterations.
#      Kernels handle warmup internally; --warmup passes the max cap.
# CSV columns: timestamp,experiment_id,kernel,abstraction,platform,
#              matrix_type,problem_size,n_rows,nnz,
#              run_id,execution_time_ms,throughput_gflops,hw_state_verified
#
# Usage:
#   ./scripts/run/run_spmv.sh --platform nvidia_rtx5060
#   ./scripts/run/run_spmv.sh --platform nvidia_rtx5060 \
#       --abstraction native --matrix laplacian_2d --size large \
#       --warmup 200 --reps 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build/spmv"
DATA_RAW="${REPO_ROOT}/data/raw"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060"
ABSTRACTION="all"
MATRIX="all"
SIZE="all"
WARMUP=200      # max adaptive warmup iterations (kernel stops early at CV<2%)
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

# ── Problem sizes (D1: small=1024, medium=8192, large=32768 rows) ─────────────
declare -A SIZES
SIZES[small]=1024
SIZES[medium]=8192
SIZES[large]=32768

# ── Matrix types (D2) ─────────────────────────────────────────────────────────
ALL_MATRICES=(laplacian_2d random_sparse power_law)

# ── Abstraction registry ──────────────────────────────────────────────────────
declare -A BINARY_NAME   # abstraction → executable stem
declare -A CSV_LABEL     # abstraction → label written to CSV

BINARY_NAME[native]="spmv-cuda"
CSV_LABEL[native]="native"

BINARY_NAME[kokkos]="spmv-kokkos"
CSV_LABEL[kokkos]="kokkos"

BINARY_NAME[raja]="spmv-raja"
CSV_LABEL[raja]="raja"

BINARY_NAME[sycl]="spmv-sycl"
CSV_LABEL[sycl]="sycl"

BINARY_NAME[julia]="spmv-julia"
CSV_LABEL[julia]="julia"

BINARY_NAME[numba]="spmv-numba"
CSV_LABEL[numba]="numba"

ALL_ABSTRACTIONS=(native kokkos raja sycl julia numba)

# ── Binary finder ─────────────────────────────────────────────────────────────
find_binary() {
    local abs="$1"
    local bin_name="${BINARY_NAME[$abs]}"
    local dir_stem
    case "${abs}" in
        native)  dir_stem="cuda"   ;;
        kokkos)  dir_stem="kokkos" ;;
        raja)    dir_stem="raja"   ;;
        sycl)    dir_stem="sycl"   ;;
        julia)   dir_stem="julia"  ;;
        numba)   dir_stem="numba"  ;;
        *)       dir_stem="${abs}" ;;
    esac

    local p="${BUILD_BASE}/${dir_stem}_${PLATFORM}/${bin_name}"
    [[ -x "${p}" ]] && { echo "${p}"; return 0; }

    # Also search all build subdirs (platform-agnostic fallback)
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
# Parses SPMV_RUN / SPMV_HW_STATE lines from binary stdout → CSV rows.
write_csv_from_output() {
    local output_file="$1"
    local csv_file="$2"
    local abs_label="$3"
    local matrix_type="$4"
    local size_label="$5"
    local ts="$6"

    # Extract N and nnz from the header comment printed by the binary
    # Pattern: "# abstraction=xxx matrix=yyy N=NNN nnz=ZZZ warmup_max=..."
    local N=0
    local NNZ=0
    local header_line
    header_line="$(grep "^# abstraction=" "${output_file}" | grep "N=" | tail -1 || true)"
    if [[ -n "${header_line}" ]]; then
        if [[ "${header_line}" =~ N=([0-9]+) ]]; then
            N="${BASH_REMATCH[1]}"
        fi
        if [[ "${header_line}" =~ nnz=([0-9]+) ]]; then
            NNZ="${BASH_REMATCH[1]}"
        fi
    fi

    # Build hw_state map from SPMV_HW_STATE lines
    declare -A hw_map
    while IFS= read -r line; do
        if [[ "${line}" =~ ^SPMV_HW_STATE\ run=([0-9]+)\ hw_state=([01])$ ]]; then
            hw_map["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done < "${output_file}"

    # Parse SPMV_RUN lines and write CSV rows
    while IFS= read -r line; do
        if [[ "${line}" =~ ^SPMV_RUN\ run=([0-9]+)\ n=([0-9]+)\ nnz=([0-9]+)\ matrix=([a-z_0-9]+)\ time_ms=([0-9.]+)\ throughput_gflops=([0-9.]+)$ ]]; then
            local run_id="${BASH_REMATCH[1]}"
            local n_rows="${BASH_REMATCH[2]}"
            local nnz="${BASH_REMATCH[3]}"
            local mtype="${BASH_REMATCH[4]}"
            local time_ms="${BASH_REMATCH[5]}"
            local gflops="${BASH_REMATCH[6]}"
            local hw="${hw_map[$run_id]:-1}"
            local run_id_padded
            run_id_padded="$(printf '%03d' "${run_id}")"
            local exp_id="spmv_${abs_label}_${PLATFORM}_${mtype}_${size_label}_n${n_rows}_${run_id_padded}"
            printf '%s,%s,spmv,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${ts}" "${exp_id}" "${abs_label}" "${PLATFORM}" \
                "${mtype}" "${size_label}" "${n_rows}" "${nnz}" \
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

echo "[run_spmv] ================================================================"
echo "[run_spmv] Platform:      ${PLATFORM}"
echo "[run_spmv] Warmup (max):  ${WARMUP}  (adaptive CV<2% protocol)"
echo "[run_spmv] Reps:          ${REPS}"
echo "[run_spmv] Sizes:         ${run_sizes[*]}"
echo "[run_spmv] Abstractions:  ${run_abstractions[*]}"
echo "[run_spmv] Matrix types:  ${run_matrices[*]}"
echo "[run_spmv] ================================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for abs in "${run_abstractions[@]}"; do
    bin="$(find_binary "${abs}")"
    if [[ -z "${bin}" || ! -x "${bin}" ]]; then
        echo "[run_spmv] SKIP ${abs}: binary not found (run build_spmv.sh first)"
        continue
    fi

    csv_file="${DATA_RAW}/spmv_${CSV_LABEL[$abs]}_${PLATFORM}_${DATE}.csv"

    if [[ ! -f "${csv_file}" ]]; then
        echo "timestamp,experiment_id,kernel,abstraction,platform,matrix_type,problem_size,n_rows,nnz,run_id,execution_time_ms,throughput_gflops,hw_state_verified" \
            > "${csv_file}"
    fi

    for mtype in "${run_matrices[@]}"; do
        for sz in "${run_sizes[@]}"; do
            N="${SIZES[$sz]}"
            ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

            echo ""
            echo "[run_spmv] ── ${abs} / ${mtype} / ${sz} (N=${N}) ──────────────"
            echo "[run_spmv]    binary: ${bin}"
            echo "[run_spmv]    csv:    ${csv_file}"

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

            # Detect Numba PTX incompatibility (CC 12.0 / Blackwell)
            if grep -q "PTX_VERSION_MISMATCH" "${tmp_out}" 2>/dev/null; then
                echo "[run_spmv] UNSUPPORTED_CC120 ${abs}: Numba 0.64.0 does not support" \
                     "CC 12.0 (Blackwell) — PTX 9.2 rejected by driver (max PTX 9.1)." \
                     "Platform limitation, not a SKIP."
            fi

            write_csv_from_output "${tmp_out}" "${csv_file}" \
                "${CSV_LABEL[$abs]}" "${mtype}" "${sz}" "${ts}"

            rm -f "${tmp_out}"
            echo "[run_spmv]    → rows appended to ${csv_file}"
        done
    done
done

echo ""
echo "[run_spmv] ================================================================"
echo "[run_spmv] Done. CSV files:"
ls -lh "${DATA_RAW}"/spmv_*_"${PLATFORM}"_"${DATE}".csv 2>/dev/null \
    | awk '{print "  "$NF" ("$5")"}' || true
echo "[run_spmv] ================================================================"
echo "[run_spmv] Next step: python scripts/process_e4.py"
