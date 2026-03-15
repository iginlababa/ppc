#!/usr/bin/env bash
# Build all abstraction variants of SpMV (E4).
# Direct compilation — bypasses CMake 4.x + CUDA 12.8 sm_52 probe incompatibility.
#
# E4 BUILD DECISIONS
# [B1] spmv-cuda:   direct nvcc (no CMake) — same pattern as E3.
# [B2] spmv-kokkos: kokkos_launch_compiler + nvcc_wrapper, sm_120 -extended-lambda flag.
#                   Requires an existing Kokkos install; auto-detected from common paths.
# [B3] spmv-raja:   two-step (nvcc -x cu → .o, then g++ link with libRAJA.a + libcamp.a).
#                   Single-step nvcc + .a input treats .a as source and produces garbage binary.
# [B4] spmv-julia/numba: bash wrapper scripts pointing at the kernel source files.
#
# Usage:
#   ./scripts/build/build_spmv.sh [--platform nvidia_rtx5060_laptop] [--clean]
#   ./scripts/build/build_spmv.sh --platform nvidia_a100 --clean
#
# Environment overrides:
#   KOKKOS_INSTALL_PREFIX  — override Kokkos install directory
#   RAJA_INSTALL_PREFIX    — override RAJA install directory
#   CUDA_ARCH              — override CUDA architecture (e.g. sm_120)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernels/spmv"
BUILD_BASE="${REPO_ROOT}/build/spmv"

PLATFORM="nvidia_rtx5060_laptop"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --clean)    CLEAN=true; shift ;;
        *) echo "[build_spmv] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── CUDA architecture detection ───────────────────────────────────────────────
if [[ -n "${CUDA_ARCH:-}" ]]; then
    ARCH="${CUDA_ARCH}"
elif [[ "${PLATFORM}" == *"rtx5060"* ]] || [[ "${PLATFORM}" == *"blackwell"* ]]; then
    ARCH="sm_120"
elif [[ "${PLATFORM}" == *"a100"* ]]; then
    ARCH="sm_80"
elif [[ "${PLATFORM}" == *"v100"* ]]; then
    ARCH="sm_70"
elif [[ "${PLATFORM}" == *"a40"* ]]; then
    ARCH="sm_86"
else
    ARCH="sm_80"   # safe default for modern NVIDIA GPUs
fi

echo "[build_spmv] =============================================================="
echo "[build_spmv] Platform: ${PLATFORM}"
echo "[build_spmv] CUDA arch: ${ARCH}"
echo "[build_spmv] Kernel dir: ${KERNEL_DIR}"
echo "[build_spmv] Build base: ${BUILD_BASE}"
echo "[build_spmv] =============================================================="

# ── Kokkos detection ──────────────────────────────────────────────────────────
if [[ -z "${KOKKOS_INSTALL_PREFIX:-}" ]]; then
    for _p in \
        "/home/obalola/projects/kokkos-cuda-install" \
        "${REPO_ROOT}/build/stream/kokkos_${PLATFORM}" \
        "/home/obalola/projects/kokkos/install" \
        "/usr/local" \
        "/opt/kokkos"; do
        if [[ -f "${_p}/include/Kokkos_Core.hpp" ]]; then
            KOKKOS_INSTALL_PREFIX="${_p}"
            break
        fi
    done
fi
if [[ -n "${KOKKOS_INSTALL_PREFIX:-}" ]] && [[ -f "${KOKKOS_INSTALL_PREFIX}/include/Kokkos_Core.hpp" ]]; then
    echo "[build_spmv] Kokkos found: ${KOKKOS_INSTALL_PREFIX}"
    _KOKKOS_OK=true
else
    echo "[build_spmv] WARNING: Kokkos not found — spmv-kokkos will be skipped."
    echo "[build_spmv]          Set KOKKOS_INSTALL_PREFIX= to override."
    _KOKKOS_OK=false
fi

# ── RAJA detection ────────────────────────────────────────────────────────────
if [[ -z "${RAJA_INSTALL_PREFIX:-}" ]]; then
    for _p in \
        "/home/obalola/projects/raja/install" \
        "/usr/local" \
        "/opt/raja"; do
        if [[ -f "${_p}/include/RAJA/RAJA.hpp" ]]; then
            RAJA_INSTALL_PREFIX="${_p}"
            break
        fi
    done
fi
if [[ -n "${RAJA_INSTALL_PREFIX:-}" ]] && [[ -f "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp" ]]; then
    echo "[build_spmv] RAJA found: ${RAJA_INSTALL_PREFIX}"
    _RAJA_OK=true
else
    echo "[build_spmv] WARNING: RAJA not found — spmv-raja will be skipped."
    echo "[build_spmv]          Set RAJA_INSTALL_PREFIX= to override."
    _RAJA_OK=false
fi

echo ""

# ── spmv-cuda ─────────────────────────────────────────────────────────────────
BUILD_CUDA="${BUILD_BASE}/cuda_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_CUDA}"
mkdir -p "${BUILD_CUDA}"

echo "[build_spmv] Building spmv-cuda (native CSR, one thread per row) ..."
nvcc -O3 -arch="${ARCH}" \
    --use_fast_math --generate-line-info \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wall,-Wextra \
    -I "${KERNEL_DIR}" \
    "${KERNEL_DIR}/kernel_spmv_cuda.cu" \
    -lcudart \
    -o "${BUILD_CUDA}/spmv-cuda"
echo "[build_spmv]   → ${BUILD_CUDA}/spmv-cuda"

# ── spmv-kokkos ───────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${_KOKKOS_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
    mkdir -p "${BUILD_KOKKOS}"

    # Two-step: nvcc -x cu → .o, then nvcc link with Kokkos .a
    # Note: g++ link fails with libkokkoscore.a (CUDA driver API refs); must use nvcc to link.
    # Note: nvcc_wrapper/kokkos_launch_compiler require -DKOKKOS_DEPENDENCE and CXX matching;
    #       direct nvcc -x cu is simpler and equivalent for single-TU builds.
    echo "[build_spmv] Building spmv-kokkos (Kokkos::RangePolicy over rows, two-step) ..."

    nvcc -O3 -arch="${ARCH}" \
        --expt-extended-lambda --expt-relaxed-constexpr \
        --use_fast_math --generate-line-info \
        -std=c++20 \
        -x cu -c \
        -I"${KOKKOS_INSTALL_PREFIX}/include" \
        -I"${KERNEL_DIR}" \
        "${KERNEL_DIR}/kernel_spmv_kokkos.cpp" \
        -o "${BUILD_KOKKOS}/kernel_spmv_kokkos.o"

    nvcc -O3 -arch="${ARCH}" -std=c++20 \
        "${BUILD_KOKKOS}/kernel_spmv_kokkos.o" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscore.a" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscontainers.a" \
        -lcudart -lcuda -ldl -lpthread \
        -o "${BUILD_KOKKOS}/spmv-kokkos"

    echo "[build_spmv]   → ${BUILD_KOKKOS}/spmv-kokkos"
else
    echo "[build_spmv] SKIP spmv-kokkos (Kokkos not found)"
fi

# ── spmv-raja ─────────────────────────────────────────────────────────────────
BUILD_RAJA="${BUILD_BASE}/raja_${PLATFORM}"
if [[ "${_RAJA_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_RAJA}"
    mkdir -p "${BUILD_RAJA}"

    echo "[build_spmv] Building spmv-raja (RAJA::forall over rows, two-step compile) ..."

    # Step 1: nvcc -x cu → object file
    nvcc -O3 -arch="${ARCH}" \
        --use_fast_math --generate-line-info \
        --expt-extended-lambda --expt-relaxed-constexpr \
        -allow-unsupported-compiler \
        -Xcompiler=-Wall,-Wextra \
        -I "${RAJA_INSTALL_PREFIX}/include" \
        -I "${KERNEL_DIR}" \
        -x cu -c \
        "${KERNEL_DIR}/kernel_spmv_raja.cpp" \
        -o "${BUILD_RAJA}/kernel_spmv_raja.o"

    # Step 2: g++ link
    g++ -O3 \
        "${BUILD_RAJA}/kernel_spmv_raja.o" \
        "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
        "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
        -lcudart -ldl -lpthread \
        -o "${BUILD_RAJA}/spmv-raja"

    echo "[build_spmv]   → ${BUILD_RAJA}/spmv-raja"
else
    echo "[build_spmv] SKIP spmv-raja (RAJA not found)"
fi

# ── spmv-julia wrapper ────────────────────────────────────────────────────────
BUILD_JULIA="${BUILD_BASE}/julia_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_JULIA}"
mkdir -p "${BUILD_JULIA}"

JULIA_SRC="${KERNEL_DIR}/kernel_spmv_julia.jl"
JULIA_WRAP="${BUILD_JULIA}/spmv-julia"
cat > "${JULIA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated by build_spmv.sh — do not edit.
JULIA_BIN="\${JULIA_BIN:-julia}"
exec "\${JULIA_BIN}" --startup-file=no --project=@. \\
     "${JULIA_SRC}" "\$@"
EOF
chmod +x "${JULIA_WRAP}"
echo "[build_spmv]   → ${JULIA_WRAP}"

# ── spmv-numba wrapper ────────────────────────────────────────────────────────
BUILD_NUMBA="${BUILD_BASE}/numba_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_NUMBA}"
mkdir -p "${BUILD_NUMBA}"

NUMBA_SRC="${KERNEL_DIR}/kernel_spmv_numba.py"
NUMBA_WRAP="${BUILD_NUMBA}/spmv-numba"
cat > "${NUMBA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated by build_spmv.sh — do not edit.
PYTHON_BIN="\${PYTHON_BIN:-python3}"
exec "\${PYTHON_BIN}" "${NUMBA_SRC}" "\$@"
EOF
chmod +x "${NUMBA_WRAP}"
echo "[build_spmv]   → ${NUMBA_WRAP}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[build_spmv] =============================================================="
echo "[build_spmv] Build summary:"
[[ -x "${BUILD_CUDA}/spmv-cuda"     ]] && echo "  [OK] spmv-cuda    → ${BUILD_CUDA}/spmv-cuda"
[[ -x "${BUILD_KOKKOS}/spmv-kokkos" ]] && echo "  [OK] spmv-kokkos  → ${BUILD_KOKKOS}/spmv-kokkos"
[[ -x "${BUILD_RAJA}/spmv-raja"     ]] && echo "  [OK] spmv-raja    → ${BUILD_RAJA}/spmv-raja"
[[ -x "${JULIA_WRAP}"               ]] && echo "  [OK] spmv-julia   → ${JULIA_WRAP}"
[[ -x "${NUMBA_WRAP}"               ]] && echo "  [OK] spmv-numba   → ${NUMBA_WRAP}"
echo "[build_spmv] Next step: ./scripts/run/run_spmv.sh --platform ${PLATFORM}"
echo "[build_spmv] =============================================================="
