#!/usr/bin/env bash
# Build all abstraction variants of SpTRSV (E5).
# Same two-step nvcc pattern as E4 for Kokkos and RAJA.
#
# E5 BUILD DECISIONS
# [B1] sptrsv-cuda:   direct nvcc — identical flags to E4.
# [B2] sptrsv-kokkos: two-step nvcc: -x cu -c → .o, then nvcc link with libkokkoscore.a.
#                     -std=c++20 required for Kokkos structured bindings.
# [B3] sptrsv-raja:   two-step: nvcc -x cu → .o, then g++ link with libRAJA.a + libcamp.a.
# [B4] sptrsv-julia:  bash wrapper script pointing at kernel_sptrsv_julia.jl.
# [B5] Numba: SKIP — UNSUPPORTED_CC120. Numba 0.64.0 generates PTX 9.2 which is
#             rejected by the driver on Blackwell (CC 12.0, max driver PTX 9.1).
#             Platform limitation, not a fixable environment gap.
# [B6] SYCL:  SKIP — NO_COMPILER. No SYCL compiler on nvidia_rtx5060_laptop.
# [B7] --verify: runs each C++ binary with --verify flag after build to confirm
#             correctness (forward substitution vs CPU reference, max_rel_err < 1e-10).
#             Julia correctness check is deferred to run_sptrsv.sh --verify.
#
# Usage:
#   ./scripts/build/build_sptrsv.sh [--platform nvidia_rtx5060_laptop] [--clean] [--verify]
#
# Environment overrides:
#   KOKKOS_INSTALL_PREFIX  — override Kokkos install directory
#   RAJA_INSTALL_PREFIX    — override RAJA install directory
#   CUDA_ARCH              — override CUDA architecture (e.g. sm_120)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernels/sptrsv"
BUILD_BASE="${REPO_ROOT}/build/sptrsv"

PLATFORM="nvidia_rtx5060_laptop"
CLEAN=false
VERIFY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --clean)    CLEAN=true;    shift ;;
        --verify)   VERIFY=true;   shift ;;
        *) echo "[build_sptrsv] Unknown argument: $1" >&2; exit 1 ;;
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
    ARCH="sm_80"
fi

echo "[build_sptrsv] =============================================================="
echo "[build_sptrsv] Platform:    ${PLATFORM}"
echo "[build_sptrsv] CUDA arch:   ${ARCH}"
echo "[build_sptrsv] Kernel dir:  ${KERNEL_DIR}"
echo "[build_sptrsv] Build base:  ${BUILD_BASE}"
echo "[build_sptrsv] Verify:      ${VERIFY}"
echo "[build_sptrsv] Skipped:     numba (UNSUPPORTED_CC120), sycl (NO_COMPILER)"
echo "[build_sptrsv] =============================================================="
echo ""

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
    echo "[build_sptrsv] Kokkos found: ${KOKKOS_INSTALL_PREFIX}"
    _KOKKOS_OK=true
else
    echo "[build_sptrsv] WARNING: Kokkos not found — sptrsv-kokkos will be skipped."
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
    echo "[build_sptrsv] RAJA found:   ${RAJA_INSTALL_PREFIX}"
    _RAJA_OK=true
else
    echo "[build_sptrsv] WARNING: RAJA not found — sptrsv-raja will be skipped."
    _RAJA_OK=false
fi

echo ""

# ── sptrsv-cuda ───────────────────────────────────────────────────────────────
BUILD_CUDA="${BUILD_BASE}/cuda_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_CUDA}"
mkdir -p "${BUILD_CUDA}"

echo "[build_sptrsv] Building sptrsv-cuda (native CUDA level-set SpTRSV) ..."
nvcc -O3 -arch="${ARCH}" \
    --use_fast_math --generate-line-info \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wall,-Wextra \
    -I "${KERNEL_DIR}" \
    "${KERNEL_DIR}/kernel_sptrsv_cuda.cu" \
    -lcudart \
    -o "${BUILD_CUDA}/sptrsv-cuda"
echo "[build_sptrsv]   → ${BUILD_CUDA}/sptrsv-cuda"

if [[ "${VERIFY}" == "true" ]]; then
    echo "[build_sptrsv]   Verifying sptrsv-cuda ..."
    "${BUILD_CUDA}/sptrsv-cuda" --verify \
        --matrix lower_triangular_laplacian --n 256
    echo "[build_sptrsv]   sptrsv-cuda verify: OK"
fi

# ── sptrsv-kokkos ─────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${_KOKKOS_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
    mkdir -p "${BUILD_KOKKOS}"

    echo "[build_sptrsv] Building sptrsv-kokkos (Kokkos level-set, two-step) ..."

    nvcc -O3 -arch="${ARCH}" \
        --expt-extended-lambda --expt-relaxed-constexpr \
        --use_fast_math --generate-line-info \
        -std=c++20 \
        -x cu -c \
        -I"${KOKKOS_INSTALL_PREFIX}/include" \
        -I"${KERNEL_DIR}" \
        "${KERNEL_DIR}/kernel_sptrsv_kokkos.cpp" \
        -o "${BUILD_KOKKOS}/kernel_sptrsv_kokkos.o"

    nvcc -O3 -arch="${ARCH}" -std=c++20 \
        "${BUILD_KOKKOS}/kernel_sptrsv_kokkos.o" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscore.a" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscontainers.a" \
        -lcudart -lcuda -ldl -lpthread \
        -o "${BUILD_KOKKOS}/sptrsv-kokkos"

    echo "[build_sptrsv]   → ${BUILD_KOKKOS}/sptrsv-kokkos"

    if [[ "${VERIFY}" == "true" ]]; then
        echo "[build_sptrsv]   Verifying sptrsv-kokkos ..."
        "${BUILD_KOKKOS}/sptrsv-kokkos" --verify \
            --matrix lower_triangular_laplacian --n 256
        echo "[build_sptrsv]   sptrsv-kokkos verify: OK"
    fi
else
    echo "[build_sptrsv] SKIP sptrsv-kokkos (Kokkos not found)"
fi

# ── sptrsv-raja ───────────────────────────────────────────────────────────────
BUILD_RAJA="${BUILD_BASE}/raja_${PLATFORM}"
if [[ "${_RAJA_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_RAJA}"
    mkdir -p "${BUILD_RAJA}"

    echo "[build_sptrsv] Building sptrsv-raja (RAJA level-set, two-step) ..."

    # Step 1: nvcc -x cu → object file
    nvcc -O3 -arch="${ARCH}" \
        --use_fast_math --generate-line-info \
        --expt-extended-lambda --expt-relaxed-constexpr \
        -allow-unsupported-compiler \
        -Xcompiler=-Wall,-Wextra \
        -I "${RAJA_INSTALL_PREFIX}/include" \
        -I "${KERNEL_DIR}" \
        -x cu -c \
        "${KERNEL_DIR}/kernel_sptrsv_raja.cpp" \
        -o "${BUILD_RAJA}/kernel_sptrsv_raja.o"

    # Step 2: g++ link
    g++ -O3 \
        "${BUILD_RAJA}/kernel_sptrsv_raja.o" \
        "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
        "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
        -lcudart -ldl -lpthread \
        -o "${BUILD_RAJA}/sptrsv-raja"

    echo "[build_sptrsv]   → ${BUILD_RAJA}/sptrsv-raja"

    if [[ "${VERIFY}" == "true" ]]; then
        echo "[build_sptrsv]   Verifying sptrsv-raja ..."
        "${BUILD_RAJA}/sptrsv-raja" --verify \
            --matrix lower_triangular_laplacian --n 256
        echo "[build_sptrsv]   sptrsv-raja verify: OK"
    fi
else
    echo "[build_sptrsv] SKIP sptrsv-raja (RAJA not found)"
fi

# ── sptrsv-julia wrapper ──────────────────────────────────────────────────────
BUILD_JULIA="${BUILD_BASE}/julia_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_JULIA}"
mkdir -p "${BUILD_JULIA}"

JULIA_SRC="${KERNEL_DIR}/kernel_sptrsv_julia.jl"
JULIA_WRAP="${BUILD_JULIA}/sptrsv-julia"
cat > "${JULIA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated by build_sptrsv.sh — do not edit.
JULIA_BIN="\${JULIA_BIN:-julia}"
exec "\${JULIA_BIN}" --startup-file=no --project=@. \\
     "${JULIA_SRC}" "\$@"
EOF
chmod +x "${JULIA_WRAP}"
echo "[build_sptrsv]   → ${JULIA_WRAP} (Julia wrapper; correctness check via run_sptrsv.sh --verify)"

# ── Numba: UNSUPPORTED_CC120 ──────────────────────────────────────────────────
echo ""
echo "[build_sptrsv] UNSUPPORTED_CC120: numba — Numba 0.64.0 generates PTX 9.2; driver"
echo "[build_sptrsv]   on Blackwell (CC 12.0) rejects PTX > 9.1. Platform limitation."
echo "[build_sptrsv]   No sptrsv-numba binary created."

# ── SYCL: NO_COMPILER ─────────────────────────────────────────────────────────
echo "[build_sptrsv] NO_COMPILER: sycl — no SYCL compiler installed on ${PLATFORM}."
echo "[build_sptrsv]   No sptrsv-sycl binary created."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[build_sptrsv] =============================================================="
echo "[build_sptrsv] Build summary:"
[[ -x "${BUILD_CUDA}/sptrsv-cuda"     ]] && echo "  [OK] sptrsv-cuda   → ${BUILD_CUDA}/sptrsv-cuda"
[[ -x "${BUILD_KOKKOS}/sptrsv-kokkos" ]] && echo "  [OK] sptrsv-kokkos → ${BUILD_KOKKOS}/sptrsv-kokkos"
[[ -x "${BUILD_RAJA}/sptrsv-raja"     ]] && echo "  [OK] sptrsv-raja   → ${BUILD_RAJA}/sptrsv-raja"
[[ -x "${JULIA_WRAP}"                 ]] && echo "  [OK] sptrsv-julia  → ${JULIA_WRAP}"
echo "  [--] sptrsv-numba  UNSUPPORTED_CC120 (Numba 0.64.0 / Blackwell CC 12.0)"
echo "  [--] sptrsv-sycl   NO_COMPILER"
echo "[build_sptrsv] Next step: ./scripts/run/run_sptrsv.sh --platform ${PLATFORM}"
echo "[build_sptrsv] =============================================================="
