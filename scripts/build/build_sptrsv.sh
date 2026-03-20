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

VENDOR="${PLATFORM%%_*}"   # "nvidia" or "amd"

# ── CUDA architecture detection (NVIDIA only) ─────────────────────────────────
if [[ "${VENDOR}" != "amd" ]]; then
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
fi

# ── AMD GFX target detection ──────────────────────────────────────────────────
if [[ "${VENDOR}" == "amd" ]]; then
    case "${PLATFORM}" in
        amd_mi300x) HIP_ARCH="gfx942" ;;
        amd_mi250x) HIP_ARCH="gfx90a" ;;
        amd_mi100)  HIP_ARCH="gfx908" ;;
        *)          HIP_ARCH="${HIP_ARCH:-gfx90a}"
                    echo "[build_sptrsv] WARNING: unknown AMD platform '${PLATFORM}', using ${HIP_ARCH}" ;;
    esac
fi

echo "[build_sptrsv] =============================================================="
echo "[build_sptrsv] Platform:    ${PLATFORM}"
if [[ "${VENDOR}" == "amd" ]]; then
    echo "[build_sptrsv] HIP arch:    ${HIP_ARCH}"
else
    echo "[build_sptrsv] CUDA arch:   ${ARCH}"
fi
echo "[build_sptrsv] Kernel dir:  ${KERNEL_DIR}"
echo "[build_sptrsv] Build base:  ${BUILD_BASE}"
echo "[build_sptrsv] Verify:      ${VERIFY}"
if [[ "${VENDOR}" == "amd" ]]; then
    echo "[build_sptrsv] Skipped:     numba (AMD: numba-hip experimental)"
else
    echo "[build_sptrsv] Skipped:     numba (UNSUPPORTED_CC120), sycl (NO_COMPILER)"
fi
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

# ── sptrsv-cuda (NVIDIA only) ─────────────────────────────────────────────────
if [[ "${VENDOR}" != "amd" ]]; then
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
fi  # NVIDIA only

# ── sptrsv-hip (AMD only) ─────────────────────────────────────────────────────
if [[ "${VENDOR}" == "amd" ]]; then
    if ! command -v hipcc &>/dev/null; then
        echo "[build_sptrsv] ERROR: hipcc not found — cannot build sptrsv-hip" >&2
        exit 1
    fi
    BUILD_HIP="${BUILD_BASE}/hip_${PLATFORM}"
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_HIP}"
    mkdir -p "${BUILD_HIP}"

    echo "[build_sptrsv] Building sptrsv-hip (native HIP level-set SpTRSV) ..."
    hipcc -O3 -ffast-math -Wall -Wextra \
        --offload-arch="${HIP_ARCH}" \
        -I "${KERNEL_DIR}" \
        "${KERNEL_DIR}/kernel_sptrsv_hip.cpp" \
        -o "${BUILD_HIP}/sptrsv-hip"
    echo "[build_sptrsv]   → ${BUILD_HIP}/sptrsv-hip"

    if [[ "${VERIFY}" == "true" ]]; then
        echo "[build_sptrsv]   Verifying sptrsv-hip ..."
        "${BUILD_HIP}/sptrsv-hip" --verify \
            --matrix lower_triangular_laplacian --n 256
        echo "[build_sptrsv]   sptrsv-hip verify: OK"
    fi
fi  # AMD only

# ── sptrsv-kokkos ─────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${_KOKKOS_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
    mkdir -p "${BUILD_KOKKOS}"

    if [[ "${VENDOR}" == "amd" ]]; then
        echo "[build_sptrsv] Building sptrsv-kokkos (Kokkos level-set, HIP backend, two-step) ..."

        hipcc -O3 -ffast-math \
            --offload-arch="${HIP_ARCH}" \
            -std=c++17 \
            -I"${KOKKOS_INSTALL_PREFIX}/include" \
            -I"${KERNEL_DIR}" \
            -c "${KERNEL_DIR}/kernel_sptrsv_kokkos.cpp" \
            -o "${BUILD_KOKKOS}/kernel_sptrsv_kokkos.o"

        hipcc --offload-arch="${HIP_ARCH}" \
            "${BUILD_KOKKOS}/kernel_sptrsv_kokkos.o" \
            "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscore.a" \
            "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscontainers.a" \
            -lamdhip64 -ldl -lpthread \
            -o "${BUILD_KOKKOS}/sptrsv-kokkos"
    else
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
    fi

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

    if [[ "${VENDOR}" == "amd" ]]; then
        echo "[build_sptrsv] Building sptrsv-raja (RAJA level-set, HIP backend, two-step) ..."

        hipcc -O3 -ffast-math \
            --offload-arch="${HIP_ARCH}" \
            -std=c++17 \
            -I "${RAJA_INSTALL_PREFIX}/include" \
            -I "${KERNEL_DIR}" \
            -c "${KERNEL_DIR}/kernel_sptrsv_raja.cpp" \
            -o "${BUILD_RAJA}/kernel_sptrsv_raja.o"

        hipcc --offload-arch="${HIP_ARCH}" \
            "${BUILD_RAJA}/kernel_sptrsv_raja.o" \
            "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
            "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
            -lamdhip64 -ldl -lpthread \
            -o "${BUILD_RAJA}/sptrsv-raja"
    else
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
    fi

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

# ── sptrsv-sycl ───────────────────────────────────────────────────────────────
BUILD_SYCL="${BUILD_BASE}/sycl_${PLATFORM}"
_SYCL_COMPILER=""
if [[ "${VENDOR}" == "amd" ]]; then
    if command -v acpp &>/dev/null; then
        _SYCL_COMPILER="acpp"
    fi
else
    for _c in icpx acpp clang++; do
        if command -v "${_c}" &>/dev/null; then
            _SYCL_COMPILER="${_c}"
            break
        fi
    done
fi

if [[ -n "${_SYCL_COMPILER}" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_SYCL}"
    mkdir -p "${BUILD_SYCL}"
    echo "[build_sptrsv] Building sptrsv-sycl (nd_range<1> level-set, ${_SYCL_COMPILER}) ..."

    if [[ "${VENDOR}" == "amd" ]]; then
        "${_SYCL_COMPILER}" -O3 \
            --acpp-targets="hip:${HIP_ARCH}" \
            -I "${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_sptrsv_sycl.cpp" \
            -o "${BUILD_SYCL}/sptrsv-sycl"
    else
        "${_SYCL_COMPILER}" -O3 -fsycl \
            -fsycl-targets=nvptx64-nvidia-cuda \
            "-Xsycl-target-backend=nvptx64-nvidia-cuda" "--cuda-gpu-arch=${ARCH}" \
            -I "${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_sptrsv_sycl.cpp" \
            -o "${BUILD_SYCL}/sptrsv-sycl"
    fi
    echo "[build_sptrsv]   → ${BUILD_SYCL}/sptrsv-sycl"

    if [[ "${VERIFY}" == "true" ]]; then
        echo "[build_sptrsv]   Verifying sptrsv-sycl ..."
        "${BUILD_SYCL}/sptrsv-sycl" --verify \
            --matrix lower_triangular_laplacian --n 256
        echo "[build_sptrsv]   sptrsv-sycl verify: OK"
    fi
else
    if [[ "${VENDOR}" == "amd" ]]; then
        echo "[build_sptrsv] NO_COMPILER: sycl — acpp not found on ${PLATFORM}."
    else
        echo "[build_sptrsv] NO_COMPILER: sycl — no SYCL compiler (icpx/acpp/clang++) on ${PLATFORM}."
    fi
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

# ── Numba: skipped on both platforms ──────────────────────────────────────────
echo ""
if [[ "${VENDOR}" == "amd" ]]; then
    echo "[build_sptrsv] SKIP: numba — numba-hip is experimental, not built on AMD."
else
    echo "[build_sptrsv] UNSUPPORTED_CC120: numba — Numba 0.64.0 generates PTX 9.2; driver"
    echo "[build_sptrsv]   on Blackwell (CC 12.0) rejects PTX > 9.1. Platform limitation."
fi
echo "[build_sptrsv]   No sptrsv-numba binary created."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[build_sptrsv] =============================================================="
echo "[build_sptrsv] Build summary:"
if [[ "${VENDOR}" == "amd" ]]; then
    [[ -x "${BUILD_HIP}/sptrsv-hip"       ]] && echo "  [OK] sptrsv-hip    → ${BUILD_HIP}/sptrsv-hip"
else
    [[ -x "${BUILD_CUDA}/sptrsv-cuda"     ]] && echo "  [OK] sptrsv-cuda   → ${BUILD_CUDA}/sptrsv-cuda"
fi
[[ -x "${BUILD_KOKKOS}/sptrsv-kokkos" ]] && echo "  [OK] sptrsv-kokkos → ${BUILD_KOKKOS}/sptrsv-kokkos"
[[ -x "${BUILD_RAJA}/sptrsv-raja"     ]] && echo "  [OK] sptrsv-raja   → ${BUILD_RAJA}/sptrsv-raja"
[[ -n "${_SYCL_COMPILER}" ]] && [[ -x "${BUILD_SYCL}/sptrsv-sycl" ]] \
    && echo "  [OK] sptrsv-sycl   → ${BUILD_SYCL}/sptrsv-sycl"
[[ -x "${JULIA_WRAP}"                 ]] && echo "  [OK] sptrsv-julia  → ${JULIA_WRAP}"
echo "  [--] sptrsv-numba  SKIP (AMD: numba-hip experimental; NVIDIA: UNSUPPORTED_CC120)"
echo "[build_sptrsv] Next step: ./scripts/run/run_sptrsv.sh --platform ${PLATFORM}"
echo "[build_sptrsv] =============================================================="
