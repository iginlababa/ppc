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
# [B5] AMD MI300X:  hipcc for native (spmv-hip); HIP Kokkos/RAJA pass --amdgpu-target=gfx942.
#                   SYCL via acpp with --acpp-targets=hip:gfx942. No numba on AMD.
#
# Usage:
#   ./scripts/build/build_spmv.sh [--platform nvidia_rtx5060_laptop] [--clean]
#   ./scripts/build/build_spmv.sh --platform amd_mi300x [--clean]
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
                    echo "[build_spmv] WARNING: unknown AMD platform '${PLATFORM}', using ${HIP_ARCH}" ;;
    esac
fi

echo "[build_spmv] =============================================================="
echo "[build_spmv] Platform:   ${PLATFORM}"
if [[ "${VENDOR}" == "amd" ]]; then
    echo "[build_spmv] HIP arch:   ${HIP_ARCH}"
else
    echo "[build_spmv] CUDA arch:  ${ARCH}"
fi
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

# ── spmv-cuda (NVIDIA only) ───────────────────────────────────────────────────
if [[ "${VENDOR}" != "amd" ]]; then
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
fi  # NVIDIA only

# ── spmv-hip (AMD only) ───────────────────────────────────────────────────────
if [[ "${VENDOR}" == "amd" ]]; then
    if ! command -v hipcc &>/dev/null; then
        echo "[build_spmv] ERROR: hipcc not found — cannot build spmv-hip" >&2
        exit 1
    fi
    BUILD_HIP="${BUILD_BASE}/hip_${PLATFORM}"
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_HIP}"
    mkdir -p "${BUILD_HIP}"

    echo "[build_spmv] Building spmv-hip (native CSR, one thread per row, HIP) ..."
    hipcc -O3 -ffast-math -Wall -Wextra \
        --offload-arch="${HIP_ARCH}" \
        -I "${KERNEL_DIR}" \
        "${KERNEL_DIR}/kernel_spmv_hip.cpp" \
        -o "${BUILD_HIP}/spmv-hip"
    echo "[build_spmv]   → ${BUILD_HIP}/spmv-hip"
fi  # AMD only

# ── spmv-kokkos ───────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${_KOKKOS_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
    mkdir -p "${BUILD_KOKKOS}"

    if [[ "${VENDOR}" == "amd" ]]; then
        echo "[build_spmv] Building spmv-kokkos (Kokkos::RangePolicy, HIP backend) ..."
        hipcc -O3 -ffast-math -Wall -Wextra \
            --offload-arch="${HIP_ARCH}" \
            -std=c++17 \
            -I"${KOKKOS_INSTALL_PREFIX}/include" \
            -I"${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_spmv_kokkos.cpp" \
            -L"${KOKKOS_INSTALL_PREFIX}/lib" \
            -lkokkoscore -lkokkoscontainers \
            -lamdhip64 -ldl -lpthread \
            -o "${BUILD_KOKKOS}/spmv-kokkos"
    else
        # Two-step: nvcc -x cu → .o, then nvcc link with Kokkos .a
        echo "[build_spmv] Building spmv-kokkos (Kokkos::RangePolicy, CUDA backend, two-step) ..."

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
    fi

    echo "[build_spmv]   → ${BUILD_KOKKOS}/spmv-kokkos"
else
    echo "[build_spmv] SKIP spmv-kokkos (Kokkos not found)"
fi

# ── spmv-raja ─────────────────────────────────────────────────────────────────
BUILD_RAJA="${BUILD_BASE}/raja_${PLATFORM}"
if [[ "${_RAJA_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_RAJA}"
    mkdir -p "${BUILD_RAJA}"

    if [[ "${VENDOR}" == "amd" ]]; then
        echo "[build_spmv] Building spmv-raja (RAJA::forall over rows, HIP backend, two-step) ..."

        hipcc -O3 -ffast-math \
            --offload-arch="${HIP_ARCH}" \
            -std=c++17 \
            -I "${RAJA_INSTALL_PREFIX}/include" \
            -I "${KERNEL_DIR}" \
            -c "${KERNEL_DIR}/kernel_spmv_raja.cpp" \
            -o "${BUILD_RAJA}/kernel_spmv_raja.o"

        hipcc --offload-arch="${HIP_ARCH}" \
            "${BUILD_RAJA}/kernel_spmv_raja.o" \
            "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
            "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
            -lamdhip64 -ldl -lpthread \
            -o "${BUILD_RAJA}/spmv-raja"
    else
        echo "[build_spmv] Building spmv-raja (RAJA::forall over rows, two-step compile) ..."

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

        g++ -O3 \
            "${BUILD_RAJA}/kernel_spmv_raja.o" \
            "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
            "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
            -lcudart -ldl -lpthread \
            -o "${BUILD_RAJA}/spmv-raja"
    fi

    echo "[build_spmv]   → ${BUILD_RAJA}/spmv-raja"
else
    echo "[build_spmv] SKIP spmv-raja (RAJA not found)"
fi

# ── spmv-sycl ─────────────────────────────────────────────────────────────────
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
    echo "[build_spmv] Building spmv-sycl (nd_range<1>, ${_SYCL_COMPILER}) ..."

    if [[ "${VENDOR}" == "amd" ]]; then
        "${_SYCL_COMPILER}" -O3 \
            --acpp-targets="hip:${HIP_ARCH}" \
            -I "${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_spmv_sycl.cpp" \
            -o "${BUILD_SYCL}/spmv-sycl"
    else
        "${_SYCL_COMPILER}" -O3 -fsycl \
            -fsycl-targets=nvptx64-nvidia-cuda \
            "-Xsycl-target-backend=nvptx64-nvidia-cuda" "--cuda-gpu-arch=${ARCH}" \
            -I "${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_spmv_sycl.cpp" \
            -o "${BUILD_SYCL}/spmv-sycl"
    fi
    echo "[build_spmv]   → ${BUILD_SYCL}/spmv-sycl"
else
    echo "[build_spmv] SKIP spmv-sycl (no SYCL compiler found: acpp / icpx / clang++)"
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

# ── spmv-numba wrapper (NVIDIA only — numba-hip is experimental) ──────────────
if [[ "${VENDOR}" != "amd" ]]; then
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
else
    echo "[build_spmv] SKIP spmv-numba (AMD: numba-hip is experimental, not built)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[build_spmv] =============================================================="
echo "[build_spmv] Build summary:"
find "${BUILD_BASE}" -maxdepth 2 -name "spmv-*" -executable \
    ! -name "*.sh" 2>/dev/null \
    | sort | sed "s|${BUILD_BASE}/|  |"
echo "[build_spmv] Next step: ./scripts/run/run_spmv.sh --platform ${PLATFORM}"
echo "[build_spmv] =============================================================="
