#!/usr/bin/env bash
# Build all abstraction variants of N-Body (E7).
# Direct compilation — bypasses CMake 4.x + CUDA 12.8 sm_52 probe incompatibility.
#
# E7 BUILD DECISIONS
# [B1] nbody-native: direct nvcc, notile + tile kernels in single TU.
# [B2] nbody-kokkos: nvcc -x cu -c -std=c++20 → nvcc link with libkokkoscore.a.
#                    Must use nvcc to link (libkokkoscore.a has CUDA driver API refs).
# [B3] nbody-raja:   two-step: nvcc -x cu -dc → nvcc link with libRAJA.a + libcamp.a.
#                    harmless warning in RAJA/policy/cuda/policy.hpp:1936 with sm_120.
# [B4] nbody-julia:  bash wrapper script pointing at nbody_julia.jl.
#
# Usage:
#   ./scripts/e7_nbody/build_nbody.sh [--platform nvidia_rtx5060] [--clean]
#
# Environment overrides:
#   KOKKOS_INSTALL_PREFIX  — override Kokkos install directory
#   RAJA_INSTALL_PREFIX    — override RAJA install directory
#   CUDA_ARCH              — override CUDA architecture (e.g. sm_120)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src/e7_nbody"
BUILD_BASE="${REPO_ROOT}/build/e7_nbody"

PLATFORM="nvidia_rtx5060"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --clean)    CLEAN=true; shift ;;
        *) echo "[build_nbody] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Platform type detection ───────────────────────────────────────────────────
if [[ "${PLATFORM}" == amd_* ]]; then
    IS_AMD=true
    # HIP arch for AMD platforms
    if [[ -n "${HIP_ARCH:-}" ]]; then
        HIP_ARCH_VAL="${HIP_ARCH}"
    elif [[ "${PLATFORM}" == *"mi300x"* ]]; then
        HIP_ARCH_VAL="gfx942"
    elif [[ "${PLATFORM}" == *"mi250x"* ]]; then
        HIP_ARCH_VAL="gfx90a"
    elif [[ "${PLATFORM}" == *"mi100"* ]]; then
        HIP_ARCH_VAL="gfx908"
    else
        HIP_ARCH_VAL="gfx942"
    fi
    HIPCC_BIN="${ROCM_HOME:-/opt/rocm}/bin/hipcc"
    ARCH=""
else
    IS_AMD=false
    # CUDA architecture detection
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
    HIP_ARCH_VAL=""
fi

echo "[build_nbody] =============================================================="
echo "[build_nbody] Platform: ${PLATFORM}"
if [[ "${IS_AMD}" == "true" ]]; then
echo "[build_nbody] HIP arch:  ${HIP_ARCH_VAL}"
else
echo "[build_nbody] CUDA arch: ${ARCH}"
fi
echo "[build_nbody] Src dir:   ${SRC_DIR}"
echo "[build_nbody] Build base: ${BUILD_BASE}"
echo "[build_nbody] =============================================================="

# ── Kokkos detection ──────────────────────────────────────────────────────────
if [[ -z "${KOKKOS_INSTALL_PREFIX:-}" ]]; then
    for _p in \
        "/home/obalola/projects/kokkos-hip-install" \
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
    echo "[build_nbody] Kokkos found: ${KOKKOS_INSTALL_PREFIX}"
    _KOKKOS_OK=true
else
    echo "[build_nbody] WARNING: Kokkos not found — nbody-kokkos will be skipped."
    echo "[build_nbody]          Set KOKKOS_INSTALL_PREFIX= to override."
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
    echo "[build_nbody] RAJA found: ${RAJA_INSTALL_PREFIX}"
    _RAJA_OK=true
else
    echo "[build_nbody] WARNING: RAJA not found — nbody-raja will be skipped."
    echo "[build_nbody]          Set RAJA_INSTALL_PREFIX= to override."
    _RAJA_OK=false
fi

echo ""

# ── nbody-hip (AMD only) ──────────────────────────────────────────────────────
BUILD_HIP="${BUILD_BASE}/hip_${PLATFORM}"
if [[ "${IS_AMD}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_HIP}"
    mkdir -p "${BUILD_HIP}"

    if ! command -v "${HIPCC_BIN}" &>/dev/null; then
        echo "[build_nbody] WARNING: hipcc not found at ${HIPCC_BIN} — SKIP nbody-hip"
        _HIP_OK=false
    else
        echo "[build_nbody] Building nbody-hip (notile kernel, HIP/ROCm) ..."
        "${HIPCC_BIN}" -O3 --offload-arch="${HIP_ARCH_VAL}" \
            -DNBODY_USE_HIP \
            -I "${SRC_DIR}" \
            "${SRC_DIR}/nbody_hip.cpp" \
            -o "${BUILD_HIP}/nbody-hip"
        echo "[build_nbody]   → ${BUILD_HIP}/nbody-hip"
        _HIP_OK=true
    fi
fi

# ── nbody-native (NVIDIA only) ────────────────────────────────────────────────
BUILD_NATIVE="${BUILD_BASE}/native_${PLATFORM}"
if [[ "${IS_AMD}" == "false" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_NATIVE}"
    mkdir -p "${BUILD_NATIVE}"

    echo "[build_nbody] Building nbody-native (notile + tile kernels) ..."
    nvcc -O3 -arch="${ARCH}" \
        --use_fast_math --generate-line-info \
        --expt-extended-lambda --expt-relaxed-constexpr \
        -Xcompiler=-Wall,-Wextra \
        -I "${SRC_DIR}" \
        "${SRC_DIR}/nbody_native.cu" \
        -lcudart \
        -o "${BUILD_NATIVE}/nbody-native"
    echo "[build_nbody]   → ${BUILD_NATIVE}/nbody-native"
fi

# ── nbody-kokkos ─────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${IS_AMD}" == "true" ]]; then
    if [[ "${_KOKKOS_OK}" == "true" ]]; then
        [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
        mkdir -p "${BUILD_KOKKOS}"

        KOKKOS_LIB_DIR=""
        for _ld in "lib" "lib64"; do
            if [[ -f "${KOKKOS_INSTALL_PREFIX}/${_ld}/libkokkoscore.a" ]]; then
                KOKKOS_LIB_DIR="${KOKKOS_INSTALL_PREFIX}/${_ld}"
                break
            fi
        done

        if [[ -z "${KOKKOS_LIB_DIR}" ]]; then
            echo "[build_nbody] WARNING: libkokkoscore.a not found — SKIP nbody-kokkos"
            _KOKKOS_OK=false
        else
            echo "[build_nbody] Building nbody-kokkos (Kokkos HIP, two-step -std=c++20) ..."
            "${HIPCC_BIN}" -O3 --offload-arch="${HIP_ARCH_VAL}" -std=c++20 \
                -DKOKKOS_ENABLE_HIP -DNBODY_USE_HIP \
                -I "${KOKKOS_INSTALL_PREFIX}/include" -I "${SRC_DIR}" \
                -c "${SRC_DIR}/nbody_kokkos.cpp" \
                -o "${BUILD_KOKKOS}/nbody_kokkos.o"

            "${HIPCC_BIN}" -O3 --offload-arch="${HIP_ARCH_VAL}" \
                "${BUILD_KOKKOS}/nbody_kokkos.o" \
                "${KOKKOS_LIB_DIR}/libkokkoscore.a" \
                "${KOKKOS_LIB_DIR}/libkokkoscontainers.a" \
                -o "${BUILD_KOKKOS}/nbody-kokkos"
            echo "[build_nbody]   → ${BUILD_KOKKOS}/nbody-kokkos"
        fi
    else
        echo "[build_nbody] SKIP nbody-kokkos (Kokkos not found)"
    fi
elif [[ "${_KOKKOS_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_KOKKOS}"
    mkdir -p "${BUILD_KOKKOS}"

    echo "[build_nbody] Building nbody-kokkos (Kokkos::RangePolicy, two-step -std=c++20) ..."

    # Step 1: compile with -std=c++20 (mandatory for Kokkos C++20 features)
    nvcc -O3 -arch="${ARCH}" \
        --expt-extended-lambda --expt-relaxed-constexpr \
        --use_fast_math --generate-line-info \
        -std=c++20 \
        -x cu -c \
        -I "${KOKKOS_INSTALL_PREFIX}/include" \
        -I "${SRC_DIR}" \
        "${SRC_DIR}/nbody_kokkos.cpp" \
        -o "${BUILD_KOKKOS}/nbody_kokkos.o"

    # Step 2: nvcc link (NOT g++ — libkokkoscore.a needs -lcuda driver API)
    nvcc -O3 -arch="${ARCH}" -std=c++20 \
        "${BUILD_KOKKOS}/nbody_kokkos.o" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscore.a" \
        "${KOKKOS_INSTALL_PREFIX}/lib/libkokkoscontainers.a" \
        -lcudart -lcuda -ldl -lpthread \
        -o "${BUILD_KOKKOS}/nbody-kokkos"

    echo "[build_nbody]   → ${BUILD_KOKKOS}/nbody-kokkos"
else
    echo "[build_nbody] SKIP nbody-kokkos (Kokkos not found)"
fi

# ── nbody-raja ────────────────────────────────────────────────────────────────
BUILD_RAJA="${BUILD_BASE}/raja_${PLATFORM}"
if [[ "${IS_AMD}" == "true" ]]; then
    if [[ "${_RAJA_OK}" == "true" ]]; then
        [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_RAJA}"
        mkdir -p "${BUILD_RAJA}"

        RAJA_LIB_DIR=""
        for _ld in "lib" "lib64"; do
            if [[ -f "${RAJA_INSTALL_PREFIX}/${_ld}/libRAJA.a" ]]; then
                RAJA_LIB_DIR="${RAJA_INSTALL_PREFIX}/${_ld}"
                break
            fi
        done

        if [[ -z "${RAJA_LIB_DIR}" ]]; then
            echo "[build_nbody] WARNING: libRAJA.a not found — SKIP nbody-raja"
            _RAJA_OK=false
        else
            echo "[build_nbody] Building nbody-raja (RAJA HIP, two-step) ..."
            "${HIPCC_BIN}" -O3 --offload-arch="${HIP_ARCH_VAL}" \
                -DNBODY_USE_HIP \
                -I "${RAJA_INSTALL_PREFIX}/include" -I "${SRC_DIR}" \
                -c "${SRC_DIR}/nbody_raja.cpp" \
                -o "${BUILD_RAJA}/nbody_raja.o"

            "${HIPCC_BIN}" -O3 --offload-arch="${HIP_ARCH_VAL}" \
                "${BUILD_RAJA}/nbody_raja.o" \
                "${RAJA_LIB_DIR}/libRAJA.a" \
                "${RAJA_LIB_DIR}/libcamp.a" \
                -o "${BUILD_RAJA}/nbody-raja"
            echo "[build_nbody]   → ${BUILD_RAJA}/nbody-raja"
        fi
    else
        echo "[build_nbody] SKIP nbody-raja (RAJA not found)"
    fi
elif [[ "${_RAJA_OK}" == "true" ]]; then
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_RAJA}"
    mkdir -p "${BUILD_RAJA}"

    echo "[build_nbody] Building nbody-raja (RAJA::forall, two-step compile) ..."

    # Step 1: nvcc -x cu → device code object
    nvcc -O3 -arch="${ARCH}" \
        --use_fast_math --generate-line-info \
        --expt-extended-lambda --expt-relaxed-constexpr \
        -allow-unsupported-compiler \
        -Xcompiler=-Wall,-Wextra \
        -I "${RAJA_INSTALL_PREFIX}/include" \
        -I "${SRC_DIR}" \
        -x cu -dc \
        "${SRC_DIR}/nbody_raja.cpp" \
        -o "${BUILD_RAJA}/nbody_raja.o"

    # Step 2: nvcc link with libRAJA.a + libcamp.a
    nvcc -O3 -arch="${ARCH}" \
        "${BUILD_RAJA}/nbody_raja.o" \
        "${RAJA_INSTALL_PREFIX}/lib/libRAJA.a" \
        "${RAJA_INSTALL_PREFIX}/lib/libcamp.a" \
        -lcudart -lcuda -ldl -lpthread \
        -o "${BUILD_RAJA}/nbody-raja"

    echo "[build_nbody]   → ${BUILD_RAJA}/nbody-raja"
else
    echo "[build_nbody] SKIP nbody-raja (RAJA not found)"
fi

# ── nbody-sycl (AMD only) ─────────────────────────────────────────────────────
BUILD_SYCL="${BUILD_BASE}/sycl_${PLATFORM}"
_SYCL_OK=false
if [[ "${IS_AMD}" == "true" ]]; then
    # Detect acpp binary
    ACPP_BIN=""
    for _ab in \
        "${ROCM_HOME:-/opt/rocm}/../acpp/bin/acpp" \
        "/opt/acpp/bin/acpp" \
        "/usr/local/bin/acpp" \
        "$(command -v acpp 2>/dev/null || true)"; do
        if [[ -n "${_ab}" ]] && command -v "${_ab}" &>/dev/null; then
            ACPP_BIN="${_ab}"
            break
        fi
    done

    if [[ -z "${ACPP_BIN}" ]]; then
        echo "[build_nbody] WARNING: acpp not found — SKIP nbody-sycl"
    else
        [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_SYCL}"
        mkdir -p "${BUILD_SYCL}"
        echo "[build_nbody] Building nbody-sycl (AdaptiveCpp, target hip:${HIP_ARCH_VAL}) ..."
        "${ACPP_BIN}" --acpp-targets="hip:${HIP_ARCH_VAL}" -O3 \
            -DNBODY_USE_HIP \
            -I "${SRC_DIR}" \
            "${SRC_DIR}/nbody_sycl.cpp" \
            -o "${BUILD_SYCL}/nbody-sycl"
        echo "[build_nbody]   → ${BUILD_SYCL}/nbody-sycl"
        _SYCL_OK=true
    fi
else
    echo "[build_nbody] SKIP nbody-sycl (NVIDIA — SYCL not built for this platform)"
fi

# ── nbody-julia wrapper ───────────────────────────────────────────────────────
BUILD_JULIA="${BUILD_BASE}/julia_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_JULIA}"
mkdir -p "${BUILD_JULIA}"

JULIA_SRC="${SRC_DIR}/nbody_julia.jl"
JULIA_WRAP="${BUILD_JULIA}/nbody-julia"
if [[ "${IS_AMD}" == "true" ]]; then
cat > "${JULIA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated wrapper — do not edit. Regenerate with build_nbody.sh.
export JULIA_GPU_BACKEND="amdgpu"
exec julia --project=@. "${JULIA_SRC}" "\$@"
EOF
else
cat > "${JULIA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated wrapper — do not edit. Regenerate with build_nbody.sh.
exec julia --project=@. "${JULIA_SRC}" "\$@"
EOF
fi
chmod +x "${JULIA_WRAP}"
echo "[build_nbody]   → ${JULIA_WRAP} (Julia wrapper)"

echo ""
echo "[build_nbody] Build complete."
echo "[build_nbody] Binaries:"
[[ "${IS_AMD}" == "true" ]]  && [[ "${_HIP_OK:-false}" == "true" ]] && echo "[build_nbody]   hip:    ${BUILD_HIP}/nbody-hip"
[[ "${IS_AMD}" == "false" ]] && echo "[build_nbody]   native: ${BUILD_NATIVE}/nbody-native"
[[ "${_KOKKOS_OK}" == "true" ]] && echo "[build_nbody]   kokkos: ${BUILD_KOKKOS}/nbody-kokkos"
[[ "${_RAJA_OK}"   == "true" ]] && echo "[build_nbody]   raja:   ${BUILD_RAJA}/nbody-raja"
[[ "${_SYCL_OK}"   == "true" ]] && echo "[build_nbody]   sycl:   ${BUILD_SYCL}/nbody-sycl"
echo "[build_nbody]   julia:  ${JULIA_WRAP}"
