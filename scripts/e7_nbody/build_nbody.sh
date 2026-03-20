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
#   ./scripts/e7_nbody/build_nbody.sh [--platform nvidia_rtx5060_laptop] [--clean]
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

PLATFORM="nvidia_rtx5060_laptop"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --clean)    CLEAN=true; shift ;;
        *) echo "[build_nbody] Unknown argument: $1" >&2; exit 1 ;;
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

echo "[build_nbody] =============================================================="
echo "[build_nbody] Platform: ${PLATFORM}"
echo "[build_nbody] CUDA arch: ${ARCH}"
echo "[build_nbody] Src dir:   ${SRC_DIR}"
echo "[build_nbody] Build base: ${BUILD_BASE}"
echo "[build_nbody] =============================================================="

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

# ── nbody-native ──────────────────────────────────────────────────────────────
BUILD_NATIVE="${BUILD_BASE}/native_${PLATFORM}"
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

# ── nbody-kokkos ──────────────────────────────────────────────────────────────
BUILD_KOKKOS="${BUILD_BASE}/kokkos_${PLATFORM}"
if [[ "${_KOKKOS_OK}" == "true" ]]; then
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
if [[ "${_RAJA_OK}" == "true" ]]; then
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

# ── nbody-julia wrapper ───────────────────────────────────────────────────────
BUILD_JULIA="${BUILD_BASE}/julia_${PLATFORM}"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_JULIA}"
mkdir -p "${BUILD_JULIA}"

JULIA_SRC="${SRC_DIR}/nbody_julia.jl"
JULIA_WRAP="${BUILD_JULIA}/nbody-julia"
cat > "${JULIA_WRAP}" <<EOF
#!/usr/bin/env bash
# Auto-generated wrapper — do not edit. Regenerate with build_nbody.sh.
exec julia --project=@. "${JULIA_SRC}" "\$@"
EOF
chmod +x "${JULIA_WRAP}"
echo "[build_nbody]   → ${JULIA_WRAP} (Julia wrapper)"

echo ""
echo "[build_nbody] Build complete."
echo "[build_nbody] Binaries:"
echo "[build_nbody]   native: ${BUILD_NATIVE}/nbody-native"
[[ "${_KOKKOS_OK}" == "true" ]] && echo "[build_nbody]   kokkos: ${BUILD_KOKKOS}/nbody-kokkos"
[[ "${_RAJA_OK}"   == "true" ]] && echo "[build_nbody]   raja:   ${BUILD_RAJA}/nbody-raja"
echo "[build_nbody]   julia:  ${JULIA_WRAP}"
