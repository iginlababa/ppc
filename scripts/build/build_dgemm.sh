#!/usr/bin/env bash
# Build E2 DGEMM abstraction variants from kernels/dgemm/.
#
# Usage:
#   ./scripts/build/build_dgemm.sh --platform nvidia_rtx5060_laptop
#   ./scripts/build/build_dgemm.sh --platform nvidia_rtx5060_laptop --cuda-arch 120 \
#       --kokkos-root /path/to/kokkos/install \
#       --raja-dir    /path/to/raja/build
#   ./scripts/build/build_dgemm.sh --platform amd_mi300x --abstraction native
#   ./scripts/build/build_dgemm.sh --platform nvidia_rtx5060_laptop --clean

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/kernels/dgemm"
BUILD_BASE="${REPO_ROOT}/build/dgemm"

# ── Defaults ──────────────────────────────────────────────────────────────────
PLATFORM="nvidia_rtx5060_laptop"
ABSTRACTION="all"
CUDA_ARCH=""          # auto-detect if empty
KOKKOS_ROOT=""        # search if empty
RAJA_DIR=""           # search if empty
CUDA_COMPILER=""      # auto-detect if empty
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2";    shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --cuda-arch)   CUDA_ARCH="$2";   shift 2 ;;
        --kokkos-root) KOKKOS_ROOT="$2"; shift 2 ;;
        --raja-dir)    RAJA_DIR="$2";    shift 2 ;;
        --cuda-compiler) CUDA_COMPILER="$2"; shift 2 ;;
        --clean)       CLEAN=true;       shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Verify source ─────────────────────────────────────────────────────────────
if [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ]]; then
    echo "ERROR: ${SOURCE_DIR}/CMakeLists.txt not found." >&2
    exit 1
fi

# ── Source platform environment ───────────────────────────────────────────────
VENDOR="${PLATFORM%%_*}"    # e.g. "nvidia" from "nvidia_a100"
env_script="${REPO_ROOT}/scripts/env/setup_${VENDOR}.sh"
if [[ -f "${env_script}" ]]; then
    _saved_platform="${PLATFORM}"
    source "${env_script}" 2>/dev/null || true
    PLATFORM="${_saved_platform}"
fi

# ── CUDA arch detection ───────────────────────────────────────────────────────
detect_cuda_arch() {
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d ' ')"
    [[ -z "${cap}" ]] && { echo ""; return; }
    echo "${cap//./}"
}

if [[ "${VENDOR}" != "amd" ]]; then
    if [[ -z "${CUDA_ARCH}" ]]; then
        CUDA_ARCH="$(detect_cuda_arch)"
        if [[ -z "${CUDA_ARCH}" ]]; then
            echo "WARNING: Could not detect CUDA compute capability from nvidia-smi." >&2
            echo "         Defaulting to sm_80 (A100/Ampere). Pass --cuda-arch N to override." >&2
            CUDA_ARCH="80"
        else
            echo "[build_dgemm] Detected CUDA arch: sm_${CUDA_ARCH}"
        fi
    fi

    # ── CUDA compiler selection ───────────────────────────────────────────────
    if [[ -z "${CUDA_COMPILER}" ]]; then
        for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
            if [[ -x "${candidate}" ]]; then
                CUDA_COMPILER="${candidate}"
                break
            fi
        done
        [[ -z "${CUDA_COMPILER}" ]] && CUDA_COMPILER="$(command -v nvcc 2>/dev/null || echo nvcc)"
    fi
    echo "[build_dgemm] CUDA compiler: ${CUDA_COMPILER}"
fi

# Map numeric CUDA arch to the Kokkos_ARCH flag (Kokkos ≥ 4.x naming).
kokkos_cuda_arch_flag() {
    case "$1" in
        70)  echo "VOLTA70"      ;;
        72)  echo "VOLTA72"      ;;
        75)  echo "TURING75"     ;;
        80)  echo "AMPERE80"     ;;
        86)  echo "AMPERE86"     ;;
        89)  echo "ADA89"        ;;
        90)  echo "HOPPER90"     ;;
        100) echo "BLACKWELL100" ;;
        120) echo "BLACKWELL120" ;;
        *)   echo ""             ;;
    esac
}

KOKKOS_ARCH="$(kokkos_cuda_arch_flag "${CUDA_ARCH:-}")"

# ── Library search ────────────────────────────────────────────────────────────
find_kokkos_root() {
    local candidates=(
        "${KOKKOS_ROOT}"
        "/usr/local"
        "/opt/kokkos"
        "/home/obalola/projects/kokkos-quick-start/builddir/cmake_packages/Kokkos"
        "/home/obalola/projects/kokkos-install"
    )
    for p in "${candidates[@]}"; do
        [[ -z "${p}" ]] && continue
        if [[ -f "${p}/KokkosConfig.cmake" ]] || \
           [[ -f "${p}/lib/cmake/Kokkos/KokkosConfig.cmake" ]]; then
            echo "${p}"
            return 0
        fi
    done
    echo ""
}

find_raja_dir() {
    local candidates=(
        "${RAJA_DIR}"
        "/usr/local/lib/cmake/raja"
        "/opt/raja/lib/cmake/raja"
        "/home/obalola/projects/raja/build"
        "/home/obalola/projects/raja/install/lib/cmake/raja"
    )
    for p in "${candidates[@]}"; do
        [[ -z "${p}" ]] && continue
        if [[ -f "${p}/raja-config.cmake" ]] || \
           [[ -f "${p}/RAJAConfig.cmake" ]]; then
            echo "${p}"
            return 0
        fi
    done
    echo ""
}

[[ -z "${KOKKOS_ROOT}" ]] && KOKKOS_ROOT="$(find_kokkos_root)"
[[ -z "${RAJA_DIR}"    ]] && RAJA_DIR="$(find_raja_dir)"

# ── Kokkos backend checks ─────────────────────────────────────────────────────
check_kokkos_cuda() {
    local cfg="${KOKKOS_ROOT}/KokkosConfigCommon.cmake"
    [[ -z "${KOKKOS_ROOT}" ]] && return 1
    [[ -f "${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake" ]] && \
        cfg="${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake"
    [[ ! -f "${cfg}" ]] && return 1
    grep -q "set(Kokkos_DEVICES SERIAL)" "${cfg}" 2>/dev/null && return 1
    return 0
}

check_kokkos_hip() {
    local cfg="${KOKKOS_ROOT}/KokkosConfigCommon.cmake"
    [[ -z "${KOKKOS_ROOT}" ]] && return 1
    [[ -f "${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake" ]] && \
        cfg="${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake"
    [[ ! -f "${cfg}" ]] && return 1
    grep -q "HIP" "${cfg}" 2>/dev/null && return 0
    return 1
}

# ── Builder ───────────────────────────────────────────────────────────────────
build_variant() {
    local abstraction="$1"; shift
    local extra_flags=("$@")
    local build_dir="${BUILD_BASE}/${abstraction}_${PLATFORM}"

    echo ""
    echo "[build_dgemm] ── ${abstraction} ─────────────────────────────────────"
    [[ "${CLEAN}" == "true" ]] && rm -rf "${build_dir}"
    mkdir -p "${build_dir}"

    local cmake_args=(cmake -S "${SOURCE_DIR}" -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        "${extra_flags[@]}")

    if [[ "${VENDOR}" != "amd" ]]; then
        cmake_args+=(
            -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)"
            -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}"
            -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        )
    fi

    echo "  cmake: ${cmake_args[*]}"
    "${cmake_args[@]}" 2>&1 | tee "${build_dir}/cmake.log"
    cmake --build "${build_dir}" --parallel "$(nproc)" \
        2>&1 | tee "${build_dir}/build.log"
    echo "[build_dgemm] ${abstraction} → ${build_dir}"
}

# ── Abstraction build functions ───────────────────────────────────────────────
build_native_cuda() {
    build_variant "cuda" \
        -DDGEMM_ENABLE_HIP=OFF \
        -DDGEMM_ENABLE_SYCL=OFF \
        -DDGEMM_ENABLE_JULIA=OFF \
        -DDGEMM_ENABLE_NUMBA=OFF
    local native_dir="${BUILD_BASE}/native_${PLATFORM}"
    if [[ ! -d "${native_dir}" ]]; then
        ln -sfn "${BUILD_BASE}/cuda_${PLATFORM}" "${native_dir}"
    fi
}

build_native_hip() {
    if ! command -v hipcc &>/dev/null; then
        echo "  SKIP hip: hipcc not found."
        return
    fi
    local hip_arch
    case "${PLATFORM}" in
        amd_mi300x) hip_arch="gfx942"  ;;
        amd_mi250x) hip_arch="gfx90a"  ;;
        amd_mi100)  hip_arch="gfx908"  ;;
        *)          hip_arch="${HIP_ARCH:-gfx90a}"
                    echo "  WARNING: unknown AMD platform '${PLATFORM}', using hip_arch=${hip_arch}" ;;
    esac
    build_variant "hip" \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DDGEMM_ENABLE_HIP=ON \
        -DDGEMM_ENABLE_SYCL=OFF \
        -DDGEMM_ENABLE_JULIA=OFF \
        -DDGEMM_ENABLE_NUMBA=OFF \
        "-DHIP_ARCH=${hip_arch}"
    local native_dir="${BUILD_BASE}/native_${PLATFORM}"
    if [[ ! -d "${native_dir}" ]]; then
        ln -sfn "${BUILD_BASE}/hip_${PLATFORM}" "${native_dir}"
    fi
}

build_kokkos() {
    if [[ -z "${KOKKOS_ROOT}" ]]; then
        echo "  SKIP kokkos: Kokkos not found. Pass --kokkos-root <prefix>"
        return
    fi

    if [[ "${VENDOR}" == "amd" ]]; then
        if ! check_kokkos_hip; then
            echo "  SKIP kokkos: Kokkos at ${KOKKOS_ROOT} not built with HIP backend."
            echo "        Rebuild Kokkos with -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942=ON"
            return
        fi
        build_variant "kokkos" \
            -DKokkos_ROOT="${KOKKOS_ROOT}" \
            -DCMAKE_CXX_COMPILER=hipcc \
            -DDGEMM_ENABLE_HIP=ON
    else
        if ! check_kokkos_cuda; then
            echo "  SKIP kokkos: Kokkos at ${KOKKOS_ROOT} not built with CUDA backend."
            return
        fi
        local _cuda_root=/usr/local/cuda
        [[ "${CUDA_COMPILER}" == *"/bin/"* ]] && _cuda_root="${CUDA_COMPILER%/bin/*}"
        local kokkos_flags=(-DKokkos_ROOT="${KOKKOS_ROOT}" -DKokkos_ENABLE_CUDA=ON
                            "-DCUDAToolkit_ROOT=${_cuda_root}")
        [[ -n "${KOKKOS_ARCH}" ]] && kokkos_flags+=("-DKokkos_ARCH_${KOKKOS_ARCH}=ON")
        build_variant "kokkos" "${kokkos_flags[@]}"
    fi
}

build_raja() {
    if [[ -z "${RAJA_DIR}" ]]; then
        echo "  SKIP raja: RAJA not found. Pass --raja-dir <prefix>/lib/cmake/raja"
        return
    fi
    if [[ "${VENDOR}" == "amd" ]]; then
        build_variant "raja" \
            -DRAJA_DIR="${RAJA_DIR}" \
            -DCMAKE_CXX_COMPILER=hipcc \
            -DDGEMM_ENABLE_HIP=ON
    else
        build_variant "raja" \
            -DRAJA_DIR="${RAJA_DIR}" \
            -DRAJA_ENABLE_CUDA=ON \
            -DRAJA_CUDA_ARCH="sm_${CUDA_ARCH}"
    fi
}

build_sycl() {
    local sycl_compiler=""
    if [[ "${VENDOR}" == "amd" ]]; then
        if ! command -v acpp &>/dev/null; then
            echo "  SKIP sycl: acpp not found. Install AdaptiveCpp."
            return
        fi
        sycl_compiler="acpp"
    else
        for c in icpx clang++ acpp; do
            if command -v "${c}" &>/dev/null; then
                sycl_compiler="${c}"
                break
            fi
        done
    fi
    if [[ -z "${sycl_compiler}" ]]; then
        echo "  SKIP sycl: no SYCL compiler found (icpx / clang++ / acpp)."
        return
    fi

    local sycl_flags="-fsycl"
    local sycl_extra_cmake=()
    if [[ "${VENDOR}" == "nvidia" ]]; then
        sycl_flags="${sycl_flags} -fsycl-targets=nvptx64-nvidia-cuda,spir64"
        sycl_flags="${sycl_flags} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_${CUDA_ARCH}"
    elif [[ "${VENDOR}" == "amd" ]]; then
        local hip_arch
        case "${PLATFORM}" in
            amd_mi300x) hip_arch="gfx942" ;;
            amd_mi250x) hip_arch="gfx90a" ;;
            amd_mi100)  hip_arch="gfx908" ;;
            *)          hip_arch="${HIP_ARCH:-gfx90a}"
                        echo "  WARNING: unknown AMD platform '${PLATFORM}', using hip_arch=${hip_arch}" ;;
        esac
        sycl_flags="--acpp-targets=hip:${hip_arch}"
        sycl_extra_cmake+=(-DDGEMM_ENABLE_HIP=OFF)
        sycl_extra_cmake+=(-DCMAKE_DISABLE_FIND_PACKAGE_RAJA=ON)
        sycl_extra_cmake+=(-DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON)
    fi

    build_variant "sycl" \
        -DCMAKE_CXX_COMPILER="${sycl_compiler}" \
        -DDGEMM_ENABLE_SYCL=ON \
        "${sycl_extra_cmake[@]}" \
        "-DSYCL_FLAGS=${sycl_flags}"
}

build_julia() {
    local julia_bin="${JULIA_BIN:-julia}"
    if ! command -v "${julia_bin}" &>/dev/null; then
        echo "  SKIP julia: julia not found on PATH."
        echo "        Install Julia ≥ 1.9 or set JULIA_BIN."
        return
    fi
    local julia_ver
    julia_ver="$("${julia_bin}" --version 2>/dev/null | awk '{print $3}')"
    echo "  julia ${julia_ver} found at $(command -v "${julia_bin}")"

    # Julia uses its own GPU runtime — disable C++ GPU abstractions entirely
    # to avoid picking up Kokkos/RAJA installed with HIP flags on AMD machines.
    build_variant "julia" -DDGEMM_ENABLE_JULIA=ON \
        -DDGEMM_ENABLE_HIP=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_RAJA=ON
    chmod +x "${BUILD_BASE}/julia_${PLATFORM}/dgemm-julia" 2>/dev/null || true

    # Instantiate Julia project — downloads CUDA.jl and/or AMDGPU.jl
    if [[ ! -f "${SOURCE_DIR}/Manifest.toml" ]]; then
        echo "  Instantiating Julia project (downloads GPU packages — takes 1–5 min)..."
        "${julia_bin}" --project="${SOURCE_DIR}" --startup-file=no \
            -e 'using Pkg; Pkg.instantiate()' \
            2>&1 | tee "${BUILD_BASE}/julia_${PLATFORM}/julia_instantiate.log"
    fi
}

build_numba() {
    local py="${PYTHON_BIN:-python3}"
    if ! "${py}" -c "import numba, numpy" 2>/dev/null; then
        echo "  SKIP numba: numba or numpy not importable by ${py}."
        return
    fi
    local numba_ver
    numba_ver="$("${py}" -c "import numba; print(numba.__version__)" 2>/dev/null)"
    echo "  numba ${numba_ver} found."
    build_variant "numba" -DDGEMM_ENABLE_NUMBA=ON
    chmod +x "${BUILD_BASE}/numba_${PLATFORM}/dgemm-numba" 2>/dev/null || true
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo "[build_dgemm] ============================================================"
echo "[build_dgemm] Source:    ${SOURCE_DIR}"
echo "[build_dgemm] Build dir: ${BUILD_BASE}"
echo "[build_dgemm] Platform:  ${PLATFORM}"
if [[ "${VENDOR}" != "amd" ]]; then
    echo "[build_dgemm] CUDA arch: sm_${CUDA_ARCH}"
    echo "[build_dgemm] CUDA nvcc: ${CUDA_COMPILER}"
fi
echo "[build_dgemm] Kokkos:    ${KOKKOS_ROOT:-not found}"
echo "[build_dgemm] RAJA:      ${RAJA_DIR:-not found}"
echo "[build_dgemm] ============================================================"

case "${PLATFORM}" in
    nvidia_*)
        case "${ABSTRACTION}" in
            all)
                build_native_cuda
                build_kokkos
                build_raja
                build_sycl
                build_julia
                build_numba
                ;;
            native|cuda)  build_native_cuda ;;
            kokkos)       build_kokkos      ;;
            raja)         build_raja        ;;
            sycl)         build_sycl        ;;
            julia)        build_julia       ;;
            numba)        build_numba       ;;
            *) echo "Unknown abstraction '${ABSTRACTION}'" >&2; exit 1 ;;
        esac
        ;;
    amd_*)
        case "${ABSTRACTION}" in
            all)
                build_native_hip
                build_kokkos
                build_raja
                build_julia
                ;;
            native|hip)   build_native_hip ;;
            kokkos)       build_kokkos     ;;
            raja)         build_raja       ;;
            sycl)         build_sycl       ;;
            julia)        build_julia      ;;
            *) echo "Unknown abstraction '${ABSTRACTION}'" >&2; exit 1 ;;
        esac
        ;;
    intel_*)
        case "${ABSTRACTION}" in
            all)
                build_sycl
                build_kokkos
                build_raja
                ;;
            native|sycl) build_sycl    ;;
            kokkos)      build_kokkos  ;;
            raja)        build_raja    ;;
            *) echo "Unknown abstraction '${ABSTRACTION}'" >&2; exit 1 ;;
        esac
        ;;
    *)
        echo "ERROR: Unknown platform '${PLATFORM}'" >&2
        echo "       Supported: nvidia_*, amd_*, intel_*" >&2
        exit 1
        ;;
esac

echo ""
echo "[build_dgemm] ============================================================"
echo "[build_dgemm] Done.  Built binaries:"
find "${BUILD_BASE}" -maxdepth 2 -name "dgemm-*" -executable \
    ! -name "*.sh" 2>/dev/null \
    | sort | sed "s|${BUILD_BASE}/|  |"
echo "[build_dgemm] ============================================================"
echo "[build_dgemm] Run with:"
echo "  ./scripts/run/run_dgemm.sh --platform ${PLATFORM}"
