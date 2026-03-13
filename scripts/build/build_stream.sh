#!/usr/bin/env bash
# Build E1 STREAM Triad abstraction variants from kernels/stream/.
#
# Usage:
#   ./scripts/build/build_stream.sh --platform nvidia_a100
#   ./scripts/build/build_stream.sh --platform nvidia_a100 --cuda-arch 120 \
#       --kokkos-root /path/to/kokkos/install \
#       --raja-dir    /path/to/raja/build
#   ./scripts/build/build_stream.sh --platform nvidia_a100 --abstraction native
#   ./scripts/build/build_stream.sh --platform nvidia_a100 --clean

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/kernels/stream"   # our own CMake project
BUILD_BASE="${REPO_ROOT}/build/stream"

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
    echo "       Expected our kernel source tree at kernels/stream/" >&2
    exit 1
fi

# ── Source platform environment ───────────────────────────────────────────────
VENDOR="${PLATFORM%%_*}"    # e.g. "nvidia" from "nvidia_a100"
env_script="${REPO_ROOT}/scripts/env/setup_${VENDOR}.sh"
if [[ -f "${env_script}" ]]; then
    source "${env_script}" 2>/dev/null || true
fi

# ── CUDA arch detection ───────────────────────────────────────────────────────
detect_cuda_arch() {
    # Reads compute capability from nvidia-smi, strips the dot.
    # "12.0" → "120",  "8.0" → "80",  "9.0" → "90"
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d ' ')"
    if [[ -z "${cap}" ]]; then
        echo ""
        return
    fi
    echo "${cap//./}"
}

if [[ -z "${CUDA_ARCH}" ]]; then
    CUDA_ARCH="$(detect_cuda_arch)"
    if [[ -z "${CUDA_ARCH}" ]]; then
        echo "WARNING: Could not detect CUDA compute capability from nvidia-smi." >&2
        echo "         Defaulting to sm_80 (A100/Ampere). Pass --cuda-arch N to override." >&2
        CUDA_ARCH="80"
    else
        echo "[build_stream] Detected CUDA arch: sm_${CUDA_ARCH}"
    fi
fi

# ── CUDA compiler selection ───────────────────────────────────────────────────
# Systems with both an apt 'nvidia-cuda-toolkit' package (/usr/bin/nvcc, often
# an older release) and a locally-installed CUDA toolkit (/usr/local/cuda/bin/nvcc)
# can confuse CMake.  We prefer the toolkit at /usr/local/cuda because it matches
# the driver version and supports the current GPU architecture.
if [[ -z "${CUDA_COMPILER}" ]]; then
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [[ -x "${candidate}" ]]; then
            CUDA_COMPILER="${candidate}"
            break
        fi
    done
    # Final fallback: whatever is on PATH
    [[ -z "${CUDA_COMPILER}" ]] && CUDA_COMPILER="$(command -v nvcc 2>/dev/null || echo nvcc)"
fi
echo "[build_stream] CUDA compiler: ${CUDA_COMPILER}"

# Map numeric CUDA arch to the Kokkos_ARCH flag (Kokkos ≥ 4.x naming).
kokkos_arch_flag() {
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

KOKKOS_ARCH="$(kokkos_arch_flag "${CUDA_ARCH}")"

# ── Library search ────────────────────────────────────────────────────────────
# If the user didn't pass --kokkos-root / --raja-dir, search well-known paths.
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

# ── Check Kokkos CUDA backend ─────────────────────────────────────────────────
# Warn if the located Kokkos was built without CUDA (SERIAL only).
check_kokkos_cuda() {
    local cfg="${KOKKOS_ROOT}/KokkosConfigCommon.cmake"
    [[ -z "${KOKKOS_ROOT}" ]] && return 1
    [[ -f "${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake" ]] && \
        cfg="${KOKKOS_ROOT}/lib/cmake/Kokkos/KokkosConfigCommon.cmake"
    if [[ ! -f "${cfg}" ]]; then return 1; fi
    if grep -q "set(Kokkos_DEVICES SERIAL)" "${cfg}" 2>/dev/null; then
        return 1   # SERIAL-only build — no CUDA
    fi
    return 0
}

# ── Builder ───────────────────────────────────────────────────────────────────
build_variant() {
    local abstraction="$1"; shift
    local extra_flags=("$@")
    local build_dir="${BUILD_BASE}/${abstraction}_${PLATFORM}"

    echo ""
    echo "[build_stream] ── ${abstraction} ────────────────────────────────────"
    [[ "${CLEAN}" == "true" ]] && rm -rf "${build_dir}"
    mkdir -p "${build_dir}"

    local cmake_args=(
        cmake -S "${SOURCE_DIR}" -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}"
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        "${extra_flags[@]}"
    )
    echo "  cmake: ${cmake_args[*]}"
    "${cmake_args[@]}" 2>&1 | tee "${build_dir}/cmake.log"
    cmake --build "${build_dir}" --parallel "$(nproc)" \
        2>&1 | tee "${build_dir}/build.log"
    echo "[build_stream] ${abstraction} → ${build_dir}"
}

# ── Abstraction build matrix ──────────────────────────────────────────────────
build_native() {
    build_variant "cuda" \
        -DSTREAM_ENABLE_HIP=OFF \
        -DSTREAM_ENABLE_SYCL=OFF \
        -DSTREAM_ENABLE_JULIA=OFF \
        -DSTREAM_ENABLE_NUMBA=OFF
    # The binary is stream-cuda; create the canonical "native" dir symlink
    local native_dir="${BUILD_BASE}/native_${PLATFORM}"
    if [[ ! -d "${native_dir}" ]]; then
        ln -sfn "${BUILD_BASE}/cuda_${PLATFORM}" "${native_dir}"
    fi
}

build_kokkos() {
    if [[ -z "${KOKKOS_ROOT}" ]]; then
        echo "  SKIP kokkos: Kokkos not found."
        echo "        Install Kokkos with CUDA backend and pass --kokkos-root <prefix>"
        return
    fi

    if ! check_kokkos_cuda; then
        echo "  SKIP kokkos: Found Kokkos at ${KOKKOS_ROOT} but it was built"
        echo "        without CUDA support (SERIAL backend only)."
        echo ""
        echo "        Rebuild Kokkos for CUDA:"
        echo "          cmake -S <kokkos-src> -B <build> \\"
        echo "                -DKokkos_ENABLE_CUDA=ON \\"
        echo "                -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \\"
        if [[ -n "${KOKKOS_ARCH}" ]]; then
            echo "                -DKokkos_ARCH_${KOKKOS_ARCH}=ON \\"
        fi
        echo "                -DCMAKE_CXX_COMPILER=$(which nvcc_wrapper 2>/dev/null || echo nvcc_wrapper)"
        echo "          cmake --install <build> --prefix <install-dir>"
        echo "        Then re-run: --kokkos-root <install-dir>"
        return
    fi

    local kokkos_flags=(-DKokkos_ROOT="${KOKKOS_ROOT}" -DKokkos_ENABLE_CUDA=ON)
    if [[ -n "${KOKKOS_ARCH}" ]]; then
        kokkos_flags+=("-DKokkos_ARCH_${KOKKOS_ARCH}=ON")
    fi
    build_variant "kokkos" "${kokkos_flags[@]}"
}

build_raja() {
    if [[ -z "${RAJA_DIR}" ]]; then
        echo "  SKIP raja: RAJA not found."
        echo "        Install RAJA and pass --raja-dir <prefix>/lib/cmake/raja"
        return
    fi
    build_variant "raja" \
        -DRAJA_DIR="${RAJA_DIR}" \
        -DRAJA_ENABLE_CUDA=ON \
        -DRAJA_CUDA_ARCH="sm_${CUDA_ARCH}"
}

build_sycl() {
    local sycl_compiler=""
    for c in icpx clang++ acpp; do
        if command -v "${c}" &>/dev/null; then
            sycl_compiler="${c}"
            break
        fi
    done
    if [[ -z "${sycl_compiler}" ]]; then
        echo "  SKIP sycl: no SYCL compiler found (icpx / clang++ / acpp)."
        echo "        Source the Intel oneAPI environment or install AdaptiveCpp."
        return
    fi
    local sycl_flags="-fsycl"
    if [[ "${VENDOR}" == "nvidia" ]]; then
        sycl_flags="${sycl_flags} -fsycl-targets=nvptx64-nvidia-cuda,spir64"
        sycl_flags="${sycl_flags} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_${CUDA_ARCH}"
    fi
    build_variant "sycl" \
        -DCMAKE_CXX_COMPILER="${sycl_compiler}" \
        -DSTREAM_ENABLE_SYCL=ON \
        "-DSYCL_FLAGS=${sycl_flags}"
}

build_julia() {
    if ! command -v julia &>/dev/null && [[ -z "${JULIA_BIN:-}" ]]; then
        echo "  SKIP julia: julia not found on PATH."
        echo "        Install Julia ≥ 1.9 or set JULIA_BIN."
        return
    fi
    local julia_bin="${JULIA_BIN:-julia}"
    local julia_ver
    julia_ver="$("${julia_bin}" --version 2>/dev/null | awk '{print $3}')"
    echo "  julia ${julia_ver} found at $(command -v "${julia_bin}")"

    build_variant "julia" -DSTREAM_ENABLE_JULIA=ON

    # Ensure the wrapper is executable (configure_file may not preserve +x)
    chmod +x "${BUILD_BASE}/julia_${PLATFORM}/stream-julia" 2>/dev/null || true

    # Instantiate the Julia project environment if CUDA.jl is not already cached
    local proj_dir="${BUILD_BASE}/julia_${PLATFORM}"
    if [[ ! -f "${proj_dir}/Manifest.toml" ]]; then
        echo "  Instantiating Julia project (downloads CUDA.jl — takes 1–5 min)..."
        "${julia_bin}" --project="${proj_dir}" --startup-file=no \
            -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee "${proj_dir}/julia_instantiate.log"
    fi
}

build_numba() {
    local py="${PYTHON_BIN:-python3}"
    if ! "${py}" -c "import numba, numpy" 2>/dev/null; then
        echo "  SKIP numba: numba or numpy not importable by ${py}."
        echo "        Install with: pip install numba numpy"
        return
    fi
    local numba_ver
    numba_ver="$("${py}" -c "import numba; print(numba.__version__)" 2>/dev/null)"
    echo "  numba ${numba_ver} found."
    build_variant "numba" -DSTREAM_ENABLE_NUMBA=ON
    chmod +x "${BUILD_BASE}/numba_${PLATFORM}/stream-numba" 2>/dev/null || true
}

# ── HIP (AMD only) ────────────────────────────────────────────────────────────
build_hip() {
    if ! command -v hipcc &>/dev/null; then
        echo "  SKIP hip: hipcc not found."
        return
    fi
    local hip_arch
    case "${PLATFORM}" in
        amd_mi250x)   hip_arch="gfx90a"  ;;
        amd_mi300x)   hip_arch="gfx942"  ;;
        amd_mi100)    hip_arch="gfx908"  ;;
        *)            hip_arch="${HIP_ARCH:-gfx90a}"
                      echo "  WARNING: unknown AMD platform '${PLATFORM}', using HIP_ARCH=${hip_arch}" ;;
    esac
    build_variant "hip" \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DSTREAM_ENABLE_HIP=ON \
        "-DHIP_ARCH=${hip_arch}"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo "[build_stream] ============================================================"
echo "[build_stream] Source:    ${SOURCE_DIR}"
echo "[build_stream] Build dir: ${BUILD_BASE}"
echo "[build_stream] Platform:  ${PLATFORM}"
echo "[build_stream] CUDA arch: sm_${CUDA_ARCH}"
echo "[build_stream] CUDA nvcc: ${CUDA_COMPILER}"
echo "[build_stream] Kokkos:    ${KOKKOS_ROOT:-not found}"
echo "[build_stream] RAJA:      ${RAJA_DIR:-not found}"
echo "[build_stream] ============================================================"

case "${PLATFORM}" in
    nvidia_*)
        case "${ABSTRACTION}" in
            all)
                build_native
                build_kokkos
                build_raja
                build_sycl
                build_julia
                build_numba
                ;;
            native|cuda) build_native  ;;
            kokkos)      build_kokkos  ;;
            raja)        build_raja    ;;
            sycl)        build_sycl    ;;
            julia)       build_julia   ;;
            numba)       build_numba   ;;
            *) echo "Unknown abstraction '${ABSTRACTION}'" >&2; exit 1 ;;
        esac
        ;;
    amd_*)
        case "${ABSTRACTION}" in
            all)
                build_hip
                build_kokkos
                build_raja
                build_sycl
                build_julia
                ;;
            native|hip) build_hip    ;;
            kokkos)     build_kokkos ;;
            raja)       build_raja   ;;
            sycl)       build_sycl   ;;
            julia)      build_julia  ;;
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
echo "[build_stream] ============================================================"
echo "[build_stream] Done.  Built binaries:"
find "${BUILD_BASE}" -maxdepth 2 -name "stream-*" -executable \
    ! -name "*.sh" 2>/dev/null \
    | sort | sed "s|${BUILD_BASE}/|  |"
echo "[build_stream] ============================================================"
echo "[build_stream] Run with:"
echo "  ./scripts/run/run_stream.sh --platform ${PLATFORM}"
