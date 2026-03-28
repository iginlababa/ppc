#!/usr/bin/env bash
# Build all abstraction variants of BFS (E6).
# Usage: ./scripts/build/build_bfs.sh [--platform nvidia_rtx5060] [--verify] [--clean]
#
# NVIDIA: bfs-native (CUDA+Thrust), bfs-kokkos, bfs-raja, bfs-julia → bin/bfs/
# AMD:    bfs-hip (HIP+ROCThrust), bfs-kokkos, bfs-raja, bfs-sycl, bfs-julia → build/bfs/*_amd_mi300x/
# Skipped: numba (UNSUPPORTED_CC120 on NVIDIA; numba-hip experimental on AMD)
#          sycl  (NO_COMPILER on RTX 5060)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernels/bfs"

PLATFORM="nvidia_rtx5060"
VERIFY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --verify)   VERIFY=true; shift ;;
        --clean)    CLEAN=true; shift ;;
        *) echo "[build_bfs] Unknown argument: $1"; exit 1 ;;
    esac
done

VENDOR="${PLATFORM%%_*}"   # "nvidia" or "amd"

# ── Kokkos / RAJA path detection ───────────────────────────────────────────────
KOKKOS_ROOT="${KOKKOS_INSTALL_PREFIX:-}"
if [[ -z "${KOKKOS_ROOT}" ]]; then
    for p in /home/obalola/projects/kokkos-cuda-install \
              /usr/local/kokkos /opt/kokkos /usr/kokkos "${HOME}/kokkos" \
              /usr/local /opt/local; do
        [[ -f "${p}/lib/libkokkoscore.a" || -f "${p}/lib64/libkokkoscore.a" ]] && \
            { KOKKOS_ROOT="${p}"; break; }
    done
fi
KOKKOS_FOUND=false
[[ -n "${KOKKOS_ROOT}" ]] && KOKKOS_FOUND=true

RAJA_ROOT="${RAJA_INSTALL_PREFIX:-}"
if [[ -z "${RAJA_ROOT}" ]]; then
    for p in /home/obalola/projects/raja/install \
              /usr/local/raja /opt/raja /usr/raja "${HOME}/raja" \
              /usr/local /opt/local; do
        [[ -f "${p}/lib/libRAJA.a" || -f "${p}/lib64/libRAJA.a" ]] && \
            { RAJA_ROOT="${p}"; break; }
    done
fi
RAJA_FOUND=false
[[ -n "${RAJA_ROOT}" ]] && RAJA_FOUND=true

CAMP_LIB=""
for p in "${RAJA_ROOT}" /usr/local/camp /opt/camp "${HOME}/camp" /usr/local /opt/local; do
    [[ -f "${p}/lib/libcamp.a" || -f "${p}/lib64/libcamp.a" ]] && \
        { CAMP_LIB="${p}"; break; }
done

# ── Helper: run --verify correctness check ─────────────────────────────────────
run_verify() {
    local bin="$1"
    local abs_name="$2"
    echo "[build_bfs] Verifying ${abs_name} ..."
    "${bin}" --verify --graph erdos_renyi --n 1024 --reps 0 --platform "${PLATFORM}" && \
    "${bin}" --verify --graph 2d_grid     --n 1024 --reps 0 --platform "${PLATFORM}" && \
    echo "[build_bfs] ${abs_name} correctness OK" || \
        { echo "[build_bfs] ${abs_name} CORRECTNESS FAILED"; return 1; }
}

# ══════════════════════════════════════════════════════════════════════════════
# AMD MI300X build block
# ══════════════════════════════════════════════════════════════════════════════
if [[ "${VENDOR}" == "amd" ]]; then
    # ── AMD GFX target ──────────────────────────────────────────────────────
    case "${PLATFORM}" in
        *mi300x*) HIP_ARCH="gfx942" ;;
        *mi250x*) HIP_ARCH="gfx90a" ;;
        *mi100*)  HIP_ARCH="gfx908" ;;
        *)        HIP_ARCH="gfx942" ;;
    esac

    BUILD_BASE="${REPO_ROOT}/build/bfs"
    [[ "${CLEAN}" == "true" ]] && rm -rf "${BUILD_BASE}"
    mkdir -p "${BUILD_BASE}"

    ROCM_ROOT="${ROCM_PATH:-/opt/rocm}"
    HIP_FLAGS="-O3 -ffast-math --offload-arch=${HIP_ARCH} -I${KERNEL_DIR} -std=c++17"

    echo "[build_bfs] =============================================================="
    echo "[build_bfs] Platform:    ${PLATFORM}"
    echo "[build_bfs] HIP arch:    ${HIP_ARCH}"
    echo "[build_bfs] Kernel dir:  ${KERNEL_DIR}"
    echo "[build_bfs] Build base:  ${BUILD_BASE}"
    echo "[build_bfs] Skipped:     numba (AMD: numba-hip experimental)"
    echo "[build_bfs] =============================================================="

    CAMP_LIBS=""
    [[ -n "${CAMP_LIB}" ]] && CAMP_LIBS="-L${CAMP_LIB}/lib -lcamp"

    # ── 1. bfs-hip (native HIP + ROCThrust) ─────────────────────────────────
    echo "[build_bfs] Building bfs-hip (native HIP + ROCThrust level-set BFS) ..."
    OUT_HIP="${BUILD_BASE}/hip_${PLATFORM}"
    mkdir -p "${OUT_HIP}"
    hipcc ${HIP_FLAGS} \
        -D__HIP_PLATFORM_AMD__ \
        "${KERNEL_DIR}/kernel_bfs_hip.cpp" \
        -o "${OUT_HIP}/bfs-hip" \
        -lamdhip64
    echo "[build_bfs]   → ${OUT_HIP}/bfs-hip"
    [[ "${VERIFY}" == "true" ]] && run_verify "${OUT_HIP}/bfs-hip" "native-hip"

    # ── 2. bfs-kokkos (HIP backend, two-step) ────────────────────────────────
    if [[ "${KOKKOS_FOUND}" == "true" ]]; then
        echo "[build_bfs] Building bfs-kokkos (Kokkos HIP backend, two-step) ..."
        OUT_KK="${BUILD_BASE}/kokkos_${PLATFORM}"
        mkdir -p "${OUT_KK}"
        KOKKOS_INC="${KOKKOS_ROOT}/include"
        KOKKOS_LIB_DIR="$(for d in "${KOKKOS_ROOT}/lib" "${KOKKOS_ROOT}/lib64"; do [[ -d "$d" ]] && echo "$d" && break; done)"
        hipcc ${HIP_FLAGS} \
            -I"${KOKKOS_INC}" \
            -DKOKKOS_ENABLE_HIP=1 \
            -c "${KERNEL_DIR}/kernel_bfs_kokkos.cpp" \
            -o "${OUT_KK}/kernel_bfs_kokkos.o"
        hipcc ${HIP_FLAGS} \
            "${OUT_KK}/kernel_bfs_kokkos.o" \
            -o "${OUT_KK}/bfs-kokkos" \
            -L"${KOKKOS_LIB_DIR}" -lkokkoscore -lkokkoscontainers \
            -lamdhip64
        echo "[build_bfs]   → ${OUT_KK}/bfs-kokkos"
        [[ "${VERIFY}" == "true" ]] && run_verify "${OUT_KK}/bfs-kokkos" "kokkos"
    else
        echo "[build_bfs] SKIP bfs-kokkos: libkokkoscore.a not found"
    fi

    # ── 3. bfs-raja (HIP backend, two-step) ──────────────────────────────────
    if [[ "${RAJA_FOUND}" == "true" ]]; then
        echo "[build_bfs] Building bfs-raja (RAJA HIP backend, two-step) ..."
        OUT_RAJA="${BUILD_BASE}/raja_${PLATFORM}"
        mkdir -p "${OUT_RAJA}"
        RAJA_INC="${RAJA_ROOT}/include"
        RAJA_LIB_DIR="$(for d in "${RAJA_ROOT}/lib" "${RAJA_ROOT}/lib64"; do [[ -d "$d" ]] && echo "$d" && break; done)"
        hipcc ${HIP_FLAGS} \
            -I"${RAJA_INC}" \
            -D__HIP_PLATFORM_AMD__ \
            -c "${KERNEL_DIR}/kernel_bfs_raja.cpp" \
            -o "${OUT_RAJA}/kernel_bfs_raja.o"
        hipcc ${HIP_FLAGS} \
            "${OUT_RAJA}/kernel_bfs_raja.o" \
            -o "${OUT_RAJA}/bfs-raja" \
            -L"${RAJA_LIB_DIR}" -lRAJA \
            ${CAMP_LIBS} \
            -lamdhip64
        echo "[build_bfs]   → ${OUT_RAJA}/bfs-raja"
        [[ "${VERIFY}" == "true" ]] && run_verify "${OUT_RAJA}/bfs-raja" "raja"
    else
        echo "[build_bfs] SKIP bfs-raja: libRAJA.a not found"
    fi

    # ── 4. bfs-sycl (AdaptiveCpp, acpp --acpp-targets=hip:gfx942) ────────────
    SYCL_COMPILER=""
    for c in acpp syclcc; do
        command -v "${c}" &>/dev/null && { SYCL_COMPILER="${c}"; break; }
    done
    if [[ -n "${SYCL_COMPILER}" ]]; then
        echo "[build_bfs] Building bfs-sycl (${SYCL_COMPILER}, acpp HIP target) ..."
        OUT_SYCL="${BUILD_BASE}/sycl_${PLATFORM}"
        mkdir -p "${OUT_SYCL}"
        "${SYCL_COMPILER}" \
            --acpp-targets="hip:${HIP_ARCH}" \
            -O3 -std=c++17 \
            -I"${KERNEL_DIR}" \
            "${KERNEL_DIR}/kernel_bfs_sycl.cpp" \
            -o "${OUT_SYCL}/bfs-sycl"
        echo "[build_bfs]   → ${OUT_SYCL}/bfs-sycl"
        [[ "${VERIFY}" == "true" ]] && run_verify "${OUT_SYCL}/bfs-sycl" "sycl"
    else
        echo "[build_bfs] SKIP bfs-sycl: no SYCL compiler found (acpp/syclcc)"
    fi

    # ── 5. bfs-julia (wrapper script) ────────────────────────────────────────
    OUT_JULIA="${BUILD_BASE}/julia_${PLATFORM}"
    mkdir -p "${OUT_JULIA}"
    cat > "${OUT_JULIA}/bfs-julia" <<'JULIA_WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec julia --project="${REPO_ROOT}" \
    "${REPO_ROOT}/kernels/bfs/kernel_bfs_julia.jl" "$@"
JULIA_WRAPPER
    chmod +x "${OUT_JULIA}/bfs-julia"
    echo "[build_bfs]   → ${OUT_JULIA}/bfs-julia (Julia wrapper; use run_bfs.sh --platform amd_mi300x)"

    echo ""
    echo "[build_bfs] =============================================================="
    echo "[build_bfs] Build summary:"
    [[ -x "${OUT_HIP}/bfs-hip" ]]           && echo "  [OK] bfs-hip    → ${OUT_HIP}/bfs-hip"
    [[ -x "${OUT_KK:-}/bfs-kokkos" ]]       && echo "  [OK] bfs-kokkos → ${OUT_KK}/bfs-kokkos"
    [[ -x "${OUT_RAJA:-}/bfs-raja" ]]       && echo "  [OK] bfs-raja   → ${OUT_RAJA}/bfs-raja"
    [[ -x "${OUT_SYCL:-}/bfs-sycl" ]]      && echo "  [OK] bfs-sycl   → ${OUT_SYCL}/bfs-sycl"
    [[ -x "${OUT_JULIA}/bfs-julia" ]]       && echo "  [OK] bfs-julia  → ${OUT_JULIA}/bfs-julia"
    echo "  [--] bfs-numba  SKIP (AMD: numba-hip experimental)"
    echo "[build_bfs] Next step: ./scripts/run/run_bfs.sh --platform ${PLATFORM}"
    echo "[build_bfs] =============================================================="
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA build block (original)
# ══════════════════════════════════════════════════════════════════════════════
BIN_DIR="${REPO_ROOT}/bin/bfs"
[[ "${CLEAN}" == "true" ]] && rm -rf "${BIN_DIR}"
mkdir -p "${BIN_DIR}"

# ── CUDA arch detection ────────────────────────────────────────────────────────
case "${PLATFORM}" in
    *rtx5060*|*blackwell*) CUDA_ARCH="sm_120" ;;
    *h100*|*hopper*)       CUDA_ARCH="sm_90"  ;;
    *a100*|*ampere*)       CUDA_ARCH="sm_80"  ;;
    *v100*|*volta*)        CUDA_ARCH="sm_70"  ;;
    *rtx30*|*ga10*)        CUDA_ARCH="sm_86"  ;;
    *rtx40*|*ada*)         CUDA_ARCH="sm_89"  ;;
    *)                     CUDA_ARCH="sm_80"  ;;
esac
echo "[build_bfs] Platform=${PLATFORM}  CUDA arch=${CUDA_ARCH}"

NVCC_FLAGS="-O3 -arch=${CUDA_ARCH} --extended-lambda -Xcompiler -O3,-march=native"
NVCC_FLAGS+=" -I${KERNEL_DIR}"
HOST_FLAGS="-O3 -march=native -std=c++20"
CUDA_LIBS="-lcuda -lcudart"

# ── 1. Native CUDA + Thrust ───────────────────────────────────────────────────
echo "[build_bfs] Building native CUDA ..."
nvcc ${NVCC_FLAGS} \
    "${KERNEL_DIR}/kernel_bfs_cuda.cu" \
    -o "${BIN_DIR}/bfs-native" \
    ${CUDA_LIBS}
echo "[build_bfs] native CUDA: OK → ${BIN_DIR}/bfs-native"
[[ "${VERIFY}" == "true" ]] && run_verify "${BIN_DIR}/bfs-native" "native"

# ── 2. Kokkos ─────────────────────────────────────────────────────────────────
if [[ "${KOKKOS_FOUND}" == "true" ]]; then
    echo "[build_bfs] Building Kokkos (root=${KOKKOS_ROOT}) ..."
    KOKKOS_INC="${KOKKOS_ROOT}/include"
    KOKKOS_LIB_DIR=$(ls -d "${KOKKOS_ROOT}/lib" "${KOKKOS_ROOT}/lib64" 2>/dev/null | head -1)
    KOKKOS_LIBS="-L${KOKKOS_LIB_DIR} -lkokkoscore -lkokkoscontainers"

    nvcc ${NVCC_FLAGS} -std=c++20 -x cu \
        -I"${KOKKOS_INC}" \
        -DKOKKOS_ENABLE_CUDA=1 \
        -c "${KERNEL_DIR}/kernel_bfs_kokkos.cpp" \
        -o "${BIN_DIR}/kernel_bfs_kokkos.o"

    nvcc ${NVCC_FLAGS} \
        "${BIN_DIR}/kernel_bfs_kokkos.o" \
        -o "${BIN_DIR}/bfs-kokkos" \
        ${KOKKOS_LIBS} ${CUDA_LIBS}
    echo "[build_bfs] Kokkos: OK → ${BIN_DIR}/bfs-kokkos"
    [[ "${VERIFY}" == "true" ]] && run_verify "${BIN_DIR}/bfs-kokkos" "kokkos"
else
    echo "[build_bfs] SKIP Kokkos: libkokkoscore.a not found"
fi

# ── 3. RAJA ───────────────────────────────────────────────────────────────────
if [[ "${RAJA_FOUND}" == "true" ]]; then
    echo "[build_bfs] Building RAJA (root=${RAJA_ROOT}) ..."
    RAJA_INC="${RAJA_ROOT}/include"
    RAJA_LIB_DIR=$(ls -d "${RAJA_ROOT}/lib" "${RAJA_ROOT}/lib64" 2>/dev/null | head -1)
    RAJA_LIBS="-L${RAJA_LIB_DIR} -lRAJA"
    CAMP_LIBS=""
    [[ -n "${CAMP_LIB}" ]] && CAMP_LIBS="-L${CAMP_LIB}/lib -lcamp"

    nvcc ${NVCC_FLAGS} -x cu \
        --expt-extended-lambda --expt-relaxed-constexpr \
        -allow-unsupported-compiler \
        -I"${RAJA_INC}" \
        -c "${KERNEL_DIR}/kernel_bfs_raja.cpp" \
        -o "${BIN_DIR}/kernel_bfs_raja.o"

    g++ ${HOST_FLAGS} \
        "${BIN_DIR}/kernel_bfs_raja.o" \
        -o "${BIN_DIR}/bfs-raja" \
        ${RAJA_LIBS} ${CAMP_LIBS} ${CUDA_LIBS} -ldl -lpthread -lstdc++
    echo "[build_bfs] RAJA: OK → ${BIN_DIR}/bfs-raja"
    [[ "${VERIFY}" == "true" ]] && run_verify "${BIN_DIR}/bfs-raja" "raja"
else
    echo "[build_bfs] SKIP RAJA: libRAJA.a not found"
fi

# ── 4. Julia wrapper ──────────────────────────────────────────────────────────
echo "[build_bfs] Installing Julia wrapper ..."
cat > "${BIN_DIR}/bfs-julia" <<'JULIA_WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec julia --project="${REPO_ROOT}" \
    "${REPO_ROOT}/kernels/bfs/kernel_bfs_julia.jl" "$@"
JULIA_WRAPPER
chmod +x "${BIN_DIR}/bfs-julia"
echo "[build_bfs] Julia wrapper: OK → ${BIN_DIR}/bfs-julia"
[[ "${VERIFY}" == "true" ]] && \
    run_verify "${BIN_DIR}/bfs-julia" "julia" || true

# ── 5. Skipped ────────────────────────────────────────────────────────────────
echo "[build_bfs] SKIPPED numba: UNSUPPORTED_CC120"
echo "[build_bfs] SKIPPED sycl: NO_COMPILER on ${PLATFORM}"

echo "[build_bfs] Done.  Binaries in ${BIN_DIR}/"
ls -lh "${BIN_DIR}/"
