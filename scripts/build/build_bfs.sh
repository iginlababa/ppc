#!/usr/bin/env bash
# Build all abstraction variants of BFS (E6).
# Usage: ./scripts/build/build_bfs.sh [--platform nvidia_rtx5060_laptop] [--verify] [--clean]
#
# Builds: native (CUDA + Thrust), kokkos, raja, julia (wrapper script)
# Skipped: numba (UNSUPPORTED_CC120 — Numba 0.64.0/Blackwell CC12.0 PTX mismatch)
#          sycl  (NO_COMPILER — no SYCL compiler on nvidia_rtx5060_laptop)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernels/bfs"
BIN_DIR="${REPO_ROOT}/bin/bfs"
mkdir -p "${BIN_DIR}"

PLATFORM="nvidia_rtx5060_laptop"
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

[[ "${CLEAN}" == "true" ]] && rm -rf "${BIN_DIR}"
mkdir -p "${BIN_DIR}"

# ── CUDA arch detection ────────────────────────────────────────────────────────
case "${PLATFORM}" in
    *rtx5060*|*blackwell*) CUDA_ARCH="sm_120" ;;
    *a100*|*ampere*)       CUDA_ARCH="sm_80"  ;;
    *v100*|*volta*)        CUDA_ARCH="sm_70"  ;;
    *rtx30*|*ga10*)        CUDA_ARCH="sm_86"  ;;
    *rtx40*|*ada*)         CUDA_ARCH="sm_89"  ;;
    *)                     CUDA_ARCH="sm_80"  ;;
esac
echo "[build_bfs] Platform=${PLATFORM}  CUDA arch=${CUDA_ARCH}"

# ── Kokkos / RAJA path detection (same as build_sptrsv.sh) ────────────────────
KOKKOS_ROOT=""
for p in /usr/local/kokkos /opt/kokkos /usr/kokkos "${HOME}/kokkos" \
          /usr/local /opt/local; do
    [[ -f "${p}/lib/libkokkoscore.a" || -f "${p}/lib64/libkokkoscore.a" ]] && \
        { KOKKOS_ROOT="${p}"; break; }
done
KOKKOS_FOUND=false
[[ -n "${KOKKOS_ROOT}" ]] && KOKKOS_FOUND=true

RAJA_ROOT=""
for p in /usr/local/raja /opt/raja /usr/raja "${HOME}/raja" \
          /usr/local /opt/local; do
    [[ -f "${p}/lib/libRAJA.a" || -f "${p}/lib64/libRAJA.a" ]] && \
        { RAJA_ROOT="${p}"; break; }
done
RAJA_FOUND=false
[[ -n "${RAJA_ROOT}" ]] && RAJA_FOUND=true

CAMP_LIB=""
for p in /usr/local/camp /opt/camp "${HOME}/camp" /usr/local /opt/local; do
    [[ -f "${p}/lib/libcamp.a" || -f "${p}/lib64/libcamp.a" ]] && \
        { CAMP_LIB="${p}"; break; }
done

# ── Common compile flags ───────────────────────────────────────────────────────
NVCC_FLAGS="-O3 -arch=${CUDA_ARCH} --extended-lambda -Xcompiler -O3,-march=native"
NVCC_FLAGS+=" -I${KERNEL_DIR}"
HOST_FLAGS="-O3 -march=native -std=c++20"
CUDA_LIBS="-lcuda -lcudart"

# ── Helper: run --verify correctness check ─────────────────────────────────────
run_verify() {
    local bin="$1"
    local abs_name="$2"
    echo "[build_bfs] Verifying ${abs_name} ..."
    "${bin}" --verify --graph erdos_renyi --n 1024 --reps 0 && \
    "${bin}" --verify --graph 2d_grid    --n 1024 --reps 0 && \
    echo "[build_bfs] ${abs_name} correctness OK" || \
        { echo "[build_bfs] ${abs_name} CORRECTNESS FAILED"; return 1; }
}

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

    # Two-step: nvcc -x cu compile then link
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
    echo "[build_bfs] SKIP Kokkos: libkokkoscore.a not found (set KOKKOS_ROOT or install)"
fi

# ── 3. RAJA ───────────────────────────────────────────────────────────────────
if [[ "${RAJA_FOUND}" == "true" ]]; then
    echo "[build_bfs] Building RAJA (root=${RAJA_ROOT}) ..."
    RAJA_INC="${RAJA_ROOT}/include"
    RAJA_LIB_DIR=$(ls -d "${RAJA_ROOT}/lib" "${RAJA_ROOT}/lib64" 2>/dev/null | head -1)
    RAJA_LIBS="-L${RAJA_LIB_DIR} -lRAJA"
    CAMP_LIBS=""
    [[ -n "${CAMP_LIB}" ]] && CAMP_LIBS="-L${CAMP_LIB}/lib -lcamp"

    # Two-step: nvcc -x cu compile, then g++ link
    nvcc ${NVCC_FLAGS} -x cu \
        -I"${RAJA_INC}" \
        -c "${KERNEL_DIR}/kernel_bfs_raja.cpp" \
        -o "${BIN_DIR}/kernel_bfs_raja.o"

    g++ ${HOST_FLAGS} \
        "${BIN_DIR}/kernel_bfs_raja.o" \
        -o "${BIN_DIR}/bfs-raja" \
        ${RAJA_LIBS} ${CAMP_LIBS} ${CUDA_LIBS} -lstdc++
    echo "[build_bfs] RAJA: OK → ${BIN_DIR}/bfs-raja"
    [[ "${VERIFY}" == "true" ]] && run_verify "${BIN_DIR}/bfs-raja" "raja"
else
    echo "[build_bfs] SKIP RAJA: libRAJA.a not found (set RAJA_ROOT or install)"
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
if [[ "${VERIFY}" == "true" ]]; then
    "${BIN_DIR}/bfs-julia" --verify --graph erdos_renyi --n 1024 --reps 0 && \
    "${BIN_DIR}/bfs-julia" --verify --graph 2d_grid    --n 1024 --reps 0 && \
    echo "[build_bfs] julia correctness OK" || \
        echo "[build_bfs] julia CORRECTNESS FAILED (non-fatal)"
fi

# ── 5. Skipped abstractions ───────────────────────────────────────────────────
echo "[build_bfs] SKIPPED numba: UNSUPPORTED_CC120 — Numba 0.64.0 generates PTX 9.2;" \
     "Blackwell CC 12.0 driver rejects PTX > 9.1.  Same issue as E2-E5."
echo "[build_bfs] SKIPPED sycl: NO_COMPILER — no SYCL compiler on ${PLATFORM}."

echo "[build_bfs] Done.  Binaries in ${BIN_DIR}/"
ls -lh "${BIN_DIR}/"
