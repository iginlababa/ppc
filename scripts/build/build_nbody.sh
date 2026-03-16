#!/usr/bin/env bash
# Build E7 N-Body benchmarks (replaces placeholder stub).
# Usage: ./scripts/build/build_nbody.sh [--platform nvidia_rtx5060_laptop] [--verify] [--clean]
#
# Builds: native (notile + tile in one binary), julia (wrapper script)
# Skipped: kokkos (library not installed), raja (library not installed),
#           numba (UNSUPPORTED_CC120), sycl (NO_COMPILER)
#
# Both kernels live in a single binary: bin/nbody/nbody-native
#   --kernel notile  (default) → neighbor-list LJ
#   --kernel tile              → all-pairs shared-memory tiling (P006)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernels/nbody"
BIN_DIR="${REPO_ROOT}/bin/nbody"
mkdir -p "${BIN_DIR}"

PLATFORM="nvidia_rtx5060_laptop"
VERIFY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --verify)   VERIFY=true; shift ;;
        --clean)    CLEAN=true; shift ;;
        *) echo "[build_nbody] Unknown argument: $1"; exit 1 ;;
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
echo "[build_nbody] Platform=${PLATFORM}  CUDA arch=${CUDA_ARCH}"

# ── Compile flags ─────────────────────────────────────────────────────────────
# --extended-lambda: allows __device__ lambdas in host code (future-proofing)
NVCC_FLAGS="-O3 -arch=${CUDA_ARCH} --extended-lambda -Xcompiler -O3,-march=native"
NVCC_FLAGS+=" -I${KERNEL_DIR} -std=c++17"

# ── 1. Native CUDA ────────────────────────────────────────────────────────────
echo "[build_nbody] Building native CUDA (notile + tile) ..."
nvcc ${NVCC_FLAGS} \
    "${KERNEL_DIR}/kernel_nbody_cuda.cu" \
    -o "${BIN_DIR}/nbody-native" \
    -lcuda -lcudart
echo "[build_nbody] native CUDA: OK → ${BIN_DIR}/nbody-native"

if [[ "${VERIFY}" == "true" ]]; then
    echo "[build_nbody] Verifying notile ..."
    "${BIN_DIR}/nbody-native" --verify --n 4000 --kernel notile && \
        echo "[build_nbody] notile correctness: OK" || \
        { echo "[build_nbody] notile CORRECTNESS FAILED"; exit 1; }
fi

# ── 2. Julia wrapper ──────────────────────────────────────────────────────────
echo "[build_nbody] Installing Julia wrapper ..."
cat > "${BIN_DIR}/nbody-julia" <<'JULIA_WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec julia --project="${REPO_ROOT}" \
    "${REPO_ROOT}/kernels/nbody/kernel_nbody_julia.jl" "$@"
JULIA_WRAPPER
chmod +x "${BIN_DIR}/nbody-julia"
echo "[build_nbody] Julia wrapper: OK → ${BIN_DIR}/nbody-julia"

if [[ "${VERIFY}" == "true" ]]; then
    "${BIN_DIR}/nbody-julia" --verify --n 4000 --reps 0 && \
        echo "[build_nbody] julia correctness: OK" || \
        echo "[build_nbody] julia CORRECTNESS FAILED (non-fatal)"
fi

# ── 3. Skipped abstractions ───────────────────────────────────────────────────
echo "[build_nbody] SKIPPED kokkos: libkokkoscore.a not installed."
echo "[build_nbody] SKIPPED raja:   libRAJA.a not installed."
echo "[build_nbody] SKIPPED numba:  UNSUPPORTED_CC120 — same as E2-E6."
echo "[build_nbody] SKIPPED sycl:   NO_COMPILER on ${PLATFORM}."

echo "[build_nbody] Done. Binaries:"
ls -lh "${BIN_DIR}/"
