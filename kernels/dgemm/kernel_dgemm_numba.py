#!/usr/bin/env python3
"""
kernel_dgemm_numba.py — E2 DGEMM: Numba cuda.jit tiled kernel.

E2 DESIGN DECISIONS
[D2-Numba] 32×32 tiled DGEMM with cuda.shared.array — functionally equivalent
  to the CUDA tiled kernel. Block=(32,32), each thread computes one C element.
  Shared memory: sA[32,32] and sB[32,32] in float64. Static size known at
  JIT compile time.
[D3] alpha=1.0, beta=0.0. Row-major NumPy/CuPy arrays.
[D4] FP64 (float64) throughout.
"""

# ── E2 DESIGN DECISIONS (logged to stdout) ───────────────────────────────────
DESIGN_DECISIONS = """
# E2 DESIGN DECISIONS (Numba)
# [D2-Numba] 32x32 tiled DGEMM, cuda.shared.array, row-major, one thread per element
# [D3] alpha=1.0, beta=0.0
# [D4] FP64
"""

import argparse
import math
import sys
import time

import numpy as np
from numba import cuda, float64
from numba.cuda.cudadrv.driver import LinkerError

TILE = 32

# ── PTX compatibility probe ───────────────────────────────────────────────────
# Numba 0.64.0 + CUDA 13.2 toolkit generates PTX 9.2 for sm_120 Blackwell GPUs,
# but drivers < 590.xx may only support PTX 9.1. Detect early and exit cleanly.
def _probe_ptx_compat() -> bool:
    """Return True if Numba can compile and run a trivial kernel."""
    try:
        @cuda.jit
        def _probe(x):
            pass
        arr = cuda.to_device(np.zeros(1, dtype=np.float64))
        _probe[1, 1](arr)
        cuda.synchronize()
        return True
    except (LinkerError, Exception):
        return False

# ── Numba CUDA tiled DGEMM kernel ────────────────────────────────────────────
@cuda.jit
def dgemm_tiled_kernel(A, B, C, alpha, beta, N):
    """32×32 tiled DGEMM. Row-major storage. One thread per output element."""
    sA = cuda.shared.array((TILE, TILE), dtype=float64)
    sB = cuda.shared.array((TILE, TILE), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    row = by * TILE + ty
    col = bx * TILE + tx

    acc = float64(0.0)
    num_tiles = (N + TILE - 1) // TILE

    for t in range(num_tiles):
        k_a = t * TILE + tx
        sA[ty, tx] = A[row, k_a] if (row < N and k_a < N) else float64(0.0)

        k_b = t * TILE + ty
        sB[ty, tx] = B[k_b, col] if (k_b < N and col < N) else float64(0.0)

        cuda.syncthreads()

        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < N and col < N:
        C[row, col] = alpha * acc + beta * C[row, col]


# ── GFLOP/s formula ──────────────────────────────────────────────────────────
def dgemm_gflops(N: int, time_s: float) -> float:
    return 2.0 * N**3 / time_s / 1e9


# ── hw_state_verified (§9.7) ─────────────────────────────────────────────────
def compute_hw_state(vals):
    if not vals:
        return []
    med = float(np.median(vals))
    denom = abs(med) if abs(med) > 1e-12 else 1.0
    return [1 if abs(v - med) / denom <= 0.15 else 0 for v in vals]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="E2 DGEMM Numba tiled")
    parser.add_argument("--n",        type=int,  default=8192,     help="Matrix dimension")
    parser.add_argument("--warmup",   type=int,  default=50,       help="Warmup iterations")
    parser.add_argument("--reps",     type=int,  default=30,       help="Timed iterations")
    parser.add_argument("--platform", type=str,  default="unknown", help="Platform tag")
    parser.add_argument("--verify",   action="store_true",          help="Correctness check at N=128 then proceed to timing")
    args = parser.parse_args()

    N        = args.n
    warmup   = args.warmup
    reps     = args.reps
    platform = args.platform

    print(DESIGN_DECISIONS, end="")
    print(f"# abstraction=numba N={N} warmup={warmup} reps={reps} platform={platform}",
          flush=True)

    # ── PTX compatibility check (must happen before JIT compilation) ──────────
    if not _probe_ptx_compat():
        print("SKIP abstraction=numba reason=PTX_VERSION_MISMATCH "
              "(Numba/CUDA toolkit generates PTX unsupported by current driver)",
              file=sys.stderr, flush=True)
        sys.exit(0)  # exit 0 — this is an environment skip, not a kernel failure

    # ── Correctness check ─────────────────────────────────────────────────────
    if args.verify:
        Nv = 128
        hAv = np.array([[1.0 / (i + j + 2) for j in range(Nv)] for i in range(Nv)],
                       dtype=np.float64)
        hBv = hAv.copy()
        hRv = hAv @ hBv   # CPU reference (alpha=1, beta=0)
        hCv = np.zeros((Nv, Nv), dtype=np.float64)

        dvA = cuda.to_device(hAv)
        dvB = cuda.to_device(hBv)
        dvC = cuda.to_device(hCv)
        threads_v = (TILE, TILE)
        blocks_v  = (math.ceil(Nv / TILE), math.ceil(Nv / TILE))
        alpha_v   = np.float64(1.0)
        beta_v    = np.float64(0.0)

        # First call triggers JIT; result is still correct
        dgemm_tiled_kernel[blocks_v, threads_v](dvA, dvB, dvC, alpha_v, beta_v, Nv)
        cuda.synchronize()
        hCv = dvC.copy_to_host()

        denom  = np.where(np.abs(hRv) < 1e-12, 1.0, np.abs(hRv))
        max_err = float(np.max(np.abs(hCv - hRv) / denom))
        ok = max_err < 1e-6
        print(f"VERIFY abstraction=numba N={Nv} max_rel_err={max_err:.2e} {'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            print(f"[E2 verify] numba FAILED — aborting before timing.", file=sys.stderr)
            sys.exit(1)
        print("[E2 verify] numba PASS — proceeding to timed measurement.", file=sys.stderr)

    # Allocate and initialize row-major arrays
    rng = np.random.default_rng(42)
    hA = (rng.random((N, N)) + 0.1).astype(np.float64)
    hB = (rng.random((N, N)) + 0.1).astype(np.float64)
    hC = np.zeros((N, N), dtype=np.float64)

    dA = cuda.to_device(hA)
    dB = cuda.to_device(hB)
    dC = cuda.to_device(hC)

    threads_per_block = (TILE, TILE)
    blocks_per_grid   = (math.ceil(N / TILE), math.ceil(N / TILE))
    alpha = np.float64(1.0)
    beta  = np.float64(0.0)

    def run_once():
        dgemm_tiled_kernel[blocks_per_grid, threads_per_block](dA, dB, dC, alpha, beta, N)
        cuda.synchronize()

    # Trigger JIT compilation on first call, then warmup
    run_once()
    for _ in range(warmup - 1):
        run_once()

    gflops_list = []

    for r in range(1, reps + 1):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0
        gf = dgemm_gflops(N, t1 - t0)
        gflops_list.append(gf)
        print(f"DGEMM_RUN run={r} n={N} time_ms={time_ms:.6f} gflops={gf:.6f}",
              flush=True)

    # hw_state_verified
    flags = compute_hw_state(gflops_list)
    for r, f in enumerate(flags, start=1):
        print(f"DGEMM_HW_STATE run={r} hw_state={f}", flush=True)

    # Summary (clean runs only)
    clean = [g for g, f in zip(gflops_list, flags) if f == 1]
    n_clean = len(clean)
    if clean:
        med  = float(np.median(clean))
        q1   = float(np.percentile(clean, 25))
        q3   = float(np.percentile(clean, 75))
        mn   = float(np.mean(clean))
        print(f"DGEMM_SUMMARY n={N} median_gflops={med:.4f} iqr_gflops={q3-q1:.4f} "
              f"min_gflops={min(clean):.4f} max_gflops={max(clean):.4f} "
              f"mean_gflops={mn:.4f} n_clean={n_clean}", flush=True)
    else:
        print(f"DGEMM_SUMMARY n={N} median_gflops=0.0 iqr_gflops=0.0 "
              f"min_gflops=0.0 max_gflops=0.0 mean_gflops=0.0 n_clean=0", flush=True)


if __name__ == "__main__":
    main()
