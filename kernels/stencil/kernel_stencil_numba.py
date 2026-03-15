#!/usr/bin/env python3
"""
kernel_stencil_numba.py — E3 3D Stencil: Numba cuda.jit kernel.

E3 DESIGN DECISIONS
[D4-Numba] @cuda.jit kernel with 3D grid. Row-major NumPy arrays [iz, iy, ix].
  cuda.blockIdx.x → ix (innermost index, fastest varying in memory → coalesced).
  Block = (32, 4, 2) = 256 threads per block.
[D7-Numba] Adaptive warmup: CV < 2% over last 10 timings, max 200 iterations.
[D3] c0=0.5, c1=(1-c0)/6, FP64.
Platform note: Numba 0.64.0 does not support Blackwell (CC 12.0).
  UNSUPPORTED_CC120 exit path is included (same pattern as E2 numba).
"""

DESIGN_DECISIONS = """
# E3 DESIGN DECISIONS (Numba)
# [D4-Numba] @cuda.jit; 3D grid; row-major [iz,iy,ix]; cuda.blockIdx.x → ix (coalesced)
# [D7-Numba] Adaptive warmup: CV < 2% over last 10 timings
# [D3] c0=0.5, c1=(1-c0)/6, FP64
"""

import argparse
import sys
import time

import numpy as np

try:
    from numba import cuda, float64
    from numba.cuda.cudadrv.driver import LinkerError
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

C0 = np.float64(0.5)
C1 = np.float64((1.0 - 0.5) / 6.0)

WARMUP_MIN    = 10
WARMUP_MAX    = 200
WARMUP_WINDOW = 10
WARMUP_CV_CEIL = 2.0

BLOCK_X, BLOCK_Y, BLOCK_Z = 32, 4, 2


# ── PTX compatibility probe ───────────────────────────────────────────────────
def _probe_ptx_compat() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
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


# ── 7-point stencil kernel ────────────────────────────────────────────────────
if _NUMBA_AVAILABLE:
    @cuda.jit
    def stencil7pt_kernel(inp, out, N, c0, c1):
        """7-point Jacobi stencil. Row-major [iz,iy,ix]. cuda.threadIdx.x → ix."""
        ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        iz = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        if 1 <= ix < N - 1 and 1 <= iy < N - 1 and 1 <= iz < N - 1:
            out[iz, iy, ix] = (c0 * inp[iz, iy, ix]
                + c1 * (inp[iz, iy, ix-1] + inp[iz, iy, ix+1]
                      + inp[iz, iy-1, ix]  + inp[iz, iy+1, ix]
                      + inp[iz-1, iy, ix]  + inp[iz+1, iy, ix]))


# ── Performance formulas ──────────────────────────────────────────────────────
def interior_cells(N: int) -> int:
    return (N - 2) ** 3

def stencil_gbs(N: int, time_s: float) -> float:
    return interior_cells(N) * 64.0 / time_s / 1e9


# ── hw_state_verified (§9.7) ─────────────────────────────────────────────────
def compute_hw_state(vals):
    if not vals:
        return []
    med = float(np.median(vals))
    denom = abs(med) if abs(med) > 1e-12 else 1.0
    return [1 if abs(v - med) / denom <= 0.15 else 0 for v in vals]


# ── Adaptive warmup ───────────────────────────────────────────────────────────
def adaptive_warmup(run_once_fn, warmup_min=WARMUP_MIN, warmup_max=WARMUP_MAX,
                    window_size=WARMUP_WINDOW, cv_ceil=WARMUP_CV_CEIL):
    window = []
    total = 0
    while total < warmup_max:
        t0 = time.perf_counter()
        run_once_fn()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        window.append(ms)
        if len(window) > window_size:
            window.pop(0)
        total += 1
        if total >= warmup_min and len(window) == window_size:
            m = np.mean(window)
            s = np.std(window)
            cv = 100.0 * s / m if m > 0 else 100.0
            if cv < cv_ceil:
                break
    return total


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="E3 3D Stencil Numba")
    parser.add_argument("--n",        type=int,  default=256,      help="Grid side length N")
    parser.add_argument("--warmup",   type=int,  default=200,      help="Max adaptive warmup iterations")
    parser.add_argument("--reps",     type=int,  default=30,       help="Timed iterations")
    parser.add_argument("--platform", type=str,  default="unknown", help="Platform tag")
    parser.add_argument("--verify",   action="store_true",          help="Correctness check at N=16")
    args = parser.parse_args()

    N         = args.n
    warmup    = args.warmup
    reps      = args.reps
    platform  = args.platform

    print(DESIGN_DECISIONS, end="")
    print(f"# abstraction=numba N={N} warmup_max={warmup} reps={reps} platform={platform}",
          flush=True)

    # ── PTX compatibility check ───────────────────────────────────────────────
    if not _probe_ptx_compat():
        print("SKIP abstraction=numba reason=PTX_VERSION_MISMATCH "
              "(Numba/CUDA toolkit generates PTX unsupported by current driver)",
              file=sys.stderr, flush=True)
        sys.exit(0)

    # ── Correctness check ─────────────────────────────────────────────────────
    if args.verify:
        Nv = 16
        hIn = np.array([[[np.sin(ix/Nv) + np.cos(iy/Nv) + np.sin(iz/Nv + 0.5)
                          for ix in range(Nv)] for iy in range(Nv)]
                        for iz in range(Nv)], dtype=np.float64)
        # CPU reference
        hRef = np.zeros((Nv, Nv, Nv), dtype=np.float64)
        for iz in range(1, Nv-1):
            for iy in range(1, Nv-1):
                for ix in range(1, Nv-1):
                    hRef[iz,iy,ix] = (C0 * hIn[iz,iy,ix]
                        + C1 * (hIn[iz,iy,ix-1] + hIn[iz,iy,ix+1]
                              + hIn[iz,iy-1,ix]  + hIn[iz,iy+1,ix]
                              + hIn[iz-1,iy,ix]  + hIn[iz+1,iy,ix]))
        hOut = np.zeros((Nv, Nv, Nv), dtype=np.float64)
        dvIn  = cuda.to_device(hIn)
        dvOut = cuda.to_device(hOut)
        bpg = (int(np.ceil(Nv / BLOCK_X)), int(np.ceil(Nv / BLOCK_Y)),
               int(np.ceil(Nv / BLOCK_Z)))
        tpb = (BLOCK_X, BLOCK_Y, BLOCK_Z)
        stencil7pt_kernel[bpg, tpb](dvIn, dvOut, Nv, C0, C1)
        cuda.synchronize()
        hOut = dvOut.copy_to_host()
        denom = np.where(np.abs(hRef) < 1e-14, 1.0, np.abs(hRef))
        max_err = float(np.max(np.abs(hOut - hRef) / denom))
        ok = max_err < 1e-10
        print(f"VERIFY abstraction=numba N={Nv} max_rel_err={max_err:.2e} {'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            print("[E3 verify] numba FAILED — aborting before timing.", file=sys.stderr)
            sys.exit(1)
        print("[E3 verify] numba PASS — proceeding to timed measurement.", file=sys.stderr)

    # ── Allocate ─────────────────────────────────────────────────────────────
    rng  = np.random.default_rng(42)
    hIn  = (rng.random((N, N, N)) + 0.1).astype(np.float64)
    hOut = np.zeros((N, N, N), dtype=np.float64)

    dIn  = cuda.to_device(hIn)
    dOut = cuda.to_device(hOut)

    blocks_per_grid = (int(np.ceil(N / BLOCK_X)),
                       int(np.ceil(N / BLOCK_Y)),
                       int(np.ceil(N / BLOCK_Z)))
    threads_per_block = (BLOCK_X, BLOCK_Y, BLOCK_Z)

    def run_once():
        nonlocal dIn, dOut
        stencil7pt_kernel[blocks_per_grid, threads_per_block](dIn, dOut, N, C0, C1)
        cuda.synchronize()
        dIn, dOut = dOut, dIn  # swap buffers

    # Trigger JIT on first call (included in warmup)
    warmup_iters = adaptive_warmup(run_once, warmup_min=WARMUP_MIN, warmup_max=warmup)
    print(f"[E3] numba: adaptive warmup done in {warmup_iters} iterations",
          file=sys.stderr, flush=True)

    # ── Timed runs ────────────────────────────────────────────────────────────
    gbs_list = []
    for r in range(1, reps + 1):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0
        gbs = stencil_gbs(N, t1 - t0)
        gbs_list.append(gbs)
        print(f"STENCIL_RUN run={r} n={N} time_ms={time_ms:.6f} throughput_gbs={gbs:.6f}",
              flush=True)

    flags = compute_hw_state(gbs_list)
    for r, f in enumerate(flags, start=1):
        print(f"STENCIL_HW_STATE run={r} hw_state={f}", flush=True)

    clean = [g for g, f in zip(gbs_list, flags) if f == 1]
    n_clean = len(clean)
    if clean:
        med = float(np.median(clean))
        q1  = float(np.percentile(clean, 25))
        q3  = float(np.percentile(clean, 75))
        mn  = float(np.mean(clean))
        print(f"STENCIL_SUMMARY n={N} median_gbs={med:.4f} iqr_gbs={q3-q1:.4f} "
              f"min_gbs={min(clean):.4f} max_gbs={max(clean):.4f} "
              f"mean_gbs={mn:.4f} n_clean={n_clean} warmup_iters={warmup_iters}",
              flush=True)
    else:
        print(f"STENCIL_SUMMARY n={N} median_gbs=0.0 iqr_gbs=0.0 "
              f"min_gbs=0.0 max_gbs=0.0 mean_gbs=0.0 n_clean=0 warmup_iters={warmup_iters}",
              flush=True)


if __name__ == "__main__":
    main()
