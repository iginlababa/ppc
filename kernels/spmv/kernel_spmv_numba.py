#!/usr/bin/env python3
"""
kernel_spmv_numba.py — E4 SpMV: Numba cuda.jit CSR kernel.

E4 DESIGN DECISIONS
[D3-Numba] @cuda.jit kernel: one thread per row. Row-major CSR arrays as
  numpy arrays. cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x → row index.
[D7-Numba] Adaptive warmup: CV < 2% over last 10 timings, max 200 iterations.
Platform note: Numba 0.64.0 does not support Blackwell (CC 12.0) — same
  UNSUPPORTED_CC120 limitation as E2 and E3 numba. Exit path included.
"""

import argparse
import sys
import time

import numpy as np

try:
    from numba import cuda
    from numba.cuda.cudadrv.driver import LinkerError
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

SPMV_BLOCK_SIZE    = 256
WARMUP_MIN         = 10
WARMUP_MAX         = 200
WARMUP_WINDOW      = 10
WARMUP_CV_CEIL     = 2.0
SPMV_SEED          = 42
SPMV_RANDOM_NNZ    = 5


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


# ── SpMV CSR kernel: one thread per row ───────────────────────────────────────
if _NUMBA_AVAILABLE:
    @cuda.jit
    def spmv_csr_kernel(row_ptr, col_idx, values, x, y, nrows):
        """CSR SpMV. row_ptr/col_idx are 0-indexed."""
        row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if row < nrows:
            s = np.float64(0.0)
            for j in range(row_ptr[row], row_ptr[row + 1]):
                s += values[j] * x[col_idx[j]]
            y[row] = s


# ── Throughput ─────────────────────────────────────────────────────────────────
def spmv_gflops(nnz: int, time_s: float) -> float:
    return 2.0 * nnz / time_s / 1e9


# ── hw_state_verified ──────────────────────────────────────────────────────────
def compute_hw_state(vals):
    if not vals:
        return []
    med   = float(np.median(vals))
    denom = abs(med) if abs(med) > 1e-12 else 1.0
    return [1 if abs(v - med) / denom <= 0.15 else 0 for v in vals]


# ── Adaptive warmup ────────────────────────────────────────────────────────────
def adaptive_warmup(run_fn, warmup_min=WARMUP_MIN, warmup_max=WARMUP_MAX,
                    window_size=WARMUP_WINDOW, cv_ceil=WARMUP_CV_CEIL):
    window = []
    total  = 0
    while total < warmup_max:
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        window.append((t1 - t0) * 1000.0)
        if len(window) > window_size:
            window.pop(0)
        total += 1
        if total >= warmup_min and len(window) == window_size:
            m  = np.mean(window)
            s  = np.std(window)
            cv = 100.0 * s / m if m > 0 else 100.0
            if cv < cv_ceil:
                break
    return total


# ── Matrix generators ──────────────────────────────────────────────────────────
def generate_laplacian_2d(target_N):
    import math
    Nx = max(2, round(math.sqrt(target_N)))
    Ny = max(2, (target_N + Nx - 1) // Nx)
    N  = Nx * Ny

    row_ptr = np.zeros(N + 1, dtype=np.int32)
    for iy in range(Ny):
        for ix in range(Nx):
            row = iy * Nx + ix
            deg = 1
            if ix > 0:      deg += 1
            if ix < Nx - 1: deg += 1
            if iy > 0:      deg += 1
            if iy < Ny - 1: deg += 1
            row_ptr[row + 1] = deg
    for i in range(1, N + 1):
        row_ptr[i] += row_ptr[i - 1]
    nnz     = int(row_ptr[N])
    col_idx = np.zeros(nnz, dtype=np.int32)
    values  = np.zeros(nnz, dtype=np.float64)

    pos = row_ptr[:N].copy()
    for iy in range(Ny):
        for ix in range(Nx):
            row = iy * Nx + ix
            entries = [(row, -4.0)]
            if ix > 0:      entries.append((row - 1,  1.0))
            if ix < Nx - 1: entries.append((row + 1,  1.0))
            if iy > 0:      entries.append((row - Nx, 1.0))
            if iy < Ny - 1: entries.append((row + Nx, 1.0))
            entries.sort()
            for col, val in entries:
                k = int(pos[row])
                col_idx[k] = col
                values[k]  = val
                pos[row]  += 1
    return row_ptr, col_idx, values, N, nnz


def generate_random_sparse(N, nnz_per_row, seed):
    rng         = np.random.default_rng(seed)
    nnz         = N * nnz_per_row
    row_ptr     = np.arange(0, (N + 1) * nnz_per_row, nnz_per_row, dtype=np.int32)
    col_idx     = np.zeros(nnz, dtype=np.int32)
    values      = np.full(nnz, 1.0 / nnz_per_row, dtype=np.float64)
    pool        = np.arange(N, dtype=np.int32)
    for row in range(N):
        rng.shuffle(pool)
        cols = np.sort(pool[:nnz_per_row])
        col_idx[row * nnz_per_row:(row + 1) * nnz_per_row] = cols
    return row_ptr, col_idx, values, N, nnz


def generate_power_law(N, seed):
    rng     = np.random.default_rng(seed + 1)
    inv_a   = 1.0 / (2.5 - 1.0)
    d_max   = min(N // 4, 500)
    u       = rng.random(N)
    degrees = np.maximum(1, np.minimum(d_max, np.floor(np.power(1.0 - u, -inv_a)).astype(int)))
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        row_ptr[i + 1] = row_ptr[i] + degrees[i]
    nnz     = int(row_ptr[N])
    col_idx = np.zeros(nnz, dtype=np.int32)
    values  = np.ones(nnz, dtype=np.float64)
    pool    = np.arange(N, dtype=np.int32)
    for row in range(N):
        deg   = degrees[row]
        start = int(row_ptr[row])
        rng.shuffle(pool)
        cols  = []
        for c in pool:
            if c != row:
                cols.append(c)
            if len(cols) == deg:
                break
        cols.sort()
        for k, c in enumerate(cols):
            col_idx[start + k] = c
            values[start + k]  = 1.0
    return row_ptr, col_idx, values, N, nnz


def build_matrix(mtype, N):
    if mtype == "laplacian_2d":
        return generate_laplacian_2d(N)
    elif mtype == "random_sparse":
        return generate_random_sparse(N, SPMV_RANDOM_NNZ, SPMV_SEED)
    elif mtype == "power_law":
        return generate_power_law(N, SPMV_SEED)
    else:
        print(f"Unknown matrix type: {mtype}", file=sys.stderr)
        sys.exit(1)


def make_x_vector(N):
    rng = np.random.default_rng(SPMV_SEED + 99)
    return (rng.random(N) + 0.1).astype(np.float64)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="E4 SpMV Numba")
    parser.add_argument("--n",        type=int,   default=32768,       help="Number of rows")
    parser.add_argument("--matrix",   type=str,   default="laplacian_2d",
                        choices=["laplacian_2d", "random_sparse", "power_law"])
    parser.add_argument("--warmup",   type=int,   default=200)
    parser.add_argument("--reps",     type=int,   default=30)
    parser.add_argument("--platform", type=str,   default="unknown")
    parser.add_argument("--verify",   action="store_true")
    args = parser.parse_args()

    print(f"# abstraction=numba matrix={args.matrix} N_target={args.n} "
          f"warmup_max={args.warmup} reps={args.reps} platform={args.platform}",
          flush=True)

    # ── PTX compatibility check ───────────────────────────────────────────────
    if not _probe_ptx_compat():
        print("SKIP abstraction=numba reason=PTX_VERSION_MISMATCH "
              "(Numba/CUDA toolkit generates PTX unsupported by current driver)",
              file=sys.stderr, flush=True)
        sys.exit(0)

    # ── Build matrix ──────────────────────────────────────────────────────────
    row_ptr, col_idx, values, nrows, nnz = build_matrix(args.matrix, args.n)
    x = make_x_vector(nrows)

    print(f"# abstraction=numba matrix={args.matrix} N={nrows} nnz={nnz} "
          f"warmup_max={args.warmup} reps={args.reps} platform={args.platform}",
          flush=True)

    # ── Correctness check ─────────────────────────────────────────────────────
    if args.verify:
        rp_v, ci_v, val_v, nrows_v, nnz_v = generate_laplacian_2d(64)
        x_v   = make_x_vector(nrows_v)
        ref   = np.zeros(nrows_v, dtype=np.float64)
        for row in range(nrows_v):
            for j in range(int(rp_v[row]), int(rp_v[row+1])):
                ref[row] += val_v[j] * x_v[ci_v[j]]

        d_rp  = cuda.to_device(rp_v)
        d_ci  = cuda.to_device(ci_v)
        d_val = cuda.to_device(val_v)
        d_x   = cuda.to_device(x_v)
        d_y   = cuda.to_device(np.zeros(nrows_v, dtype=np.float64))
        bpg   = (nrows_v + SPMV_BLOCK_SIZE - 1) // SPMV_BLOCK_SIZE
        spmv_csr_kernel[bpg, SPMV_BLOCK_SIZE](d_rp, d_ci, d_val, d_x, d_y, nrows_v)
        cuda.synchronize()
        res = d_y.copy_to_host()

        denom   = np.where(np.abs(ref) < 1e-14, 1.0, np.abs(ref))
        max_err = float(np.max(np.abs(res - ref) / denom))
        ok      = max_err < 1e-10
        print(f"VERIFY abstraction=numba matrix=laplacian_2d N={nrows_v} max_rel_err={max_err:.2e} {'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            print("[E4 verify] numba FAILED — aborting.", file=sys.stderr)
            sys.exit(1)
        print("[E4 verify] numba PASS — proceeding to timed measurement.", file=sys.stderr)

    # ── Allocate device arrays ────────────────────────────────────────────────
    d_rp  = cuda.to_device(row_ptr)
    d_ci  = cuda.to_device(col_idx)
    d_val = cuda.to_device(values)
    d_x   = cuda.to_device(x)
    d_y   = cuda.to_device(np.zeros(nrows, dtype=np.float64))

    bpg = (nrows + SPMV_BLOCK_SIZE - 1) // SPMV_BLOCK_SIZE

    def run_once():
        spmv_csr_kernel[bpg, SPMV_BLOCK_SIZE](d_rp, d_ci, d_val, d_x, d_y, nrows)
        cuda.synchronize()

    warmup_iters = adaptive_warmup(run_once, warmup_min=WARMUP_MIN, warmup_max=args.warmup)
    print(f"[E4] numba: adaptive warmup done in {warmup_iters} iterations",
          file=sys.stderr, flush=True)

    # ── Timed runs ────────────────────────────────────────────────────────────
    mtype = args.matrix
    gflops_list = []
    for r in range(1, args.reps + 1):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0
        gf      = spmv_gflops(nnz, t1 - t0)
        gflops_list.append(gf)
        print(f"SPMV_RUN run={r} n={nrows} nnz={nnz} matrix={mtype} "
              f"time_ms={time_ms:.6f} throughput_gflops={gf:.6f}", flush=True)

    flags = compute_hw_state(gflops_list)
    for r, f in enumerate(flags, start=1):
        print(f"SPMV_HW_STATE run={r} hw_state={f}", flush=True)

    clean   = [g for g, f in zip(gflops_list, flags) if f == 1]
    n_clean = len(clean)
    if clean:
        med = float(np.median(clean))
        q1  = float(np.percentile(clean, 25))
        q3  = float(np.percentile(clean, 75))
        mn  = float(np.mean(clean))
        print(f"SPMV_SUMMARY n={nrows} nnz={nnz} matrix={mtype} "
              f"median_gflops={med:.4f} iqr_gflops={q3-q1:.4f} "
              f"min_gflops={min(clean):.4f} max_gflops={max(clean):.4f} "
              f"mean_gflops={mn:.4f} n_clean={n_clean} warmup_iters={warmup_iters}",
              flush=True)
    else:
        print(f"SPMV_SUMMARY n={nrows} nnz={nnz} matrix={mtype} "
              f"median_gflops=0.0 iqr_gflops=0.0 min_gflops=0.0 max_gflops=0.0 "
              f"mean_gflops=0.0 n_clean=0 warmup_iters={warmup_iters}", flush=True)


if __name__ == "__main__":
    main()
