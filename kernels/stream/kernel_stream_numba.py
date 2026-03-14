#!/usr/bin/env python3
"""
kernel_stream_numba.py — Numba/CUDA abstraction for E1 STREAM Triad.

Implements all five BabelStream operations using explicit Numba @cuda.jit
kernels (not NumPy/CuPy broadcasting), so that the measured overhead vs.
the CUDA native baseline isolates Numba JIT and dispatch costs:

  Copy  : c[i] = a[i]
  Mul   : b[i] = scalar * c[i]
  Add   : c[i] = a[i] + b[i]
  Triad : a[i] = b[i] + scalar * c[i]   <- PRIMARY E1 metric
  Dot   : sum += a[i] * b[i]

Timing: numba.cuda.event() — device-side, millisecond precision, identical
method to kernel_stream_cuda.cu, kernel_stream_hip.cpp, and
kernel_stream_julia.jl.

JIT warm-up: first @cuda.jit kernel call compiles PTX.  Warmup iterations
absorb compilation so timed runs see pre-compiled kernels.  With
cache=True, Numba writes compiled PTX to __pycache__; subsequent invocations
skip recompilation.

Memory: cuda.device_array (cudaMalloc via Numba) for the three arrays plus
a single-element partial accumulator for Dot.

Type note: Copy/Mul/Add/Triad are fully generic (Numba infers T from the
argument dtype).  The Dot kernel uses cuda.shared.array whose dtype must be
a compile-time literal; it is hardcoded to float64 (the default STREAM_FLOAT).
For float32 builds (STREAM_USE_FLOAT=1), change the two numba.float64
references in dot_kernel to numba.float32.

Dependencies: numba >= 0.57, numpy, cuda-python (optional, for device query)
Setup:   pip install numba numpy  (or conda install -c conda-forge numba)
Run:     python kernel_stream_numba.py --arraysize 268435456 --numtimes 30
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys

import numpy as np
import numba
from numba import cuda

# ── PTX compatibility shim ────────────────────────────────────────────────────
# CUDA toolkit 13.2 / NVVM 4.0.0 emits PTX `.version 9.2`, but CUDA driver
# 590 (CUDA 13.1) only accepts PTX up to version 9.1 in cuLinkAddData.
# Our kernels contain no PTX 9.2-specific instructions (all operations are
# plain load/store/FMA present since PTX 7.x), so downgrading the header is safe.
# Remove this block once the system driver is updated to ≥ 591 (CUDA 13.2).
try:
    from numba.cuda.cudadrv import driver as _nbdriver
    # The concrete linker is CtypesLinker (not the abstract Linker base)
    _CtypesLinker = _nbdriver.CtypesLinker
    _orig_add_ptx = _CtypesLinker.add_ptx

    def _compat_add_ptx(self, ptx, name="<cudapy-ptx>"):
        if isinstance(ptx, (bytes, bytearray)):
            ptx_str = ptx.decode("utf-8", errors="replace")
        else:
            ptx_str = ptx
        ptx_str = ptx_str.replace(".version 9.2", ".version 9.1", 1)
        return _orig_add_ptx(self, ptx_str.encode("utf-8"), name)

    _CtypesLinker.add_ptx = _compat_add_ptx
except Exception as _e:
    import sys
    print(f"WARNING: PTX compat shim failed to install ({_e}); PTX 9.2→9.1 "
          "rewrite inactive — may hit cuLinkAddData version error", file=sys.stderr)

# ── Precision selection ───────────────────────────────────────────────────────
# Set STREAM_USE_FLOAT=1 to use single precision.
# NOTE: also change the numba.float64 literals in dot_kernel below.
_USE_FLOAT    = os.environ.get("STREAM_USE_FLOAT", "") == "1"
StreamFloat   = np.float32  if _USE_FLOAT else np.float64
STREAM_PREC   = "float"     if _USE_FLOAT else "double"

# ── Constants matching stream_common.h ───────────────────────────────────────
STREAM_INIT_A      = StreamFloat(0.1)
STREAM_INIT_B      = StreamFloat(0.2)
STREAM_INIT_C      = StreamFloat(0.0)
STREAM_SCALAR      = StreamFloat(0.4)
STREAM_CORRECT_TOL = 1.0e-6
STREAM_WARMUP      = 10
STREAM_TIMED       = 30
DOT_BLOCK_SIZE     = 1024   # must match cuda.shared.array size in dot_kernel

# ── GPU kernels ───────────────────────────────────────────────────────────────
# Non-Dot kernels: fully generic — Numba compiles a specialization for each
# element dtype it encounters on first call.  cache=True persists PTX to
# __pycache__ to avoid recompilation on subsequent runs.

@cuda.jit(cache=True)
def init_kernel(a, b, c, va, vb, vc):
    i = cuda.grid(1)
    if i < a.shape[0]:
        a[i] = va
        b[i] = vb
        c[i] = vc

@cuda.jit(cache=True)
def copy_kernel(a, c):
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i]

@cuda.jit(cache=True)
def mul_kernel(b, c, scalar):
    i = cuda.grid(1)
    if i < b.shape[0]:
        b[i] = scalar * c[i]

@cuda.jit(cache=True)
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

@cuda.jit(cache=True)
def triad_kernel(a, b, c, scalar):
    i = cuda.grid(1)
    if i < a.shape[0]:
        a[i] = b[i] + scalar * c[i]

# Dot: two-stage reduction mirroring kernel_stream_cuda.cu.
# Stage 1: each thread accumulates products into a register (grid-stride).
# Stage 2: in-block tree reduction in static shared memory.
# One cuda.atomic.add per block accumulates into the single-element output.
#
# cuda.shared.array requires a compile-time literal dtype — hardcoded to
# numba.float64.  For float32, change both occurrences below to numba.float32.
@cuda.jit(cache=True)
def dot_kernel(partial, a, b):
    sdata = cuda.shared.array(DOT_BLOCK_SIZE, dtype=numba.float64)  # see note above
    n   = a.shape[0]
    tid    = cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    acc = numba.float64(0)                                           # see note above
    i = cuda.blockIdx.x * cuda.blockDim.x + tid
    while i < n:
        acc += a[i] * b[i]
        i += stride
    sdata[tid] = acc
    cuda.syncthreads()

    s = cuda.blockDim.x >> 1
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s >>= 1

    if tid == 0:
        cuda.atomic.add(partial, 0, sdata[0])

# ── Bandwidth formulae (mirror stream_common.h) ───────────────────────────────
_sz = np.dtype(StreamFloat).itemsize

def triad_bw_gbs(n: int, t_s: float) -> float: return 3.0 * n * _sz / t_s / 1e9
def copy_bw_gbs(n: int,  t_s: float) -> float: return 2.0 * n * _sz / t_s / 1e9
def mul_bw_gbs(n: int,   t_s: float) -> float: return 2.0 * n * _sz / t_s / 1e9
def add_bw_gbs(n: int,   t_s: float) -> float: return 3.0 * n * _sz / t_s / 1e9
def dot_bw_gbs(n: int,   t_s: float) -> float: return 2.0 * n * _sz / t_s / 1e9

# ── Analytical expected values (mirror compute_expected in stream_common.h) ───
def compute_expected(n_passes: int):
    a = float(STREAM_INIT_A)
    b = float(STREAM_INIT_B)
    c = float(STREAM_INIT_C)
    s = float(STREAM_SCALAR)
    for _ in range(n_passes):
        c = a           # Copy
        b = s * c       # Mul
        c = a + b       # Add
        a = b + s * c   # Triad
    return a, b, c

# ── Statistics (mirror compute_stats in stream_common.h) ──────────────────────
def compute_stats(vals: list[float]) -> dict:
    n      = len(vals)
    srt    = sorted(vals)
    median = (srt[n // 2 - 1] + srt[n // 2]) / 2 if n % 2 == 0 else srt[n // 2]
    q1     = srt[n // 4]
    q3     = srt[(3 * n) // 4]
    iqr    = q3 - q1
    mean   = sum(vals) / n
    sigma  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    nout   = sum(1 for v in vals if abs(v - mean) > 2 * sigma)
    return dict(median=median, iqr=iqr, mean=mean,
                min=min(vals), max=max(vals), n_outliers=nout)

# ── Output helpers ────────────────────────────────────────────────────────────
def print_run_line(kernel: str, run_id: int, n: int,
                   time_ms: float, bw_gbs: float) -> None:
    print(f"STREAM_RUN kernel={kernel} run={run_id} n={n} "
          f"time_ms={time_ms:.5f} bw_gbs={bw_gbs:.4f}", flush=True)

def print_summary(kernel: str, s: dict) -> None:
    print(f"STREAM_SUMMARY kernel={kernel} "
          f"median_bw_gbs={s['median']:.4f} iqr_bw_gbs={s['iqr']:.4f} "
          f"min_bw_gbs={s['min']:.4f} max_bw_gbs={s['max']:.4f} "
          f"mean_bw_gbs={s['mean']:.4f} outliers={s['n_outliers']}", flush=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Numba/CUDA STREAM Triad benchmark (E1)")
    p.add_argument("--arraysize", "-n", type=int, default=1 << 26,
                   metavar="N", help="Elements per array (default: 2^26)")
    p.add_argument("--numtimes",  "-t", type=int, default=STREAM_TIMED,
                   metavar="N", help=f"Timed iterations (default: {STREAM_TIMED})")
    p.add_argument("--warmup",    "-w", type=int, default=STREAM_WARMUP,
                   metavar="N", help=f"Warm-up iterations (default: {STREAM_WARMUP})")
    p.add_argument("--blocksize", "-b", type=int, default=256,
                   metavar="N", help="GPU thread-block size (default: 256)")
    p.add_argument("--all-kernels", action="store_true",
                   help="Run Copy/Mul/Add/Triad/Dot (default: Triad only)")
    return p.parse_args()

# ── On-device initialization ──────────────────────────────────────────────────
def do_init(a, b, c, block_size: int) -> None:
    n = a.shape[0]
    grid = math.ceil(n / block_size)
    init_kernel[grid, block_size](a, b, c, STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C)
    cuda.synchronize()

# ── Timing helper ─────────────────────────────────────────────────────────────
# cuda.event_elapsed_time returns milliseconds (float).
def _elapsed_s(start, stop) -> float:
    return cuda.event_elapsed_time(start, stop) * 1e-3

# ── One timed pass ─────────────────────────────────────────────────────────────
def run_pass(a, b, c, partial, n: int, block_size: int,
             all_kernels: bool, ev_start, ev_stop) -> dict:
    scalar  = STREAM_SCALAR
    grid    = math.ceil(n / block_size)
    dot_grid = math.ceil(n / DOT_BLOCK_SIZE)

    def time_kernel(launch_fn, bw_fn):
        ev_start.record()
        launch_fn()
        ev_stop.record()
        ev_stop.synchronize()
        return bw_fn(n, _elapsed_s(ev_start, ev_stop))

    result = {}

    if all_kernels:
        result["copy"] = time_kernel(
            lambda: copy_kernel[grid, block_size](a, c),
            copy_bw_gbs)
        result["mul"] = time_kernel(
            lambda: mul_kernel[grid, block_size](b, c, scalar),
            mul_bw_gbs)
        result["add"] = time_kernel(
            lambda: add_kernel[grid, block_size](a, b, c),
            add_bw_gbs)

    result["triad"] = time_kernel(
        lambda: triad_kernel[grid, block_size](a, b, c, scalar),
        triad_bw_gbs)

    if all_kernels:
        # Reset accumulator; copy scalar zero to device
        partial[:] = cuda.to_device(np.zeros(1, dtype=StreamFloat))
        ev_start.record()
        dot_kernel[dot_grid, DOT_BLOCK_SIZE](partial, a, b)
        ev_stop.record()
        ev_stop.synchronize()
        result["dot"] = dot_bw_gbs(n, _elapsed_s(ev_start, ev_stop))

    return result

# ── Correctness check ──────────────────────────────────────────────────────────
def check_correctness(a, b, c, n: int, n_passes: int, all_kernels: bool) -> bool:
    if all_kernels:
        exp_a, exp_b, exp_c = compute_expected(n_passes)
    else:
        exp_a = float(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C)
        exp_b = float(STREAM_INIT_B)
        exp_c = float(STREAM_INIT_C)

    def relerr(got, expected):
        denom = abs(expected) if abs(expected) > 1e-12 else 1.0
        return abs(float(got) - expected) / denom

    def sample(arr, idx):
        """Copy a single element from device to host via a 1-element slice."""
        return arr[idx : idx + 1].copy_to_host()[0]

    indices = [0, n // 2, n - 1]
    max_ea = max_eb = max_ec = 0.0
    for idx in indices:
        max_ea = max(max_ea, relerr(sample(a, idx), exp_a))
        max_eb = max(max_eb, relerr(sample(b, idx), exp_b))
        max_ec = max(max_ec, relerr(sample(c, idx), exp_c))

    passed = (max_ea < STREAM_CORRECT_TOL and
              max_eb < STREAM_CORRECT_TOL and
              max_ec < STREAM_CORRECT_TOL)
    tag = "PASS" if passed else "FAIL"
    print(f"STREAM_CORRECT {tag} "
          f"max_err_a={max_ea:.3e} max_err_b={max_eb:.3e} max_err_c={max_ec:.3e}",
          flush=True)
    if not passed:
        if max_ea >= STREAM_CORRECT_TOL:
            print(f"STREAM_CORRECT DETAIL array=a expected={exp_a:.10f}", flush=True)
        if max_eb >= STREAM_CORRECT_TOL:
            print(f"STREAM_CORRECT DETAIL array=b expected={exp_b:.10f}", flush=True)
        if max_ec >= STREAM_CORRECT_TOL:
            print(f"STREAM_CORRECT DETAIL array=c expected={exp_c:.10f}", flush=True)
    return passed

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    opts        = parse_args()
    n           = opts.arraysize
    num_times   = opts.numtimes
    warmup      = opts.warmup
    block_size  = opts.blocksize
    all_kernels = opts.all_kernels

    # ── Device metadata ───────────────────────────────────────────────────────
    dev      = cuda.get_current_device()
    dev_name = dev.name if isinstance(dev.name, str) else dev.name.decode()

    print(f"STREAM_META abstraction=numba backend=cuda device=\"{dev_name}\" "
          f"numba={numba.__version__} python={platform.python_version()} "
          f"precision={STREAM_PREC} n={n} sizeof={_sz} "
          f"warmup={warmup} timed={num_times} all_kernels={int(all_kernels)}",
          flush=True)
    mb_per = n * _sz / (1024 ** 2)
    print(f"STREAM_META array_mb={mb_per:.1f} total_mb={3*mb_per:.1f}", flush=True)

    # ── Allocate device arrays ─────────────────────────────────────────────────
    # cuda.device_array calls cudaMalloc; no host-to-device transfer.
    a       = cuda.device_array(n, dtype=StreamFloat)
    b       = cuda.device_array(n, dtype=StreamFloat)
    c       = cuda.device_array(n, dtype=StreamFloat)
    partial = cuda.device_array(1, dtype=StreamFloat)  # Dot accumulator

    # ── Timing events ──────────────────────────────────────────────────────────
    ev_start = cuda.event(timing=True)
    ev_stop  = cuda.event(timing=True)

    # ── Initialize on device ──────────────────────────────────────────────────
    do_init(a, b, c, block_size)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    # First @cuda.jit call triggers PTX compilation (or loads from __pycache__
    # if cache=True hit).  All subsequent warmup passes run compiled kernels.
    for _ in range(warmup):
        run_pass(a, b, c, partial, n, block_size, all_kernels, ev_start, ev_stop)
    cuda.synchronize()

    # ── Correctness check ─────────────────────────────────────────────────────
    if not all_kernels:
        do_init(a, b, c, block_size)
        grid = math.ceil(n / block_size)
        triad_kernel[grid, block_size](a, b, c, STREAM_SCALAR)
        cuda.synchronize()

    passes_so_far = warmup if all_kernels else 0
    if not check_correctness(a, b, c, n, passes_so_far, all_kernels):
        print("CORRECTNESS CHECK FAILED — aborting.", file=sys.stderr, flush=True)
        return 1

    # ── Re-initialize for timed runs ───────────────────────────────────────────
    do_init(a, b, c, block_size)

    # ── Timed runs ─────────────────────────────────────────────────────────────
    triad_bw = []
    copy_bw = []; mul_bw = []; add_bw = []; dot_bw = []

    for i in range(1, num_times + 1):
        r = run_pass(a, b, c, partial, n, block_size, all_kernels, ev_start, ev_stop)
        triad_bw.append(r["triad"])
        if all_kernels:
            copy_bw.append(r["copy"])
            mul_bw.append(r["mul"])
            add_bw.append(r["add"])
            dot_bw.append(r["dot"])
        time_ms = 3.0 * n * _sz / (r["triad"] * 1e9) * 1e3
        print_run_line("triad", i, n, time_ms, r["triad"])
    cuda.synchronize()

    # ── Statistics ─────────────────────────────────────────────────────────────
    print_summary("triad", compute_stats(triad_bw))
    if all_kernels:
        print_summary("copy", compute_stats(copy_bw))
        print_summary("mul",  compute_stats(mul_bw))
        print_summary("add",  compute_stats(add_bw))
        print_summary("dot",  compute_stats(dot_bw))

    return 0


if __name__ == "__main__":
    sys.exit(main())
