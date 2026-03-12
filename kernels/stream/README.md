# E1 — STREAM Triad: `kernels/stream/`

Implements the STREAM Triad bandwidth benchmark (`A[i] = B[i] + scalar * C[i]`)
across all abstraction layers. See `benchmarks/stream/config.yaml` for experiment
parameters and `project_spec.md §8.2` for the full specification.

## Files

| File | Description |
|---|---|
| `stream_common.h` | Shared types, constants, stats, output format — included by ALL implementations |
| `kernel_stream_cuda.cu` | CUDA native baseline (`abstraction=native` on NVIDIA) |
| `kernel_stream_hip.cpp` | HIP native baseline (AMD MI250X/MI300X, `abstraction=native` on AMD) |
| `kernel_stream_kokkos.cpp` | Kokkos portable abstraction (CUDA/HIP/OpenMP backends) |
| `kernel_stream_raja.cpp` | RAJA portable abstraction (CUDA/HIP/OpenMP backends) |
| `kernel_stream_sycl.cpp` | SYCL 2020 (Intel DPC++ / AdaptiveCpp, GPU and CPU backends) |
| `kernel_stream_julia.jl` | Julia/CUDA.jl — explicit `@cuda` kernels, CUDA event timing |
| `stream-julia` | Shell wrapper — invokes `julia --project=. kernel_stream_julia.jl` |
| `Project.toml` | Julia package environment (CUDA.jl ≥ 5, Julia ≥ 1.9) |
| `kernel_stream_numba.py` | Python / Numba CUDA — TODO |
| `CMakeLists.txt` | CMake build for C++/CUDA/HIP/Kokkos/RAJA/SYCL targets |

## Build

```bash
# CUDA baseline (NVIDIA A100, sm_80)
cmake -S kernels/stream -B build/stream/cuda_nvidia_a100 \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/stream/cuda_nvidia_a100 --parallel

# For single precision (non-default):
cmake ... -DSTREAM_USE_FLOAT=ON

# Or use the build script (handles all abstractions and sets platform flags):
./scripts/build/build_stream.sh --platform nvidia_a100
```

## Run

```bash
# Medium size (2^26 elements, ~1.6 GB total) — 30 timed runs
./build/stream/cuda_nvidia_a100/stream-cuda \
    --arraysize 67108864 --numtimes 30 --warmup 10

# Large size (2^28 elements, ~6.4 GB) — primary E1 result
./build/stream/cuda_nvidia_a100/stream-cuda \
    --arraysize 268435456 --numtimes 30 --warmup 10

# All 5 BabelStream kernels (Copy, Mul, Add, Triad, Dot)
./build/stream/cuda_nvidia_a100/stream-cuda \
    --arraysize 268435456 --all-kernels

# Or use the run script:
./scripts/run/run_stream.sh --platform nvidia_a100 --reps 30
```

## Output format

Each timed run writes one `STREAM_RUN` line. After all runs, a `STREAM_SUMMARY`
line is written. `scripts/parse/parse_results.py` parses these into `data/performance.csv`.

```
STREAM_META device="NVIDIA A100 SXM4 80GB" cc=8.0 precision=double
STREAM_META abstraction=cuda n=268435456 sizeof=8 warmup=10 timed=30 block_size=1024 all_kernels=0
STREAM_META array_mb=2048.0 total_mb=6144.0
STREAM_CORRECT PASS max_err_a=4.44e-16 max_err_b=0.00e+00 max_err_c=0.00e+00
STREAM_RUN kernel=triad run=1 n=268435456 time_ms=0.52340 bw_gbs=1823.45
STREAM_RUN kernel=triad run=2 n=268435456 time_ms=0.51980 bw_gbs=1836.09
...
STREAM_SUMMARY kernel=triad median_bw_gbs=1830.12 iqr_bw_gbs=8.45 min_bw_gbs=1820.34 max_bw_gbs=1840.56 mean_bw_gbs=1829.87 outliers=0
```

## Correctness gate

The correctness check runs **after warmup, before timed runs**. If it fails,
the process exits with code 1 — no timing data is produced. This enforces the
rule from `project_spec.md §7`: "Abstractions must be functionally equivalent —
validated by correctness test before any timing run."

```bash
# Explicit correctness test:
pytest tests/correctness/ -v --experiment E1
# Or directly:
./tests/correctness/test_stream_correctness.sh --platform nvidia_a100
```

## Roofline context

| Platform | Peak BW (GB/s) | Triad target (80%) |
|---|---|---|
| NVIDIA A100 SXM4 | 2,039 | 1,631 GB/s |
| AMD MI250X | 3,277 | 2,622 GB/s |
| Intel PVC Max 1550 | 3,276 | 2,621 GB/s |

If the CUDA baseline does not reach ≥ 80% of peak on Large size, do not proceed
to abstraction experiments — the environment is not correctly configured
(see `project_spec.md §9.3`).

## Implementing a new abstraction

1. Copy `stream_common.h` constants to your implementation (or `#include` it
   if the language supports it).
2. Name the file `kernel_stream_<abstraction>.<ext>`.
3. Produce the same `STREAM_RUN` / `STREAM_SUMMARY` / `STREAM_CORRECT` output lines.
4. Add a `add_executable(stream-<abstraction> ...)` target to `CMakeLists.txt`.
5. Add the abstraction to `benchmarks/stream/config.yaml` under `abstractions:`.
6. Run `pytest tests/correctness/ -v --experiment E1 --abstraction <name>` and
   confirm PASS before any timing run.
