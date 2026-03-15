# E3 3D Stencil — Profiling Notes
Platform: NVIDIA RTX 5060 Laptop (Blackwell, CC 12.0)
Date: 2026-03-15

## Summary

| Abstraction | small (32³) GB/s | Eff | medium (128³) GB/s | Eff | large (256³) GB/s | Eff | n_clean |
|-------------|-----------------|-----|-------------------|-----|------------------|-----|---------|
| native      | 160.21          | 1.000 | 878.31          | 1.000 | 435.38         | 1.000 | 30/28/28 |
| kokkos      | 121.74          | 0.760 | 799.16          | 0.910 | 306.06         | 0.703 | 30/27/22 |
| raja        | 150.87          | 0.942 | 869.47          | 0.990 | 371.66         | 0.854 | 30/30/13 |
| julia       | 112.78          | 0.704 | 607.22          | 0.691 | 598.91         | 1.376 | 30/30/16 |
| numba       | —               | —     | —               | —     | —              | —     | UNSUPPORTED_CC120 |
| sycl        | —               | —     | —               | —     | —              | —     | not built |

---

## Issue 1: No sudo for clock locking — thermal stepping at N=256

**Symptom**: N=256 (large) shows bimodal distributions. Early runs in each abstraction session
run at boosted clocks (~1400 MHz), then the GPU throttles to base clock (~1400→1100 MHz range)
after sustained load. This is visible in the raw data as a step-down in throughput mid-run.

**Effect on measurements**:
- `native` large: 435.38 GB/s median, CV=5.52% — high variance, bimodal runs
- `kokkos` large: 306.06 GB/s, but first 8 runs at ~384 GB/s before thermal step
- `raja` large: 371.66 GB/s, only 13/30 clean; early runs at boosted clock
- `julia` large: 598.91 GB/s efficiency=1.376 vs native — artifact: native large was
  measured earlier and had stronger thermal throttle than julia's session

**Root cause**: `nvidia-smi --lock-gpu-clocks` requires sudo. This laptop GPU does not expose
a stable base-clock lock without root. E1 (STREAM) used locked clocks; E3 cannot.

**Mitigation applied**: Adaptive warmup (CV<2% over 10-run window) stabilizes the thermal
state before timing begins. hw_state_verified flag filters outliers during timed phase.
However, warmup only stabilizes the warmup phase — thermal stepping can still occur during
the 30 timed reps themselves at large sizes.

**Recommendation**: For publication, re-run E3 on a server GPU (A100/H100) with locked clocks,
or repeat on the laptop in a cool environment with extended warmup (500+ iterations).
The julia large "efficiency>1" result should be footnoted as a thermal artifact.

---

## Issue 2: Julia small/medium efficiency < 0.85 — launch overhead hypothesis

**Flagged configurations**:
- julia small (N=32³): eff=0.70, 112.78 GB/s vs native 160.21 GB/s
- julia medium (N=128³): eff=0.69, 607.22 GB/s vs native 878.31 GB/s

**Hypothesis**: Julia's @cuda kernel incurs CUDA driver call overhead (~0.2–0.5 ms per launch)
that dominates at small kernel times. At N=32, each timed rep takes ~0.014 ms (7-pt kernel on
32³=32768 cells). Launch overhead is ~10–20× the kernel execution time → severe amortization penalty.

At N=128, kernel time ≈ 0.16 ms — still in the regime where JIT recompilation artifacts and
driver overhead affect measurements significantly. Runs 1–3 of each N=128 Julia session showed
elevated times (JIT warmup effect despite adaptive warmup), reducing the effective clean count.

**Evidence**: Julia N=256 (598.91 GB/s, efficiency=1.376) shows Julia is not fundamentally limited
— at large N where kernel time dominates, Julia equals or exceeds native (when native is thermally
throttled). The overhead penalty disappears when compute time >> launch overhead.

**Deep profiling action** (pending): Run `nsys profile` on julia N=32 and N=128 to quantify
CUDA API call overhead vs kernel execution time. Compare kernel time distribution with native.

```bash
nsys profile --trace=cuda,nvtx -o profiles/e3/julia_small \
  build/stencil/julia_nvidia_rtx5060_laptop/stencil-julia --n 32 --warmup 5 --reps 5 --platform nvidia_rtx5060_laptop

nsys profile --trace=cuda,nvtx -o profiles/e3/julia_medium \
  build/stencil/julia_nvidia_rtx5060_laptop/stencil-julia --n 128 --warmup 5 --reps 5 --platform nvidia_rtx5060_laptop
```

---

## Issue 3: Kokkos small/large efficiency < 0.85 — MDRangePolicy overhead

**Flagged configurations**:
- kokkos small (N=32³): eff=0.76, 121.74 GB/s vs native 160.21 GB/s
- kokkos large (N=256³): eff=0.70, 306.06 GB/s vs native 435.38 GB/s (but native large is thermally biased)

**Hypothesis for small**: MDRangePolicy with tiling hint {2,4,32} introduces block scheduling
overhead at very small grids. N=32 interior = 30³ = 27000 active cells; the block dimensions
may fragment occupancy or cause warp underutilization at the boundary.

**Hypothesis for large**: The kokkos large result shows a clear step-down at run 9 (397→313 GB/s
thermal transition). The 22/30 clean runs are all from the throttled thermal state. Native large
also throttled but its thermal step was earlier in its session. Without locked clocks, kokkos
and native saw different clock profiles during their separate measurement sessions, making
direct efficiency comparison unreliable at N=256.

**RAJA contrast**: RAJA small eff=0.94 (better than Kokkos 0.76) — RAJA's kernel<3D> policy
with direct block/thread mapping is closer to native CUDA thread launch semantics than
MDRangePolicy's tiling abstraction at small sizes.

---

## Issue 4: UNSUPPORTED_CC120 — Numba

Numba 0.64.0 (CUDA toolkit version in conda env) generates PTX 9.2 for Blackwell (CC 12.0),
but the NVIDIA driver on this system only accepts up to PTX 9.1. This is a platform compatibility
limitation, not a code defect.

**Evidence**: All 3 sizes produced 0 data rows in `stencil_numba_nvidia_rtx5060_laptop_20260315.csv`.
The binary detects PTX_VERSION_MISMATCH and exits cleanly with the UNSUPPORTED_CC120 message.

**Resolution**: Upgrade to Numba ≥ 0.65 when it supports PTX 9.2 fully, or run on an older
GPU (Hopper CC 9.0, Ampere CC 8.0) that numba 0.64 supports.

---

## Issue 5: SYCL — not built

`stencil-sycl` requires a SYCL-capable compiler (Intel oneAPI DPC++ or Clang with SYCL plugins).
The system has g++/nvcc only. SYCL target remains `pending` for this platform.
Will evaluate on Intel PVC (E-series experiments) or via CUDA SYCL backend (AdaptiveCpp/hipSYCL).

---

## N=128 anomaly: apparent bandwidth > DRAM peak

native medium: 878.31 GB/s, kokkos medium: 799.16 GB/s — both exceed DRAM peak (270 GB/s).

**Explanation**: The N=128 grid occupies 128³ × 8 × 2 buffers = 33.6 MB. This is right at the
RTX 5060 Laptop L2 cache boundary (~32–48 MB). With adaptive warmup loading the data into L2
(152 iterations for native N=32 vs 11 for N=128), the N=128 data fits partially in the L2
cache during measurement, giving apparent BW >> DRAM. This is L2 cache-served bandwidth, not DRAM.

This is consistent across abstractions: native=878, kokkos=799, raja=869 — all well above
the 270 GB/s DRAM ceiling. The AI=0.203 roofline analysis applies to DRAM-bound workloads;
at N=128 the kernel is L2-bound.

**Implication**: The "memory-bound hypothesis" is L2-bound at medium, DRAM-bound only at large.
For roofline analysis, N=256 is the valid DRAM-bound point.

---

## Hypothesis verdict: memory-bound → all abstractions near-native

**Partially confirmed** (for DRAM-bound large sizes, modulo thermal issues):
- RAJA large: eff=0.854 — just above the "excellent" threshold, within thermal noise
- Kokkos large: eff=0.703 — flagged, but thermally biased (native session was hotter)
- Julia large: eff=1.376 — thermally biased (inverted, native was more throttled)

**Not confirmed** for small sizes:
- Julia small eff=0.70, Kokkos small eff=0.76 — kernel launch overhead and tiling overhead
  are significant relative to compute time at 32³

**Conclusion**: The hypothesis holds for RAJA at medium/large. Kokkos and Julia show
abstraction-specific overheads at small granularity. Thermal instability prevents clean
efficiency measurements at N=256 without locked clocks.
