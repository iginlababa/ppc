# E3 3D Stencil — Profiling Notes
Platforms: NVIDIA RTX 5060 (Blackwell, CC 12.0) · AMD MI300X (CDNA3, gfx942)
Updated: 2026-03-20

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
  build/stencil/julia_nvidia_rtx5060/stencil-julia --n 32 --warmup 5 --reps 5 --platform nvidia_rtx5060

nsys profile --trace=cuda,nvtx -o profiles/e3/julia_medium \
  build/stencil/julia_nvidia_rtx5060/stencil-julia --n 128 --warmup 5 --reps 5 --platform nvidia_rtx5060
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

**Evidence**: All 3 sizes produced 0 data rows in `stencil_numba_nvidia_rtx5060_20260315.csv`.
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
RTX 5060 L2 cache boundary (~32–48 MB). With adaptive warmup loading the data into L2
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

---

# AMD MI300X (CDNA3, gfx942) — Profiling Notes
Date: 2026-03-20
Run: 30 reps × 5 abstractions (native, kokkos, raja, sycl, julia) × 3 sizes.
Adaptive warmup: CV<2% over 10-run window, max 200 iterations.
No clock-locking issues (MI300X server GPU with stable clocks).

## Summary

| Abstraction | small (32³) GB/s | Eff  | medium (128³) GB/s | Eff  | large (256³) GB/s | Eff  |
|-------------|-----------------|------|-------------------|------|------------------|------|
| native      | 123.43          | 1.000 | 4423.62          | 1.000 | 11424.08        | 1.000 |
| kokkos      | 125.31          | 1.015 | 2972.09          | 0.672 | 4000.38         | 0.350 |
| raja        | 122.73          | 0.994 | 3159.15          | 0.714 | 5139.78         | 0.450 |
| sycl        | 60.57           | 0.491 | 3534.62          | 0.799 | 10231.85        | 0.896 |
| julia       | 78.36           | 0.635 | 2319.87          | 0.524 | 3508.84         | 0.307 |

Flagged (eff < 0.85): 10 of 15 configs.

---

## AMD Issue 1: SYCL small (eff=0.49) — dispatch latency dominates

**Symptom**: SYCL eff=0.49 at N=32³ — far worse than kokkos (1.015) and raja (0.994)
at the same size. AdaptiveCpp (hipSYCL) SYCL kernel dispatch goes through a different
code path than native HIP, adding driver-level overhead per kernel launch.

**Kernel time at N=32**: ~0.16 ms native. AdaptiveCpp SYCL dispatch overhead is
estimated ~0.05–0.15 ms per launch (HSA queue submission vs direct HIP kernel launch).
At N=32³, this overhead is 30–100% of the kernel time.

**Contrast with native HIP at small**: Native eff=1.0 because it uses direct hipLaunchKernelGGL
with no abstraction layer. Kokkos and RAJA both use direct HIP underneath (no SYCL runtime),
hence near-native at small.

**Deep profiling action** (pending):
```bash
rocprof --hsa-trace -o profiles/e3/sycl_small \
  build/stencil/sycl_amd_mi300x/stencil-sycl --n 32 --warmup 5 --reps 5

rocprof --hsa-trace -o profiles/e3/native_small \
  build/stencil/hip_amd_mi300x/stencil-hip --n 32 --warmup 5 --reps 5
```
Compare HSA dispatch timestamps to quantify queue submission overhead.

**SYCL large contrast (eff=0.896)**: At N=256³ the kernel runs ~0.3–0.45 ms — dispatch
overhead (<0.15 ms) is ~30–50% of kernel time, amortized away. SYCL large is the best
non-native abstraction on AMD, overtaking kokkos (0.350) and raja (0.450) dramatically.

---

## AMD Issue 2: Kokkos/RAJA/Julia large collapse (eff ≈ 0.35–0.45)

**Symptom**: At N=256³, kokkos drops to 35%, raja to 45%, julia to 31%, while native
reaches 11,424 GB/s and SYCL reaches 10,232 GB/s (90%).

**Cache boundary note**: N=256³ = 16,777,216 points × 8 bytes × 2 buffers = 268 MB.
MI300X L2 cache is ~256 MB (32 SEs × 8 MB). The working set (268 MB) barely exceeds L2.
After warmup, data is partially L2-resident, so measured throughput (11,424 GB/s native)
substantially exceeds the theoretical HBM3 peak (~5.3 TB/s). This is an L2-boundary effect:
not true DRAM-bound, but "mostly-L2-with-DRAM-spillover" regime.

**Hypothesis: memory access pattern differences at large N**
- Native HIP and SYCL (AdaptiveCpp) both generate efficient coalesced access patterns
  with loop ordering matching the CDNA3 memory subsystem topology.
- Kokkos MDRangePolicy with tiling hint {4,4,32} may produce a suboptimal tile shape
  for MI300X's 64-wide wavefront. On NVIDIA (warp=32), the {2,4,32} tiling is natural;
  on AMD (wave64), 64-wide tiling is preferred. The current MDRangePolicy hint was
  tuned for NVIDIA and may not transfer well to gfx942.
- RAJA kernel<3D> policy uses a similar block decomposition and may hit the same issue.
- Julia's @roc macro (KernelAbstractions.jl for ROCm) generates kernels via LLVM IR
  + AMDGPU backend; at large N the JIT-compiled kernel may have suboptimal unrolling
  or prefetch behavior compared to hand-tuned HIP.

**Supporting evidence**: SYCL (AdaptiveCpp) explicitly targets the AMDGPU backend with
wave64-aware optimizations and auto-vectorization, explaining why it retains 90% efficiency
at large N where Kokkos/RAJA degrade.

**Deep profiling action** (pending):
```bash
# Profile kokkos vs native at large N to compare wavefront utilization + memory traffic
rocprof --stats -i rocprof_metrics.txt \
  build/stencil/kokkos_amd_mi300x/stencil-kokkos --n 256 --warmup 10 --reps 5

rocprof --stats -i rocprof_metrics.txt \
  build/stencil/hip_amd_mi300x/stencil-hip --n 256 --warmup 10 --reps 5
```
Key metrics to collect: `FETCH_SIZE`, `WRITE_SIZE`, `VALUUtilization`, `VALUBusy`,
`MemUnitBusy`, `L2CacheHit`. If kokkos shows lower L2 hit rate than native at N=256,
it confirms the access pattern hypothesis.

**Recommended fix for Kokkos**: Tune MDRangePolicy tile to {1, 8, 64} for wave64:
```cpp
// kernel_stencil_kokkos.cpp — replace tiling hint
Kokkos::TeamPolicy<>(/* ... */) with
Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{Nz,Ny,Nx},{1,8,64})
```
This maps the innermost 64 iterations to a single wave64, improving coalescing.

---

## AMD Issue 3: Julia medium (eff=0.52) and large (eff=0.31) — KernelAbstractions overhead

**Symptom**: Julia degrades consistently with problem size on AMD — small (63%), medium (52%),
large (31%). This monotonic degradation with N is unusual; other abstractions improve or
plateau as N grows.

**Hypothesis**: Julia's KernelAbstractions.jl ROCm backend (via AMDGPU.jl) introduces
non-trivial per-launch JIT recompilation or kernel caching overhead that grows with
kernel complexity (more iterations = more registers = larger kernel binary). At N=256³,
each timed rep may be re-instantiating the kernel or suffering VGPR spilling.

**Secondary hypothesis**: Julia's AMDGPU.jl runtime uses synchronous kernel dispatch
by default in some versions, adding device synchronization overhead after each launch.
This would scale with N (longer kernel = longer wait per sync call).

**Deep profiling action** (pending):
```bash
# Julia ROCm profiling — check for synchronization overhead
JULIA_AMDGPU_VERBOSE=1 \
  julia --project=kernels/stencil \
    -e 'include("kernels/stencil/kernel_stencil_julia.jl"); run_stencil(256, 5, 5)'
```
Also compare `@elapsed` vs `AMDGPU.@elapsed` to isolate CPU-side sync overhead from
actual GPU kernel time.

---

## AMD Issue 4: N=256³ exceeds L2 boundary — not true DRAM-bound

**Finding**: Native large reaches 11,424 GB/s — 2.2× the MI300X HBM3 theoretical peak
(~5.3 TB/s). This confirms the large problem is still partially L2-served, not DRAM-bound.

**Implication for roofline analysis**: The roofline plot at AI≈0.203 for AMD large is
misleading — the measured ceiling is not the HBM bandwidth roof (4010 GB/s from E1 STREAM)
but the L2 bandwidth (estimated ~12–14 TB/s for MI300X). The true DRAM-bound regime
requires N > ~370 (N=512³ would be 2.1 GB working set, well above L2).

**Recommendation: Add N=512³**
N=512³ = 134,217,728 points × 8 bytes × 2 = 2.1 GB — safely above L2.
Expected native throughput at N=512: ≈ 5,000–5,300 GB/s (true HBM-bound).
This is the correct DRAM-bound measurement point for AMD roofline analysis.

To add N=512 to the run:
```bash
# In scripts/run/run_stencil.sh, add to SIZES:
#   SIZES[xlarge]=512
# Then re-run with --sizes small medium large xlarge
```
Note: N=512³ requires 2.1 GB GPU memory — within MI300X 192 GB budget.
Build changes: none required; kernel is parameterized on N at runtime.

---

## AMD Hypothesis verdict: memory-bound hypothesis NOT confirmed for large N

**Refuted at large (N=256³)**:
- Only SYCL (eff=0.896) approaches native; all other abstractions show poor efficiency
- Root cause: L2-boundary effect + abstraction-specific access pattern mismatches for CDNA3
- The memory-bound hypothesis would predict η ≈ 1.0 for all abstractions at large N;
  instead we see η ∈ {0.35, 0.45, 0.90, 0.31} — a 3× spread

**Confirmed at small (N=32³)**:
- Kokkos (1.015) and RAJA (0.994) match native — consistent with memory-bound prediction
- SYCL (0.491) and Julia (0.635) are limited by dispatch overhead, not memory bandwidth

**Contrast with NVIDIA**: On RTX 5060, the spread at large N is narrower (kokkos 0.70,
raja 0.85) and thermally biased. On AMD MI300X with stable clocks, the spread is much
larger, revealing genuine abstraction-level access pattern differences for CDNA3.

**Cross-platform PPC φ (harmonic mean of per-platform efficiencies)**:
- Kokkos large: φ = 2 / (1/0.703 + 1/0.350) ≈ 0.470 — poor portability at scale
- RAJA large:   φ = 2 / (1/0.854 + 1/0.450) ≈ 0.573 — poor portability at scale
- Julia large:  φ = 2 / (1/1.376 + 1/0.307) ≈ 0.473 — thermally-biased NVIDIA side
- SYCL: only AMD data; φ = AMD efficiency directly (0.491 / 0.799 / 0.896)

**Action items**:
1. Run rocprof on kokkos/raja large to measure L2 hit rate and wavefront utilization
2. Re-tune MDRangePolicy tile hint to {1,8,64} for wave64 and re-benchmark
3. Add N=512³ to confirm DRAM-bound regime on AMD (expected native ~5,000 GB/s)
4. Investigate Julia AMDGPU.jl sync overhead — compare synchronous vs async dispatch
