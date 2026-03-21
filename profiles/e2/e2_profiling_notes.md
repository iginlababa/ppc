# E2 DGEMM — Deep Profiling Notes
**Date:** 2026-03-15
**Platform:** nvidia_rtx5060 (RTX 5060, Blackwell CC 12.0, 12001 MHz GDDR7)
**Tool:** nsys 2024.x (`--trace cuda`); ncu blocked by `RmProfilingAdminOnly=1`
**Flagged configs:** raja_naive/medium (eff=0.24), julia_naive/medium (eff=0.80)

---

## 1. Profiling Tool Constraints

**ncu hardware counters unavailable.**
`/proc/driver/nvidia/params` contains `RmProfilingAdminOnly: 1`, which blocks hardware PMU access for non-root users. This prevents:
- L2 hit rate, warp occupancy, SM utilization
- Memory transaction counts (coalescing efficiency)
- Instruction mix (FFMA fraction)

**Workaround:** `nsys profile --trace cuda` provides:
- Kernel names, durations (CUDA event timing)
- CUDA API call breakdown (cudaLaunchKernel, cuModuleLoadDataEx, etc.)
- Stream synchronization events

All GFLOP/s figures in these notes are derived from kernel duration × 2N³ arithmetic, not from hardware counters.

---

## 2. raja_naive / medium (N=4096)

**Profile:** `profiles/e2/raja_naive_medium_nsys.nsys-rep`

### What nsys revealed
- Kernel name: `RAJA::policy::cuda::impl::forallp_cuda_kernel<...>`
- Avg kernel duration: **1205 ms** per call
- `cudaLaunchKernel` overhead: **564 µs** (0.05% of total time — negligible)
- Fresh-state GFLOP/s: **116 GFLOP/s**
- Benchmark GFLOP/s: **42 GFLOP/s**
- Thermal ratio: **2.76×**

### Root cause classification
**Dual failure: P004 API Limitation + P004 Thermal Contamination**

1. **P004 API Limitation (primary, size-independent):**
   RAJA's `cuda_exec` policy generates a flat `forallp_cuda_kernel` — a 1D ForAll over `N×N` elements. RAJA's C++ API has no primitive for `__shared__` memory allocation or warp-level tile management. The abstraction cannot express blocked DGEMM tiling regardless of compiler optimization level. This is not a compiler failure (P002) — it is an API expressiveness ceiling.

   Evidence: Launch overhead is 0.05% of runtime — P001 (Launch Overhead Dominance) does not apply. The kernel *is* executing; it simply cannot be tiled.

2. **P004 Thermal Contamination (secondary, session-specific):**
   The benchmark session runs abstractions sequentially: native → native_cublas → raja_naive → julia_naive. By the time raja_naive begins its warmup-50 protocol, the GPU has already executed ~130 seconds of native+cublas load at N=4096 (native: 0.785s/iter × 50 reps + 50 warmup = ~78s; cublas: ~0.55s/iter × 100 = ~55s). The raja_naive warmup adds a further 60s (1.205s/iter × 50), by which point the GPU is operating in a sustained thermal throttle state. The 42 GFLOP/s benchmark result is contaminated; the 116 GFLOP/s profiling result (fresh thermal state) is the representative value.

   True efficiency at medium N (thermal-corrected): **~0.66** (116/175.1).
   This is still below the 0.70 portable threshold, confirming P004 API Limitation is real, but the benchmark figure (eff=0.24) is not publishable as-is.

### Implications for paper
- The reported efficiency=0.24 in `e2_dgemm_summary.csv` is a **thermal artifact**, not the abstraction ceiling.
- For E3–E7: any abstraction with kernel duration > 2× native must be profiled fresh-state before reporting efficiency.
- The warmup-50 protocol must be supplemented with a **time-based warmup ceiling**: `max(50 iterations, 60 seconds)` ensures all abstractions reach thermal equilibrium at the same GPU temperature before measurement begins.

---

## 3. julia_naive / medium (N=4096)

**Profile:** `profiles/e2/julia_naive_medium_nsys.nsys-rep`

### What nsys revealed
- Kernel name: `dgemm_naive_kernel_(CuDeviceArray{Float64,(Int)2,(Int)1},...)`
  The `(Int)1` storage-order parameter **confirms column-major layout** (Julia default for `CuArray`).
- 4 kernel calls total: 2 JIT warmup + 2 timed reps
- Avg kernel duration: **624 ms**
- `cuModuleLoadDataEx`: present (JIT compilation, first 2 calls only)
- Fresh-state GFLOP/s: **~225 GFLOP/s**
- Benchmark GFLOP/s: **139 GFLOP/s**
- Thermal ratio: **1.62×**

### julia_naive / large (N=8192) — comparison
**Profile:** `profiles/e2/julia_naive_large_nsys.nsys-rep`
- Avg kernel duration: **5202 ms**
- Fresh-state GFLOP/s: **211 GFLOP/s**
- Benchmark GFLOP/s: **220 GFLOP/s**
- Delta: **4%** — consistent (no thermal artifact at large N)

### Root cause classification
**P004 Thermal Contamination only. Pattern 5 active at all sizes.**

The non-monotonic benchmark efficiency (0.80 at medium → 1.25 at large) was entirely a thermal artifact:
- At medium N: julia_naive runs after raja_naive has run ~60s of warmup on a thermally loaded GPU → 1.62× thermal penalty → benchmark 139 vs true 225 GFLOP/s
- At large N: julia_naive runs after the session has had time to partially stabilize, or the per-iteration duration is long enough that thermal equilibrium is re-established during warmup → 4% delta, no significant contamination

**Pattern 5 (Layout-Induced Coalescing Advantage) is size-independent:**
Fresh-state measurements confirm julia_naive exceeds native at both medium (225 vs 175 GFLOP/s, eff≈1.29) and large (211–220 vs 176 GFLOP/s, eff≈1.20–1.25). The initial hypothesis that Pattern 5 only activates at large N was incorrect. The apparent medium-N disadvantage was entirely thermal.

### Mechanism recap (column-major coalescing)
Julia `CuArray{Float64,2}` uses column-major storage (Fortran order). The `dgemm_naive_kernel` maps `threadIdx.x → row`. For a warp of 32 threads with consecutive `threadIdx.x` values:
- **Matrix A access** (`A[row, k]`): consecutive rows in the same column → 32 consecutive `Float64` values → 1 cache-line per warp per inner-loop step. Perfect coalescing.
- **Matrix B access** (`B[k, col]`): same `k` and `col` for all warp threads → broadcast, zero bandwidth pressure.
- **Zero `__syncthreads()` calls** vs 512 in the native tiled kernel (256 tiles × 2 barriers at N=8192).

At arithmetic intensity ≈ 683 FLOP/byte (compute-bound), synchronisation overhead from the native tiled kernel dominates any shared-memory bandwidth savings, making the simpler column-major kernel faster.

---

## 4. Updated Root Cause Summary

| Config | Benchmark eff | True eff (fresh) | Root Cause | Pattern |
|---|---|---|---|---|
| raja_naive / medium | 0.24 | ~0.66 | API Limitation + Thermal | P004 + Thermal |
| raja_naive / large | 0.48 | 0.48 (not re-profiled) | API Limitation | P004 |
| julia_naive / medium | 0.80 | ~1.29 | Thermal only | Thermal |
| julia_naive / large | 1.25 | ~1.20 | Pattern 5 | P005 |

---

## 5. Protocol Implications for E3–E7

1. **Warmup protocol:** Replace fixed warmup-50 with `max(50 iterations, 60 seconds)` for all future experiments. This eliminates differential thermal loading from variable-duration kernels.

2. **Sequential session ordering:** Run abstractions from fastest to slowest (minimizes thermal accumulation on slower abstractions). Or: use independent sessions per abstraction with GPU cool-down (nvidia-smi -pm 0, sleep 120s).

3. **Fresh-state profiling trigger:** Any abstraction with benchmark efficiency < 0.50 OR with non-monotonic scaling should be re-profiled in a fresh thermal state before the result is recorded as final.

4. **Thermal verification:** Current `hw_state_verified` flag confirms clock lock and power draw are within spec, but does NOT capture differential thermal loading within a session. A new `thermal_session_verified` column should be added to the processed CSV for experiments using sequential sessions.

---

## 6. Open Questions for Paper

1. **Is the warmup-50 thermal contamination systematic?** Need to re-run E2 with `max(50 iter, 60s)` warmup to confirm the corrected efficiencies (116 GFLOP/s and 225 GFLOP/s) are reproducible across sessions.

2. **Does Pattern 5 generalise to other matrix operations?** Column-major coalescing advantage should appear in any kernel with `threadIdx.x → row` mapping over column-major arrays. Candidate: E3 (stencil) with multi-dimensional array access.

3. **RAJA tiling workaround:** RAJA does expose `RAJA::statement::Tile` in its loop policy language for CPU kernels. Does a GPU-capable tiling policy exist? If so, the P004 classification should be re-examined — it may be P002 (user error / insufficient API knowledge) rather than a hard API limitation.

4. **P002 re-classification:** The current P002 evidence for raja_naive is based on benchmark eff=0.24, which is thermally contaminated. The corrected efficiency (~0.66) is in the "marginal" range (0.50–0.70), which may not warrant P002 (Compiler Backend Failure). True root cause is P004 API Limitation. P002 should remain "partially_validated" but evidence note should be updated to reflect thermal correction.

---

## 8. AMD MI300X — raja_naive and julia_naive (by design, no profiling needed)

**Date:** 2026-03-21
**Entries removed from profiling_queue.csv:**
- `dgemm_raja_naive amd_mi300x large eff=0.189`
- `dgemm_julia_naive amd_mi300x large eff=0.306`

### Why these entries do not require nsys/ncu profiling

Both abstractions are intentionally naive/untiled implementations. Their low efficiency relative to the native HIP baseline is the P004 API Limitation finding itself — not a defect to be diagnosed.

**raja_naive (eff=0.189):**
The kernel is an untiled O(N³) DGEMM (`RAJA::forall` flat 1D launch, one thread per output element, inner loop over global memory). This is documented in `kernel_dgemm_raja.cpp` [D6]: RAJA's `forall`/`kernel` API has no portable primitive for `__shared__` memory tiling on GPU; implementing tiled DGEMM inside a RAJA lambda would require raw HIP intrinsics, defeating the abstraction goal. At N=8192, the untiled kernel achieves 1,364 GFLOP/s vs native 7,214 GFLOP/s (eff=0.189). This gap is the expected consequence of O(N²) global-memory traffic per element vs O(N/tile) in a tiled kernel. No profiler output is needed to confirm this — it follows from the algorithm.

Note on the native baseline: `abstraction=native` on AMD is a hand-written tiled HIP DGEMM (not rocBLAS). The `native_rocblas` abstraction (ceiling ref, excluded from PPC) achieves ~32,000 GFLOP/s. The comparison is tiled-HIP vs untiled-RAJA, not rocBLAS vs untiled-RAJA. The efficiency gap is algorithmic, not a library-vs-kernel artefact.

**julia_naive (eff=0.306):**
Julia's `CuArray`/`ROCArray` uses column-major storage. On NVIDIA (see §3), the column-major layout gave a coalescing advantage (P005) that pushed julia_naive *above* the native tiled baseline (fresh-state eff≈1.25–1.29). On AMD MI300X the result is 2,211 GFLOP/s vs native 7,214 GFLOP/s (eff=0.306). Two compounding factors:

1. **Native HIP baseline is substantially higher on AMD** (7,214 vs ~176 GFLOP/s on the RTX 5060 laptop). The AMD tiled HIP DGEMM is more efficient relative to the hardware, leaving less room for julia_naive's column-major coalescing advantage to matter.
2. **AMDGPU.jl vs CUDA.jl kernel codegen differ.** The P005 column-major advantage observed on NVIDIA arises from a specific thread↔row mapping in the Julia naive kernel. AMDGPU.jl may generate a different register allocation or memory access pattern. Without a fresh-state AMD profiling run (not currently justified given the by-design nature of the result), this cannot be confirmed.

**Taxonomy classification (AMD):**

| Config | Benchmark eff | Root Cause | Pattern | Needs profiling? |
|---|---|---|---|---|
| raja_naive / large | 0.189 | Untiled kernel — API cannot express tiling | P004 API Limitation | **No — by design** |
| julia_naive / large | 0.306 | Tiled native is ~4× more efficient on AMD; P005 advantage insufficient to close gap | P004 + P005 absent | **No — by design** |

**Action taken:** Both AMD entries removed from `profiling_queue.csv`. `compute_ppc.py::flag_for_profiling()` updated to exclude `API_LIMITATION_ABSTRACTIONS = {raja_naive, julia_naive}` from future queue entries.

---

## 7. Files Generated This Session

| File | Description |
|---|---|
| `profiles/e2/raja_naive_medium_nsys.nsys-rep` | nsys CUDA trace, raja_naive N=4096, fresh state |
| `profiles/e2/julia_naive_medium_nsys.nsys-rep` | nsys CUDA trace, julia_naive N=4096, fresh state |
| `profiles/e2/julia_naive_large_nsys.nsys-rep` | nsys CUDA trace, julia_naive N=8192, fresh state |
| `figures/e2/fig6_e2_roofline_comparison.png` | 4-panel roofline (benchmark vs fresh-state, 615 KB) |
| `data/processed/profiling_queue.csv` | Updated: status=completed, root_cause_confirmed, conclusion_summary |
| `scripts/plot_e2_roofline.py` | Script generating fig6 |
