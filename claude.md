# CLAUDE.md
> **AI Agent Reference File** — Keep this brief. For full detail on any topic, see [`project_spec.md`](./project_spec.md).

---

## 1. Project Goals

Investigate **measurement-driven selective abstraction** in GPU-accelerated HPC:
- Quantify the performance overhead of abstraction layers (Kokkos, RAJA, SYCL, Julia, Python/Numba) relative to native CUDA/HIP baselines.
- Identify *when* abstraction is safe, costly, or recoverable via tuning.
- Produce a **taxonomy** of abstraction failure modes and a **decision framework** for practitioners.
- Deliver a reproducible artifact (code + data + paper) targeting a top HPC venue (SC / ICS / PPoPP).

> Full research questions, scope, and expected outcomes → `project_spec.md §1–3`

---

## 2. Architectural Overview

```
repo/
├── kernels/          # One subdirectory per experiment (E1–E7)
│   ├── stream/       # E1: STREAM Triad
│   ├── dgemm/        # E2: DGEMM
│   ├── spmv/         # E3: SpMV
│   ├── stencil/      # E4: 7-point Stencil
│   ├── bfs/          # E5: Graph BFS
│   ├── fft/          # E6: FFT
│   └── nbody/        # E7: N-Body
├── abstractions/     # Per-layer implementations (cuda, hip, kokkos, raja, sycl, julia, numba)
├── scripts/
│   ├── run_experiment.sh     # Single-experiment driver
│   ├── run_all.sh            # Full sweep launcher
│   ├── collect_metrics.py    # Timing + hardware counter harvester
│   ├── compute_ppc.py        # PPC formula + roofline normalization
│   ├── overhead_attr.py      # Overhead attribution pipeline
│   ├── build_taxonomy.py     # Pattern classifier → taxonomy JSON
│   ├── gen_figures.py        # All 21 paper figures
│   ├── process_e1.py         # E1 STREAM data processing
│   └── process_e2.py         # E2 DGEMM data processing
│   # NOTE: per-experiment processing scripts live in scripts/ top-level,
│   # not scripts/analysis/ — consistency with E1 takes precedence.
├── data/
│   ├── raw/                  # One CSV per run (never edited)
│   ├── processed/            # Aggregated, outlier-filtered tables
│   └── taxonomy/             # taxonomy.json + decision_framework.json
├── paper/                    # LaTeX source
│   ├── introduction.tex
│   ├── section2_background.tex
│   ├── related_work.tex
│   └── ...
├── docs/
│   └── project_spec.md       # ← Single source of truth for the full project
├── tests/                    # Correctness + regression tests
├── environment.yml           # Conda environment
└── CLAUDE.md                 # ← This file
```

> Full directory spec with file-level descriptions → `project_spec.md §4`

---

## 3. Design Style Guide

### Code
- **Language per layer:** CUDA/HIP (`.cu/.cpp`), Kokkos (`.cpp`), RAJA (`.cpp`), SYCL (`.cpp`), Julia (`.jl`), Python/Numba (`.py`)
- **Naming:** `kernel_<name>_<abstraction>.<ext>` — e.g., `kernel_stream_kokkos.cpp`
- **No magic numbers** — all problem sizes, thread counts, and iteration counts go in `config.yaml` per experiment
- **Every kernel must have a native CUDA/HIP baseline** — this is the reference for PPC computation
- **Abstractions must be functionally equivalent** — validated by correctness test before any timing run

### Data & Scripts
- Raw CSVs are **append-only** — never overwrite, never edit by hand
- All scripts accept `--experiment`, `--abstraction`, `--platform` flags
- Scripts must be **idempotent** — re-running produces the same output
- Use `pandas` + `numpy` for data processing; `matplotlib` + `seaborn` for figures

### LaTeX / Paper
- One `.tex` file per section
- All figures generated programmatically via `gen_figures.py` — no manual figure editing
- Citation keys: `AuthorYYYY` format (e.g., `Godoy2023`, `Deakin2019`)

> Full abstraction layer philosophy and tuning surface → `project_spec.md §7`

---

## 4. Constraints & Policies

| Constraint | Rule |
|---|---|
| **Reproducibility** | Every result must be reproducible from raw data + scripts alone |
| **Baseline parity** | Native baseline must be compiled with `-O3` and vendor-recommended flags |
| **Statistical validity** | Minimum 30 timed iterations; report median ± IQR; flag outliers > 2σ |
| **PPC threshold** | PPC ≥ 0.70 = portable; 0.50–0.70 = marginal; < 0.50 = non-portable |
| **Deep profiling trigger** | Auto-trigger Nsight/rocprof if overhead > 15% vs baseline |
| **No vendor lock-in in scripts** | All orchestration scripts must run on Linux; no macOS-only tools |
| **Data privacy** | No proprietary benchmark data committed to the repo |

> Full measurement protocol and thresholds → `project_spec.md §9`

---

## 5. Repository Etiquette

- **Branch naming:** `feature/<short-desc>`, `exp/<E1-stream>`, `fix/<issue>`, `paper/<section>`
- **Commit messages:** `[E2] Add Kokkos DGEMM kernel + correctness test` — always prefix with experiment ID or scope
- **Never commit to `main` directly** — open a PR, even if solo
- **Raw data (`data/raw/`)** is tracked via Git LFS — do not commit large CSVs directly
- **`environment.yml` must be updated** whenever a new dependency is added
- **Tag releases:** `v0.1-e1-complete`, `v1.0-submission`, `v1.1-artifact`
- **One experiment per PR** — keep diffs reviewable

---

## 6. Frequently Used Commands

```bash
# Build all abstractions for a single experiment
./scripts/build.sh --experiment E1 --platform a100

# Run a single experiment (all abstractions, all sizes)
./scripts/run_experiment.sh --experiment E1 --platform a100 --reps 30

# Run the full sweep (all 7 experiments)
./scripts/run_all.sh --platform a100

# Compute PPC for all collected data
python scripts/compute_ppc.py --input data/raw/ --output data/processed/ppc_results.csv

# Run overhead attribution pipeline
python scripts/overhead_attr.py --experiment E1 --platform a100

# Generate all paper figures
python scripts/gen_figures.py --input data/processed/ --output paper/figures/

# Build the paper
cd paper && latexmk -pdf main.tex

# Activate environment
conda activate hpc-abstraction
```

---

## 7. Testing Instructions

### Correctness Tests (run before any timing)
```bash
# Verify kernel output matches baseline within tolerance (1e-6 relative error)
pytest tests/correctness/ -v --experiment E1

# Run all correctness tests across all experiments
pytest tests/correctness/ -v
```

### Performance Regression Tests
```bash
# Check that baseline performance hasn't regressed > 5% from last recorded run
pytest tests/regression/ -v --platform a100
```

### Unit Tests (scripts)
```bash
# Test PPC computation logic
pytest tests/unit/test_compute_ppc.py -v

# Test overhead attribution logic
pytest tests/unit/test_overhead_attr.py -v

# Test taxonomy classifier
pytest tests/unit/test_taxonomy.py -v
```

### CI Policy
- All correctness tests must pass before a timing run is accepted
- Regression tests run automatically on every PR via GitHub Actions
- Paper build (`latexmk`) must succeed on every PR touching `paper/`

---

## 8. Key Reference Numbers

| Item | Value |
|---|---|
| Total experiments | 7 (E1–E7) |
| Abstractions per experiment | 5 (Kokkos, RAJA, SYCL, Julia, Numba) + 2 baselines |
| Target platforms | 3 (NVIDIA A100, AMD MI250X, Intel PVC) |
| Configs per experiment | 45 (7 abstractions × ~3 sizes × 3 platforms) |
| Total timed runs | ~9,450 (315 configs × 30 reps) |
| PPC portable threshold | ≥ 0.70 |
| Deep profiling trigger | overhead > 15% |
| Paper target venue | SC / ICS / PPoPP |

---

## 9. Where to Find Everything

| Need | Go to |
|---|---|
| Full experiment catalogue (E1–E7) | `project_spec.md §8` |
| Measurement protocol & PPC formula | `project_spec.md §9` |
| CSV schemas & data formats | `project_spec.md §10` |
| Analysis pipeline (8 stages) | `project_spec.md §11` |
| Taxonomy framework | `project_spec.md §12` |
| Decision framework & thresholds | `project_spec.md §13` |
| Hardware specs & peak BW values | `project_spec.md §5` |
| Full software stack | `project_spec.md §6` |
| Paper structure & figure list | `project_spec.md §16` |
| Timeline & critical path | `project_spec.md §17` |
| Risk register | `project_spec.md §18` |

---

## 10. Completed Experiment Status

### E1 — STREAM Triad (BabelStream)
- **Status:** COMPLETE
- **Platform:** nvidia_rtx5060_laptop
- **Abstractions run:** native, kokkos, raja, sycl (SKIP), julia, numba
- **Sizes:** 2^20, 2^26, 2^28
- **Key finding:** native bandwidth ~400 GB/s; kokkos/raja within 5% of native; julia within 10%; numba SKIP (PTX mismatch)
- **CSV:** `data/raw/stream_*_nvidia_rtx5060_laptop_*.csv`
- **Processed:** `data/processed/e1_stream_summary.csv`

### E2 — DGEMM (Dense Matrix Multiply)
- **Status:** COMPLETE — 2026-03-14
- **Platform:** nvidia_rtx5060_laptop (locked clocks, warmup-50 protocol §9.1)
- **Abstractions run:** native, native_cublas, raja_naive, julia_naive, julia_cublas
- **Skipped:** kokkos (binary not found), sycl (binary not found)
- **UNSUPPORTED_CC120:** numba — see Platform Limitations below
- **Sizes:** small (N=1024), medium (N=4096), large (N=8192)
- **Raw CSVs:** `data/raw/dgemm_*_nvidia_rtx5060_laptop_20260314.csv`
- **Processed:** `data/processed/e2_dgemm_summary.csv`
- **Key findings (N=8192):**
  - native: 176.3 GFLOP/s (baseline)
  - native_cublas: 176.6 GFLOP/s (≈native at large N — cuBLAS advantage appears at medium N=4096: 198.5 GFLOP/s)
  - raja_naive: 85.1 GFLOP/s (eff=0.48) — API Limitation pattern: RAJA lacks shared memory tiling primitives
  - julia_naive: 220.4 GFLOP/s (eff=1.25, faster than native) — see Unexpected Findings below
  - julia_cublas: 254.0 GFLOP/s (eff=1.44) — cuBLAS ceiling via Julia CUBLAS.gemm!
- **Hardware stability:** hw_state_verified=1 for all 450 runs (0 outliers)
- **JIT:** Julia 50-warmup fully absorbed compilation; run_id=1 indistinguishable from steady state
- **Methodology note:** PLATFORM constant in `process_e2.py` was `nvidia_rtx5060_laptop_locked` — corrected to `nvidia_rtx5060_laptop`

#### Platform Limitations
- **numba: UNSUPPORTED_CC120** — Numba 0.64.0 does not support Blackwell (CC 12.0). PTX 9.2 generated by libnvvm rejected by driver 590.48.01 (max PTX 9.1 via cuLinkAddData). No pip-installable fix exists as of E2 completion date (0.64.0 is latest available). CC 12.0 absent from `numba.cuda.cudadrv.nvvm.NVVM().supported_ccs` (list ends at Hopper 9.0). Distinction: SKIP = fixable environment gap; UNSUPPORTED_CC120 = hard toolchain–platform ceiling.

#### Unexpected Findings
- **julia_naive efficiency=1.25 at N=8192 is VALID** (confirmed by CUDA event timing: GPU-only time 4982–4990 ms matches wall time within 1.1%). Mechanism: Julia uses column-major storage; kernel maps `threadIdx.x → row`, so a warp of 32 threads with consecutive rows accesses the same column of A — 32 consecutive `float64`s → perfect coalescing, 1 cache-line per warp per inner-loop step. All 32 threads access the same `B[k, col]` → broadcast, zero bandwidth pressure. Zero `__syncthreads()` calls vs 512 in the native tiled kernel (256 tiles × 2 barriers at N=8192). At AI ≈ 683 FLOP/byte (compute-bound), sync overhead dominates tiling benefit. New taxonomy pattern: **Layout-Induced Coalescing Advantage** (§12.2 Pattern 5).

### E3 — 3D Stencil (7-point Jacobi)
- **Status:** COMPLETE — 2026-03-15
- **Platform:** nvidia_rtx5060_laptop (no locked clocks — sudo unavailable; adaptive warmup CV<2%)
- **Abstractions run:** native, kokkos, raja, julia
- **Skipped:** sycl (no SYCL compiler on platform)
- **UNSUPPORTED_CC120:** numba — same Numba 0.64.0/CC12.0 PTX mismatch as E2
- **Sizes:** small (N=32³), medium (N=128³), large (N=256³)
- **Raw CSVs:** `data/raw/stencil_*_nvidia_rtx5060_laptop_20260315.csv`
- **Processed:** `data/processed/e3_stencil_summary.csv`
- **Figures:** `figures/e3/fig7` through `fig12`
- **Profiling notes:** `profiles/e3/e3_profiling_notes.md`
- **Key findings (DRAM-bound at N=256, AI≈0.203 FLOP/byte):**
  - native large: 435.38 GB/s (thermally throttled, CV=5.52%)
  - kokkos large: 306.06 GB/s (eff=0.703) — thermally biased; Kokkos MDRangePolicy tiling overhead
  - raja large: 371.66 GB/s (eff=0.854) — borderline excellent; direct block/thread mapping
  - julia large: 598.91 GB/s (eff=1.376) — **thermal artifact**: native session ran hotter
  - N=128 anomaly: all abstractions show >700 GB/s — L2 cache-bound (33.6 MB data at L2 boundary)
- **Flagged for deep profiling (eff < 0.85):**
  - julia small (eff=0.70), julia medium (eff=0.69) — launch overhead dominant
  - kokkos small (eff=0.76), kokkos large (eff=0.70) — tiling policy + thermal bias
- **Hypothesis verdict:** PARTIALLY confirmed. RAJA medium eff=0.99 confirms memory-bound masks abstraction overhead. Julia small/medium invalidated by P001 Launch Overhead (now validated). Thermal instability prevents clean N=256 verification.
- **New taxonomy evidence:** P001 Launch Overhead Dominance validated (julia E3); P003 Memory Layout Mismatch NOT confirmed (RAJA vs Kokkos divergence points to tiling overhead P006 candidate)

#### Platform Limitations
- **No sudo for clock locking:** `nvidia-smi --lock-gpu-clocks` requires root. Adaptive warmup mitigates warmup phase but not thermal stepping during timed runs at N=256. Affects all large-size efficiency comparisons.

### E7 — N-Body (Lennard-Jones Molecular Dynamics)
- **Status:** COMPLETE — 2026-03-16
- **Platform:** nvidia_rtx5060_laptop
- **Abstractions run:** native (notile + tile), julia (notile only)
- **Skipped:** kokkos/raja (not installed), numba (UNSUPPORTED_CC120), sycl (NO_COMPILER)
- **Kernel variants:** `notile` (neighbor-list, global mem), `tile` (all-pairs, shared-mem TILE_SIZE=32, P006 test)
- **Physics:** LJ pairwise force, r_cut=2.5σ, ρ=0.8442 (FCC), one-sided, 20 FLOP/pair
- **Sizes:** small (N=4000, M=10), medium (N=32000, M=20), large (N=256000, M=40)
- **Raw CSVs:** `data/raw/nbody_{native,julia}_nvidia_rtx5060_laptop_20260316.csv`
- **Processed:** `data/processed/e7_nbody_summary.csv`
- **Figures:** `figures/e7/fig13–fig15`
- **Key findings:**
  - **n_nbrs = 54.0 exactly** for all sizes (perfect FCC crystal — uniform degree)
  - **AI = 0.971 FLOP/byte** — memory-bound, below ridge (~36.8 FLOP/byte on RTX 5060)
  - **P006 CONFIRMED**: tile speedup 4.1×/7.0×/13.6× over notile (small/medium/large). Cooperative warp loading eliminates repeated DRAM fetches; benefit grows with N as L2 cache saturation increases.
  - **Julia notile small eff=0.31**: P001 Launch Overhead Dominance — kernel duration ~0.02ms at N=4K; @cuda dispatch overhead (~0.3ms) dominates
  - **Julia notile medium eff=0.51**: still P001-limited at N=32K (~0.1ms kernel)
  - **Julia notile large eff=0.95 (good tier)**: kernel duration ~3ms at N=256K — P001 overhead amortized
  - **Julia large eff=0.95 confirms P001 crossover**: abstraction cost disappears when kernel duration ≫ dispatch latency
  - **VRAM at N=256K: 134.4 MB used** (≈34 MB positions + 64 MB neighbor list at MAX_NBRS=512 + 12 MB forces)
- **Design notes:** FCC positions are exactly periodic → CPU reference forces ≈ 0 (symmetry cancel); GPU uses min_image convention to match. Tile kernel uses all-pairs O(N²) algorithm, not neighbor-list; FLOPs reported accordingly.
- **CSV columns:** timestamp, experiment_id, kernel, abstraction, platform, problem_size, n_atoms, n_nbrs_mean, n_nbrs_max, actual_flops, run_id, execution_time_ms, throughput_gflops, hw_state_verified

### E6 — BFS (Breadth-First Search)
- **Status:** COMPLETE — 2026-03-16
- **Platform:** nvidia_rtx5060_laptop
- **Abstractions run:** native, julia (kokkos/raja libraries not installed)
- **Skipped:** numba (UNSUPPORTED_CC120), sycl (NO_COMPILER), kokkos (no libkokkoscore), raja (no libRAJA)
- **Graph types:** erdos_renyi (G(N, 10/N), irregular frontiers), 2d_grid (√N×√N 4-neighbor, regular diamond)
- **Sizes:** small (N=1024), medium (N=16384), large (N=65536)
- **Raw CSVs:** `data/raw/bfs_{native,julia}_nvidia_rtx5060_laptop_20260316.csv`
- **Processed:** `data/processed/e6_bfs_summary.csv`
- **Figures:** `figures/e6/fig23–fig26`
- **Key findings:**
  - **P008 confirmed again**: julia 2d_grid eff=0.352/0.439/0.358 for n_levels=63/255/511 — deep DAG collapses julia efficiency just as in E5
  - **Irregularity does NOT predict lower eff**: 2d_grid (low irregularity=0.57, regular) is WORSE than erdos_renyi (high irregularity=1.6, eff=0.43–0.73). n_levels is the dominant predictor.
  - **erdos_renyi is shallow (n_levels=6–8)**: few synchronisation barriers → julia maintains eff=0.43–0.73. Native also higher GTEPS on ER due to fewer levels.
  - **julia/erdos_renyi/small eff=0.731**: best julia result — only 6 levels, wide frontiers
  - **julia/2d_grid/small eff=0.352**: worst — 63 levels × @cuda dispatch cost ≈ 19ms overhead
  - **No eff > 1.0**: unlike E5 random/large (max_lw=524), BFS ER frontiers are too irregular and 2d_grid levels too narrow for Kokkos/RAJA to outperform native
- **P006 note:** P006 (Tiling Policy Overhead) absent — no tiling, flat forall/RangePolicy only
- **CSV columns:** timestamp, experiment_id, kernel, abstraction, platform, graph_type, problem_size, n_vertices, n_edges, n_levels, max_frontier_width, min_frontier_width, peak_frontier_fraction, run_id, execution_time_ms, throughput_gflops, hw_state_verified

### E5 — SpTRSV (Sparse Triangular Solve)
- **Status:** COMPLETE — 2026-03-16
- **Platform:** nvidia_rtx5060_laptop
- **Abstractions run:** native, kokkos, raja, julia
- **Skipped:** numba (UNSUPPORTED_CC120), sycl (NO_COMPILER)
- **Matrix types:** lower_triangular_laplacian, lower_triangular_random
- **Sizes:** small (N=256), medium (N=2048), large (N=8192)
- **Raw CSVs:** `data/raw/sptrsv_{native,kokkos,raja,julia}_nvidia_rtx5060_laptop_20260316.csv`
- **Processed:** `data/processed/e5_sptrsv_summary.csv`
- **Figures:** `figures/e5/fig19–fig22`
- **Key findings:**
  - **RAJA** best abstraction: only one with eff ≥ 0.93 at all laplacian sizes; eff=1.30 on random/large
  - **Julia** worst: eff=0.46 at laplacian/large (n_levels=181) — confirmed P001×n_levels mechanism
  - **Kokkos/RAJA random/large eff > 1.0** (1.11/1.30): wide levels (max_lw=524) fill GPU better than native's many small kernels
  - **P008 candidate validated**: efficiency keyed to n_levels, not N. Julia laplacian: eff=0.75→0.74→0.46 as n_levels=31→90→181; same N random matrix (n_levels=75) gives eff=0.90
  - **Irregular random < structured laplacian** for Julia (confirmed) but NOT for RAJA/Kokkos (random gives better efficiency due to wider levels)
- **Methodology note:** x reset (cudaMemset) excluded from timed region; included in warmup loop

### E4 — SpMV (Sparse Matrix-Vector Multiplication)
- **Status:** COMPLETE — 2026-03-16
- **Baseline note:** native SpMV baseline is naive one-thread-per-row; efficiency > 1.0 indicates abstraction scheduler outperforms naive assignment for this access pattern.
- **Platform:** nvidia_rtx5060_laptop
- **Abstractions:** native, kokkos, raja, julia, numba
- **SYCL:** disabled (no SYCL compiler on platform)
- **UNSUPPORTED_CC120:** numba — same Numba 0.64.0/CC12.0 PTX mismatch as E2/E3
- **Matrix types:** laplacian_2d (structured), random_sparse (uniform), power_law (Pareto γ=2.5 load imbalance)
- **Sizes:** small (N≈1024), medium (N≈8192), large (N≈32768 rows)
- **Primary metric:** GFLOP/s = 2×nnz / time_s / 1e9
- **AI:** ≈0.13 FLOP/byte — firmly memory-bound (ridge point ≈0.96 on RTX 5060)
- **Kernel design:** one CUDA thread per row, no warp-per-row reductions (honest baseline)
- **Kernel files:** `kernels/spmv/{spmv_common.h, kernel_spmv_{cuda,kokkos,raja,julia,numba}.*}`
- **Build:** `scripts/build/build_spmv.sh` (direct nvcc / kokkos_launch_compiler / two-step RAJA)
- **Run:** `scripts/run/run_spmv.sh` (loops abstraction × matrix_type × size)
- **Process:** `scripts/process_e4.py` → `data/processed/e4_spmv_summary.csv`
- **Figures:** `scripts/plot_e4.py` (fig13–fig17), `scripts/plot_e4_roofline.py` (fig18)
- **CSV columns:** timestamp, experiment_id, kernel, abstraction, platform, matrix_type, problem_size, n_rows, nnz, run_id, execution_time_ms, throughput_gflops, hw_state_verified
- **Expected findings:** memory-bound AI≈0.13 should mask abstraction overhead for laplacian_2d/random_sparse. power_law may reveal load imbalance effects. P001 Launch Overhead expected for julia at small/medium. Taxonomy E4 entry: `status: planned`.
