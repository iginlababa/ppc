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
