# Project Specification: Measurement-Driven Selective Abstraction for Performance-Portable HPC Software

**Document Type:** Full Project Specification  
**Status:** Active — Research & Implementation  
**Version:** 1.0  
**Last Updated:** 2026-03-12  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Questions](#2-research-questions)
3. [Scope and Boundaries](#3-scope-and-boundaries)
4. [Repository Structure](#4-repository-structure)
5. [Hardware Platforms](#5-hardware-platforms)
6. [Software Stack](#6-software-stack)
7. [Abstraction Layers](#7-abstraction-layers)
8. [Experiment Catalogue](#8-experiment-catalogue)
9. [Measurement Protocol](#9-measurement-protocol)
10. [Data Schema](#10-data-schema)
11. [Analysis Pipeline](#11-analysis-pipeline)
12. [Taxonomy Framework](#12-taxonomy-framework)
13. [Decision Framework](#13-decision-framework)
14. [Validation Plan](#14-validation-plan)
15. [Tool Architecture](#15-tool-architecture)
16. [Paper Structure](#16-paper-structure)
17. [Timeline and Milestones](#17-timeline-and-milestones)
18. [Risk Register](#18-risk-register)
19. [Resource Requirements](#19-resource-requirements)
20. [Artifact Release Plan](#20-artifact-release-plan)

---

## 1. Project Overview

### 1.1 One-Line Summary

> This project develops the first measurement-driven, empirically grounded methodology for deciding *when* to use high-level programming abstractions and *when* to retain native GPU control in exascale HPC software.

### 1.2 The Problem

Exascale HPC systems are heterogeneous. Three major GPU vendors — NVIDIA, AMD, and Intel — each expose different memory hierarchies, execution models, and compiler backends. To write portable code, developers use abstraction layers: Kokkos, RAJA, SYCL, and Julia. These layers promise portability. They do not always deliver performance.

The current state of practice is guesswork. Developers choose abstractions based on team familiarity, project history, or vendor recommendation — not evidence. When performance degrades, root cause analysis is ad hoc. There is no systematic, measurement-driven framework for making these decisions.

### 1.3 The Contribution

This project does **not** propose a new abstraction layer or programming model. Instead, it provides:

1. **An empirical taxonomy** — the first systematic classification of abstraction failure modes and success patterns across architectures and workloads
2. **A measurement methodology** — a reproducible protocol for cross-layer performance attribution
3. **A quantified decision framework** — evidence-based thresholds and rules for abstraction selection
4. **A performance portability dataset** — 120+ configurations across 5 abstractions × 3 architectures × 8 workloads
5. **A decision support tool** — a command-line advisor that ingests workload profiles and recommends abstraction strategies

### 1.4 What Makes This Different

Most prior work either (a) benchmarks a single abstraction on a single platform, or (b) proposes a new framework. This project does neither. It treats abstraction selection as an **engineering decision problem** and builds the measurement infrastructure needed to make that decision rationally. The meta-level contribution — teaching the community *how* to evaluate abstractions systematically — has broader and more lasting impact than any single programming model innovation.

---

## 2. Research Questions

The project is organized around four primary research questions:

| # | Question | Where Answered |
|---|---|---|
| RQ1 | **When does abstraction help?** Under what workload characteristics and architectural features do abstractions deliver portability without performance penalty? | §8 Experiments, §12 Taxonomy |
| RQ2 | **When does abstraction fail?** What are the systematic patterns of abstraction breakdown, and which layers (compiler, runtime, memory model) contribute? | §9 Measurement Protocol, §12 Taxonomy |
| RQ3 | **How can we quantify the trade-off?** Can measurable metrics capture the multi-dimensional cost-benefit of abstraction vs. native? | §9 PPC, §11 Analysis |
| RQ4 | **What guidance can we provide?** Can we create actionable, evidence-based decision rules for abstraction granularity selection? | §13 Decision Framework, §15 Tool |

---

## 3. Scope and Boundaries

### 3.1 In Scope

- GPU-accelerated computing on discrete accelerators (NVIDIA, AMD, Intel)
- Established portability frameworks: Kokkos, RAJA, SYCL, Julia
- Scientific computing kernels: dense/sparse linear algebra, stencil, graph algorithms
- Performance portability as the primary outcome metric
- Single-node GPU performance (no distributed-memory MPI layer)

### 3.2 Explicitly Out of Scope

- Proposing new programming models or runtime systems
- Distributed-memory programming (MPI layer)
- Fault tolerance and resilience mechanisms
- Power and energy consumption (may be added as secondary metrics in future work)
- CPU-only workloads

---

## 4. Repository Structure

Every file in the project lives in a predictable location. No exceptions.

```
measurement-driven-abstraction/
│
├── benchmarks/                        # Benchmark source code (cloned, not written)
│   ├── stream/
│   │   └── BabelStream/               # git clone https://github.com/UoB-HPC/BabelStream
│   ├── dgemm/
│   │   └── RAJAPerf/                  # git clone https://github.com/LLNL/RAJAPerf
│   ├── stencil/
│   │   └── MiniGhost/
│   ├── spmv/
│   │   └── KokkosKernels/
│   ├── sptrsv/
│   ├── bfs/
│   │   └── GAP/                       # git clone https://github.com/sbeamer/gapbs
│   └── nbody/
│       └── CoMD/
│
├── scripts/
│   ├── env/
│   │   ├── setup_nvidia.sh            # Module loads, env vars for A100
│   │   ├── setup_amd.sh               # Module loads, env vars for MI250X
│   │   └── setup_intel.sh             # Module loads, env vars for PVC
│   ├── build/
│   │   ├── build_stream.sh            # Builds all 5 abstraction variants of STREAM
│   │   ├── build_dgemm.sh
│   │   ├── build_stencil.sh
│   │   ├── build_spmv.sh
│   │   ├── build_sptrsv.sh
│   │   ├── build_bfs.sh
│   │   └── build_nbody.sh
│   ├── run/
│   │   ├── run_experiments.py         # Master experiment driver
│   │   ├── run_stream.sh
│   │   ├── run_dgemm.sh
│   │   └── ...                        # One per kernel
│   ├── profile/
│   │   ├── collect_nsys.sh            # NVIDIA Nsight Systems profiling
│   │   ├── collect_ncu.sh             # NVIDIA Nsight Compute profiling
│   │   ├── collect_rocprof.sh         # AMD rocprof profiling
│   │   ├── collect_omniperf.sh        # AMD omniperf profiling
│   │   └── collect_vtune.sh           # Intel VTune profiling
│   └── parse/
│       ├── parse_results.py           # Extract metrics from raw output → CSV
│       └── validate_schema.py         # Assert CSV schema correctness before analysis
│
├── results/                           # Raw experiment output — never edited manually
│   ├── nvidia_a100/
│   │   ├── stream/
│   │   ├── dgemm/
│   │   └── ...
│   ├── amd_mi250x/
│   │   └── ...
│   └── intel_pvc/
│       └── ...
│
├── data/                              # Structured, validated data — source of truth
│   ├── performance.csv                # All performance results (1,350+ rows for E1 alone)
│   ├── profiling_metrics.csv          # Deep profiling counters
│   ├── productivity_metrics.csv       # LoC, tuning params, time-to-performance
│   └── taxonomy.json                  # Structured failure/success pattern database
│
├── analysis/
│   ├── compute_ppc.py                 # PPC calculation (Pennycook et al. formulation)
│   ├── compute_roofline.py            # Roofline normalization per platform
│   ├── overhead_attribution.py        # Decompose overhead by layer
│   ├── root_cause_analysis.py         # Flag gaps, correlate with profiling data
│   ├── generate_plots.py              # All paper figures
│   └── decision_framework.py          # Rule-based recommendation engine
│
├── notebooks/
│   ├── 01_baseline_exploration.ipynb
│   ├── 02_ppc_analysis.ipynb
│   ├── 03_overhead_attribution.ipynb
│   ├── 04_taxonomy_construction.ipynb
│   └── 05_decision_framework_validation.ipynb
│
├── tool/
│   ├── abstraction_advisor/           # CLI decision support tool
│   │   ├── profiler.py                # Workload characteristic extractor
│   │   ├── database.py                # Taxonomy database interface
│   │   ├── recommender.py             # Recommendation engine
│   │   └── cli.py                     # Command-line interface
│   └── README.md
│
├── paper/
│   ├── main.tex
│   ├── introduction.tex               # ✓ Written
│   ├── section2_background.tex        # ✓ Written
│   ├── section3_related_work.tex      # ✓ Written (needs gap statement bridge)
│   ├── section4_methodology.tex       # → In progress
│   ├── section5_experiments.tex       # → Pending
│   ├── section6_taxonomy.tex          # → Pending
│   ├── section7_decision_framework.tex # → Pending
│   ├── section8_validation.tex        # → Pending
│   ├── section9_conclusion.tex        # → Pending
│   ├── figures/
│   └── references.bib
│
├── containers/
│   ├── Dockerfile.nvidia
│   ├── Dockerfile.amd
│   └── Singularity.def                # For HPC systems that don't allow Docker
│
├── .env.example                       # Template for environment variables
├── REPRODUCIBILITY.md                 # Step-by-step reproduction guide
├── CONTRIBUTING.md
├── LICENSE                            # BSD/MIT
└── project_spec.md                    # This file
```

---

## 5. Hardware Platforms

### 5.1 Primary Target Architectures

| Vendor | GPU | Architecture | Memory | Peak BW | Peak FP64 | Access Path |
|---|---|---|---|---|---|---|
| NVIDIA | A100 SXM4 | Ampere (GA100) | 80 GB HBM2e | 2,039 GB/s | 19.5 TFLOP/s | University cluster / AWS p4d |
| AMD | MI250X | CDNA 2 (Aldebaran) | 128 GB HBM2e | 3,277 GB/s | 47.9 TFLOP/s | OLCF Frontier / AMD Cloud |
| Intel | GPU Max 1550 | Ponte Vecchio | 128 GB HBM2e | 3,276 GB/s | 22.2 TFLOP/s | ALCF Aurora / Intel DevCloud |

### 5.2 Peak Bandwidth Reference (Required for Roofline Normalization)

These values are used in every roofline calculation. They are hardware specifications, not measured values.

| GPU | Peak Memory Bandwidth | Source |
|---|---|---|
| NVIDIA A100 SXM4 | 2,039 GB/s | NVIDIA datasheet |
| AMD MI250X | 3,277 GB/s | AMD datasheet (both dies) |
| Intel PVC Max 1550 | 3,276 GB/s | Intel datasheet |

### 5.3 Hardware State Control Requirements

Before every experiment session, the following must be verified and logged:

**NVIDIA A100:**
- Persistence mode: enabled (`nvidia-smi -pm 1`)
- Clock lock: GPU and memory clocks locked to maximum stable frequency
- Auto-boost: disabled
- Record: driver version, CUDA runtime version, locked clock frequencies

**AMD MI250X:**
- Performance determinism mode: enabled via `rocm-smi`
- Clock frequency: locked
- Record: ROCm version, hipcc version, clock state

**Intel Ponte Vecchio:**
- GPU frequency: set via Intel GPU tools
- Record: oneAPI version, Level Zero runtime version

**All platforms:**
- CPU frequency scaling: disabled (set to `performance` governor)
- No other GPU jobs running on the node
- Record: OS kernel version, compiler versions, framework versions

> **Rule:** If hardware state cannot be verified and locked, do not run experiments. Results from unlocked hardware are not publishable.

### 5.4 Access Strategy

| Platform | Primary Access | Backup |
|---|---|---|
| NVIDIA A100 | University cluster | AWS p4d.24xlarge (~$15/hr spot) |
| AMD MI250X | OLCF Frontier (Director's Discretionary allocation) | AMD Accelerator Cloud |
| Intel PVC | ALCF Aurora (INCITE/ALCC allocation) | Intel DevCloud (free for academic) |

Apply for national lab allocations **at least 3 months before needed**.

---

## 6. Software Stack

### 6.1 Compilers and Runtimes

**NVIDIA Platform:**

| Component | Version | Purpose |
|---|---|---|
| CUDA Toolkit | 12.3+ | CUDA compilation, cuBLAS/cuSPARSE |
| NVHPC SDK | 23.11+ | Host compiler, math libraries |
| GCC | 11.x or 12.x | Host C++ compiler for Kokkos/RAJA |

**AMD Platform:**

| Component | Version | Purpose |
|---|---|---|
| ROCm | 5.7+ | HIP compilation, rocBLAS/rocSPARSE |
| AOMP | Latest | AMD OpenMP compiler |
| AdaptiveCpp (hipSYCL) | 0.9.4+ | SYCL implementation for AMD GPUs |

**Intel Platform:**

| Component | Version | Purpose |
|---|---|---|
| Intel oneAPI | 2024.0+ | Complete suite (DPC++, MKL, TBB) |
| icpx | Latest | Host and device compilation |
| Level Zero | Included in oneAPI | Low-level GPU runtime API |

### 6.2 Portability Frameworks

| Framework | Version | Install Method | Backends |
|---|---|---|---|
| Kokkos | 4.2+ | CMake from source | CUDA, HIP, SYCL, OpenMP, Serial |
| Kokkos-Kernels | 4.2+ | CMake (depends on Kokkos) | Same as Kokkos |
| RAJA | 2024.02+ | CMake from source or Spack | CUDA, HIP, OpenMP, Sequential |
| RAJAPerf Suite | Latest develop | Git + CMake | Matches RAJA backends |
| Julia | 1.10+ | juliaup | N/A (language runtime) |
| CUDA.jl | 5.x | Julia Pkg | NVIDIA CUDA |
| AMDGPU.jl | 0.9+ | Julia Pkg | AMD ROCm |
| oneAPI.jl | Latest | Julia Pkg | Intel (experimental) |

### 6.3 Benchmark Suites

| Suite | URL | Implementations Available |
|---|---|---|
| BabelStream | github.com/UoB-HPC/BabelStream | CUDA, HIP, SYCL, Kokkos, RAJA, Julia, OpenMP |
| RAJAPerf | github.com/LLNL/RAJAPerf | RAJA, CUDA, HIP, OpenMP (70+ kernels) |
| Mantevo Mini-Apps | mantevo.org | Multiple per app |
| GAP Benchmark Suite | github.com/sbeamer/gapbs | C++ OpenMP, custom CUDA |
| SuiteSparse | sparse.tamu.edu | Matrix collection (data only) |
| CoMD | github.com/ECP-copa/CoMD | C with OpenMP, CUDA variants |

### 6.4 Profiling Tools

| Tool | Platform | Purpose |
|---|---|---|
| Nsight Systems (`nsys`) | NVIDIA | System-wide timeline, kernel launch overhead |
| Nsight Compute (`ncu`) | NVIDIA | Detailed kernel counters, occupancy, memory throughput |
| `rocprof` | AMD | HIP kernel profiling, performance counters |
| Omniperf | AMD | Deep memory hierarchy analysis, wavefront behavior |
| Intel VTune | Intel | GPU offload analysis, EU utilization |
| Intel Advisor | Intel | Vectorization analysis, roofline generation |
| LIKWID | CPU | CPU performance counters, memory bandwidth |
| `perf` | All | General system profiling, cache analysis |

### 6.5 Analysis and Automation Tools

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Primary analysis language |
| Pandas | Latest | Tabular data manipulation, CSV parsing |
| NumPy | Latest | PPC calculation, numerical operations |
| Matplotlib + Seaborn | Latest | All paper figures |
| Jupyter | Latest | Interactive exploration, reproducible analysis |
| CMake | 3.20+ | Cross-platform build system |
| Spack | Latest | HPC package manager |
| Git | Latest | Version control |
| Zenodo / Figshare | — | Long-term data archival with DOI |
| Docker / Singularity | Latest | Containerized reproducibility environments |

---

## 7. Abstraction Layers

### 7.1 Native Implementations (Baseline — Performance Ceiling)

Native implementations define the performance ceiling for each platform. All PPC scores are computed relative to native.

| Platform | Native Model | Version | Key Notes |
|---|---|---|---|
| NVIDIA | CUDA | 12.x with `nvcc` | Industry standard; mature toolchain |
| AMD | HIP | ROCm 5.7+ with `hipcc` | CUDA-like syntax; distinct wavefront model |
| Intel | SYCL/DPC++ | oneAPI 2024.x | SYCL serves dual role: native for Intel, abstraction elsewhere |

**Implementation strategy:** Use hand-optimized versions from benchmark suites. Do not write native kernels from scratch. Validate that native implementations achieve ≥ 80% of theoretical peak before running any abstraction experiments.

### 7.2 Kokkos

- **Version:** 4.x
- **Philosophy:** Data structure abstraction — portable Views, Execution Spaces, Memory Spaces
- **Key mechanisms:** `parallel_for`, `parallel_reduce`, `parallel_scan`, `View`, `LayoutLeft/Right`
- **Tuning surface:** Team size, vector length, memory layout, memory traits, execution space config
- **Why included:** Most widely adopted portability layer in ECP (~30% of codes per Evans et al.)

### 7.3 RAJA

- **Version:** 2024.x
- **Philosophy:** Loop abstraction — execution policies, index sets, kernel fusion
- **Key mechanisms:** Execution policies, `RAJA::View`, `IndexSet`, kernel fusion
- **Tuning surface:** Execution policy selection, segment types, loop ordering
- **Why included:** Different philosophy from Kokkos — loop-level vs. data-structure-level abstraction. Comparison reveals whether abstraction philosophy affects portability.

### 7.4 SYCL

- **Implementations tested:** DPC++ (Intel), AdaptiveCpp/hipSYCL (CUDA/HIP targets)
- **Philosophy:** Open standard (Khronos Group) — vendor-neutral heterogeneous programming
- **Key mechanisms:** Queues, Buffers/Accessors, USM (Unified Shared Memory), lambda kernels
- **Tuning surface:** Buffer vs. USM model, work-group size, sub-group operations
- **Why included:** Tests whether open standards translate to portable performance or whether implementation quality dominates

### 7.5 Julia

- **Version:** 1.10+
- **Philosophy:** High-level language with JIT compilation — radically different from C++ template metaprogramming
- **Key mechanisms:** `CuArray`/`ROCArray`, broadcasting, `@cuda`/`@roc` macros, LLVM JIT
- **Tuning surface:** Minimal by design — but JIT compilation introduces warm-up overhead
- **Why included:** Godoy et al. show Julia achieves 0.9+ efficiency on AMD but struggles on NVIDIA. Investigating *why* is scientifically valuable and directly feeds the taxonomy.

---

## 8. Experiment Catalogue

### 8.1 Three-Dimensional Experimental Matrix

The full experiment space is:

```
5 abstractions × 3 platforms × 3 problem sizes × 7 kernels = 315 configurations
315 configurations × 30 timed runs = 9,450 individual timed kernel executions
```

Each kernel also has a deep profiling subset (flagged configurations only).

### 8.2 Workload Groups

#### GROUP 1 — Regular Workloads (Predictable Computation)

These are the **best-case scenario** for abstractions. If an abstraction fails here, it will fail everywhere.

---

**E1 — STREAM Triad**

| Property | Value |
|---|---|
| Source | BabelStream |
| Pattern | `A[i] = B[i] + scalar * C[i]` |
| Characteristics | Unit-stride access, bandwidth-bound (AI ≈ 0.25 FLOP/byte), no divergence |
| Primary metric | GB/s (effective memory bandwidth) |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL ✓ \| Julia ✓ |
| Platforms | All 3 |
| Problem sizes | Small: 2²⁰ \| Medium: 2²⁶ \| Large: 2²⁸ elements |
| Configs | 5 × 3 × 3 = **45** |
| Runs | 45 × 30 = **1,350** |
| Role in paper | Calibration experiment. Establishes performance ceiling. First PPC scores. |
| Bandwidth formula | `BW (GB/s) = (3 × N × 8) / time_seconds / 1e9` |

**Size rationale:**

| Label | Size | Total Data (3 arrays, FP64) | Purpose |
|---|---|---|---|
| Small | 2²⁰ ≈ 1M | ~24 MB | Exposes launch overhead, cache effects |
| Medium | 2²⁶ ≈ 67M | ~1.6 GB | Realistic working set |
| Large | 2²⁸ ≈ 268M | ~6.4 GB | Forces DRAM saturation |

**Verification requirement:** Confirm Large size exceeds L2 cache capacity on each GPU before running.

---

**E2 — Dense Matrix Multiplication (DGEMM)**

| Property | Value |
|---|---|
| Source | RAJAPerf Suite, Julia microbenchmarks (Godoy et al.) |
| Pattern | `C = A × B` (blocked implementation) |
| Characteristics | Compute-bound (AI ≈ 16–64 FLOP/byte), memory layout sensitive, compiler optimization sensitive |
| Primary metric | GFLOP/s |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL ✓ \| Julia ✓ |
| Platforms | All 3 |
| Problem sizes | Small: N=1024 \| Medium: N=4096 \| Large: N=16384 |
| Configs | 5 × 3 × 3 = **45** |
| Role in paper | Tests compiler optimization quality and memory layout awareness. Known Julia/NVIDIA gap (Godoy et al.) must be reproduced and explained. |

---

**E3 — 3D Stencil Computation**

| Property | Value |
|---|---|
| Source | Mantevo MiniGhost, RAJAPerf |
| Pattern | 7-point or 27-point stencil on structured grid |
| Characteristics | Regular grid traversal, halo exchanges, benefits from shared memory blocking |
| Primary metric | GFLOP/s and GB/s (balanced) |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL ✓ \| Julia ✓ |
| Platforms | All 3 |
| Problem sizes | Small: 64³ \| Medium: 256³ \| Large: 512³ |
| Configs | 5 × 3 × 3 = **45** |
| Role in paper | Middle-ground kernel. Tests multi-dimensional indexing abstractions and memory layout defaults. Known Kokkos layout mismatch pattern (§12 Pattern 3). |

---

#### GROUP 2 — Semi-Irregular Workloads (Structured Sparsity)

These test whether abstractions can handle indirect memory access and load imbalance.

---

**E4 — Sparse Matrix-Vector Multiplication (SpMV)**

| Property | Value |
|---|---|
| Source | RAJAPerf, Kokkos-Kernels, SuiteSparse Matrix Collection |
| Pattern | `y = A * x` where A is sparse (CSR format) |
| Characteristics | Irregular memory access, memory-bound, load imbalance across threads |
| Primary metric | GFLOP/s (effective, accounting for sparsity) |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL ✓ \| Julia ✓ |
| Platforms | All 3 |
| Matrix variants | Structured (2D/3D Laplacian) \| Power-law (social network) \| Random (worst-case) |
| Configs | 5 × 3 × 3 = **45** |
| Role in paper | Workhorse of iterative solvers. Tests indirect memory access handling. Known RAJA overhead pattern (§12 Pattern 1). |

---

**E5 — Sparse Triangular Solve (SpTRSV)**

| Property | Value |
|---|---|
| Source | Kokkos-Kernels, RAJA SpTRSV implementations |
| Pattern | Forward/backward substitution on sparse triangular matrix |
| Characteristics | Sequential dependencies, level-set parallelism required, irregular access |
| Primary metric | Solve time (ms), effective GFLOP/s |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL (partial) \| Julia (partial) |
| Platforms | All 3 |
| Configs | ~5 × 3 × 3 = **45** (some abstraction-platform combinations may be unavailable) |
| Role in paper | Tests task-based abstractions' ability to expose parallelism in apparently sequential code. |

---

#### GROUP 3 — Irregular Workloads (Dynamic Unpredictability)

These are the **stress test** for abstractions. Expected to show the largest performance gaps.

---

**E6 — Breadth-First Search (BFS)**

| Property | Value |
|---|---|
| Source | Graph500 reference, GAP Benchmark Suite |
| Pattern | Level-synchronous graph traversal |
| Characteristics | Highly irregular memory access, extreme load imbalance, control flow divergence, atomics |
| Primary metric | GTEPS (Giga Traversed Edges Per Second) |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA (partial) \| SYCL (partial) \| Julia (partial) |
| Platforms | All 3 |
| Graph variants | RMAT scale-22 \| Road network \| Random Erdős–Rényi |
| Configs | ~5 × 3 × 3 = **45** (availability-dependent) |
| Role in paper | Limits test. Expected to show abstraction failure. Feeds taxonomy irregular patterns. |

---

**E7 — N-Body Force Calculation (Molecular Dynamics)**

| Property | Value |
|---|---|
| Source | CoMD (Mantevo), miniMD |
| Pattern | Pairwise force computation with cutoff radius |
| Characteristics | Irregular neighbor lists, complex control flow, high AI but irregular memory |
| Primary metric | ns/day (simulation throughput) |
| Abstractions | Native ✓ \| Kokkos ✓ \| RAJA ✓ \| SYCL ✓ \| Julia (partial) |
| Platforms | All 3 |
| Configs | 5 × 3 × 3 = **45** |
| Role in paper | Production workload. Tests dynamic data structure handling. Bridges to real application validation. |

---

### 8.3 Experiment Summary Table

| ID | Kernel | Group | Source | Metric | Configs | Runs |
|---|---|---|---|---|---|---|
| E1 | STREAM Triad | Regular | BabelStream | GB/s | 45 | 1,350 |
| E2 | DGEMM | Regular | RAJAPerf | GFLOP/s | 45 | 1,350 |
| E3 | 3D Stencil | Regular | MiniGhost/RAJAPerf | GFLOP/s + GB/s | 45 | 1,350 |
| E4 | SpMV | Semi-irregular | RAJAPerf/KokkosKernels | GFLOP/s | 45 | 1,350 |
| E5 | SpTRSV | Semi-irregular | KokkosKernels | ms + GFLOP/s | ~45 | ~1,350 |
| E6 | BFS | Irregular | GAP/Graph500 | GTEPS | ~45 | ~1,350 |
| E7 | N-Body | Irregular | CoMD/miniMD | ns/day | 45 | 1,350 |
| **Total** | | | | | **~315** | **~9,450** |

---

## 9. Measurement Protocol

### 9.1 Timing Methodology

Every experiment follows this exact protocol. No deviations.

1. **Hardware lock** — verify and log GPU clock state (§5.3)
2. **Environment log** — record all software versions
3. **Warm-up** — execute kernel **10 times** (never recorded)
   - Purpose: amortize JIT compilation (Julia), warm GPU caches, exclude runtime init overhead
4. **Timed runs** — execute kernel **30 times** (all recorded)
5. **Per-run logging** — compute and store bandwidth/throughput for every individual run
6. **Statistics** — compute median, IQR, min, max over the 30 runs

> **Never report only the mean.** GPU kernels have outliers from OS jitter and DVFS. Median is the primary reported statistic.

### 9.2 Primary Metrics by Kernel

| Kernel | Primary Metric | Formula |
|---|---|---|
| STREAM Triad | GB/s | `(3 × N × sizeof(double)) / time_s / 1e9` |
| DGEMM | GFLOP/s | `(2 × N³) / time_s / 1e9` |
| 3D Stencil | GFLOP/s | `(flops_per_point × N³) / time_s / 1e9` |
| SpMV | GFLOP/s | `(2 × nnz) / time_s / 1e9` |
| SpTRSV | ms | Raw solve time |
| BFS | GTEPS | `edges_traversed / time_s / 1e9` |
| N-Body | ns/day | Simulation throughput |

### 9.3 Roofline Normalization

For every configuration, compute hardware utilization:

```
Utilization = BW_measured / BW_peak_theoretical
```

This is mandatory. Raw GB/s or GFLOP/s numbers without hardware context are not publishable. Reviewers will ask: *"Did the kernel saturate the hardware?"*

Native implementations must achieve ≥ 80% of theoretical peak on the Large problem size before abstraction experiments begin. If they do not, the environment setup is wrong.

### 9.4 Performance Portability Coefficient (PPC)

Following Pennycook et al. (2019):

```
PPC(a, p) = |H| / Σ_{h ∈ H} (1 / E_h)
```

Where:
- `a` = abstraction being evaluated
- `p` = problem (kernel + size)
- `H` = set of platforms where `a` has a working implementation
- `E_h` = efficiency on platform `h` = `Performance_abstraction_h / Performance_native_h`

**Efficiency is always relative to measured native performance, not theoretical peak.**

**Interpretation thresholds:**

| PPC | Meaning | Action |
|---|---|---|
| > 0.80 | Excellent portability | Document as success pattern |
| 0.60 – 0.80 | Acceptable — tuning needed | Document tuning requirements |
| < 0.60 | Poor — abstraction failing | Trigger deep profiling, add to failure taxonomy |

### 9.5 Deep Profiling Trigger

**Rule:** If any abstraction achieves `< 0.85 × native performance` on any platform, that configuration is automatically flagged for deep profiling.

Deep profiling is not optional for flagged configurations. It is the mechanism by which experiments feed the taxonomy.

**Deep profiling protocol:**

1. Run `nsys` (NVIDIA) / `rocprof` (AMD) / VTune (Intel) for timeline analysis
2. Run `ncu` (NVIDIA) / `omniperf` (AMD) for kernel-level counters
3. Inspect generated code (PTX for NVIDIA, GCN for AMD)
4. Compare native vs. abstraction generated code side-by-side
5. Attribute root cause to one of the four taxonomy categories (§12.1)

### 9.6 Overhead Attribution

For every flagged configuration, decompose total overhead into:

| Overhead Type | Measurement Method |
|---|---|
| Kernel launch latency | `nsys` CUDA API timeline: time between host call and kernel start |
| Synchronization overhead | Explicit/implicit barrier time from profiler timeline |
| Host-side framework overhead | Total time − kernel time |
| Memory transfer overhead | Compare total data movement: native vs. abstraction |
| Compiler code quality | PTX/GCN inspection: unrolling, vectorization, register usage |

### 9.7 Productivity Metrics

For every abstraction × kernel combination, record:

| Metric | How to Measure |
|---|---|
| Lines of Code (LoC) | Count kernel implementation only (not boilerplate) |
| Tuning parameter count | Count parameters requiring human decision |
| Time-to-performance | Iterations to reach 80% of best-known performance |
| Error message quality | Rate 1–5: clarity of compiler/runtime errors |
| Debugging difficulty | Rate 1–5: can you set breakpoints in actual kernel code? |
| Documentation completeness | Rate 1–5: are tuning guidelines available and accurate? |

---

## 10. Data Schema

### 10.1 Performance Results Table (`data/performance.csv`)

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | When this run executed |
| `experiment_id` | string | Unique ID: `{kernel}_{abstraction}_{platform}_{size}_{run_id}` |
| `kernel` | string | `stream`, `dgemm`, `stencil`, `spmv`, `sptrsv`, `bfs`, `nbody` |
| `abstraction` | string | `native`, `kokkos`, `raja`, `sycl`, `julia` |
| `platform` | string | `nvidia_a100`, `amd_mi250x`, `intel_pvc` |
| `problem_size` | string | `small`, `medium`, `large` |
| `problem_size_n` | integer | Actual numeric size (e.g., 268435456 for Large STREAM) |
| `run_id` | integer | 1–30 |
| `execution_time_ms` | float | Raw kernel time for this individual run |
| `throughput` | float | Performance in kernel-specific units (GB/s or GFLOP/s) |
| `efficiency` | float | `throughput / native_throughput` on same platform (computed post-hoc) |
| `hardware_state_verified` | boolean | Was clock locking confirmed before this session? |
| `compiler_version` | string | Compiler used for this build |
| `framework_version` | string | Framework version (Kokkos 4.2, ROCm 5.7, etc.) |

### 10.2 Profiling Metrics Table (`data/profiling_metrics.csv`)

| Column | Type | Description |
|---|---|---|
| `experiment_id` | string | Foreign key to performance table |
| `kernel_time_ms` | float | Pure GPU kernel execution time |
| `launch_overhead_ms` | float | Time in kernel launch and setup |
| `memory_transfer_mb` | float | Total data transferred host↔device |
| `occupancy_percent` | float | GPU occupancy |
| `sm_efficiency_percent` | float | SM/CU utilization |
| `memory_bandwidth_achieved_gbs` | float | Measured memory throughput |
| `l2_cache_hit_rate` | float | L2 cache hit percentage |
| `warp_divergence_percent` | float | Warp/wavefront divergence |
| `register_usage` | integer | Registers per thread |
| `profiler_tool` | string | `nsys`, `ncu`, `rocprof`, `omniperf`, `vtune` |

### 10.3 Productivity Metrics Table (`data/productivity_metrics.csv`)

| Column | Type | Description |
|---|---|---|
| `kernel` | string | Benchmark name |
| `abstraction` | string | Abstraction layer |
| `loc_kernel` | integer | Lines of code (kernel only) |
| `loc_total` | integer | Lines of code (including boilerplate) |
| `tuning_param_count` | integer | Number of parameters requiring human decision |
| `time_to_80pct_performance` | integer | Tuning iterations to reach 80% of best-known |
| `error_message_quality` | integer | 1–5 rating |
| `debugging_difficulty` | integer | 1–5 rating |
| `documentation_completeness` | integer | 1–5 rating |

### 10.4 Taxonomy Database (`data/taxonomy.json`)

```json
{
  "patterns": [
    {
      "id": "P001",
      "type": "failure",
      "name": "Launch Overhead Dominance",
      "symptom": "Abstraction 30–60% slower than native for kernels with execution time < 50 µs",
      "root_cause": "Extra function call layers not fully inlined; runtime dispatch overhead",
      "affected_workloads": ["fine-grained kernels", "irregular applications with many small launches"],
      "affected_platforms": ["all"],
      "mitigation": ["Use native for inner-loop hotspots", "Coarsen task granularity", "Enable aggressive inlining flags"],
      "evidence": [],
      "ppc_impact": "0.40–0.70 depending on kernel duration"
    }
  ]
}
```

---

## 11. Analysis Pipeline

The analysis pipeline runs in strict order. Each stage depends on the previous.

### Stage 1 — Parse Raw Results

**Script:** `scripts/parse/parse_results.py`  
**Input:** Raw benchmark output files in `results/`  
**Output:** Rows appended to `data/performance.csv`  
**Validation:** Run `scripts/parse/validate_schema.py` after every parse run

### Stage 2 — Compute Statistics

**Script:** `analysis/compute_ppc.py`  
**For each configuration (30 runs):** compute median, IQR, min, max  
**For each abstraction × kernel × size:** compute PPC across platforms  
**Flag:** any configuration where abstraction < 0.85 × native → add to profiling queue

### Stage 3 — Roofline Normalization

**Script:** `analysis/compute_roofline.py`  
**For each native configuration:** compute `utilization = measured / theoretical_peak`  
**Gate:** if native utilization < 0.80 on Large size → halt, fix environment, rerun

### Stage 4 — Overhead Attribution

**Script:** `analysis/overhead_attribution.py`  
**Input:** `data/profiling_metrics.csv` (flagged configurations only)  
**Output:** Overhead decomposition table per flagged configuration

### Stage 5 — Root Cause Analysis

**Script:** `analysis/root_cause_analysis.py`  
**Input:** Overhead attribution + generated code inspection results  
**Output:** Root cause label per flagged configuration (one of four taxonomy categories)

### Stage 6 — Taxonomy Construction

**Notebook:** `notebooks/04_taxonomy_construction.ipynb`  
**Input:** Root cause labels + performance data  
**Output:** Updated `data/taxonomy.json` with evidence-backed patterns

### Stage 7 — Decision Framework

**Script:** `analysis/decision_framework.py`  
**Input:** `data/taxonomy.json` + workload characteristics  
**Output:** Recommendation rules with empirically derived thresholds

### Stage 8 — Paper Figures

**Script:** `analysis/generate_plots.py`  
**Required figures per kernel:**
- Plot 1: Bandwidth/throughput vs. abstraction (per platform, with IQR error bars, peak reference line)
- Plot 2: Efficiency relative to native (per platform, per abstraction)
- Plot 3: PPC summary bar chart (per abstraction, grouped by problem size, with 0.60 and 0.80 reference lines)

---

## 12. Taxonomy Framework

### 12.1 Failure Mode Categories

Every root cause must be attributed to exactly one of these four categories:

| Category | Description | Detection Method |
|---|---|---|
| **Compiler Backend Failure** | Poor code generation: missed optimizations, bad unrolling, suboptimal register allocation | PTX/GCN inspection, compare with native generated code |
| **Runtime Coordination Overhead** | Extra synchronization, task scheduling costs, dispatch overhead | `nsys` timeline, API call analysis |
| **Memory Model Mismatch** | Abstraction forces non-optimal layout or extra indirection | L2 cache hit rate, memory access pattern analysis |
| **API Limitation** | Abstraction cannot express a required optimization (e.g., shared memory tiling) | Source code analysis, performance counter comparison |

### 12.2 Known Patterns (Hypotheses to Validate)

**Pattern 1 — Launch Overhead Dominance**
- Symptom: Abstraction 30–60% slower for kernels with execution time < 50 µs
- Root cause: Runtime Coordination Overhead
- Affected: Fine-grained kernels, irregular workloads with many small launches
- Platforms: All (cross-platform)
- Mitigation: Native for inner-loop hotspots; coarsen granularity; aggressive inlining

**Pattern 2 — Compiler Backend Unrolling Failure**
- Symptom: Abstraction 2–4× slower on NVIDIA for nested loops; matches on AMD
- Root cause: Compiler Backend Failure (PTX-specific)
- Affected: Multi-dimensional array access, complex indexing
- Platforms: NVIDIA-specific
- Mitigation: Manual loop unrolling; vendor-tuned library; compiler bug report
- Literature: Julia DGEMM on A100 (Godoy et al. 2023)

**Pattern 3 — Memory Layout Mismatch**
- Symptom: 20–40% bandwidth reduction vs. native
- Root cause: Memory Model Mismatch
- Affected: Multi-dimensional arrays with high data reuse
- Platforms: Platform-dependent (NVIDIA prefers LayoutLeft)
- Mitigation: Explicit layout specification in abstraction API
- Literature: Kokkos stencil; L2 hit rate drops from 75% to 45%

### 12.3 Success Pattern Template

For every abstraction that achieves PPC > 0.80, document:
- Workload characteristics that enabled success
- Why the compiler could optimize effectively
- Generalization rule for the decision framework

### 12.4 Pattern Template (for new patterns discovered during experiments)

```
Pattern Name:       [Descriptive identifier]
Type:               [failure | success]
Symptom:            [Observable performance characteristic]
Root Cause:         [Technical explanation — one of four categories]
Affected Workloads: [Computational characteristics that trigger this]
Affected Platforms: [Architecture-specific or cross-platform]
Mitigation:         [Actionable recommendations]
Evidence:           [experiment_id references from performance.csv]
PPC Impact:         [Expected PPC range when this pattern is active]
```

---

## 13. Decision Framework

### 13.1 Workload Characterization Inputs

Before a recommendation can be made, the following workload characteristics must be measured or estimated:

| Characteristic | Scale | How to Measure |
|---|---|---|
| Memory regularity | 0–1 | Stride variance analysis from profiler |
| Arithmetic intensity | FLOP/byte | Roofline analysis |
| Kernel duration | µs | Profiler timeline |
| Control flow divergence | 0–1 | Warp divergence counter from profiler |
| Data structure type | categorical | Static analysis |

### 13.2 Decision Logic

```
IF regularity > 0.8 AND divergence < 0.1:
    # Regular workload — abstractions likely to succeed
    IF kernel_duration > 100µs AND expected_PPC > 0.80:
        → FULL ABSTRACTION: Use Kokkos/RAJA/SYCL
    ELIF expected_PPC > 0.60:
        → ABSTRACTION WITH TUNING: Use framework, budget platform-specific tuning
    ELSE:
        → CAUTION: Kernel too short; launch overhead may dominate. Profile first.

ELIF regularity > 0.5 AND divergence < 0.3:
    # Semi-irregular — selective abstraction
    → HYBRID: Kokkos for data management + native CUDA/HIP for hot loops

ELSE:
    # Irregular — abstractions likely to fail
    IF task_runtime_applicable:
        → TASK RUNTIME: Consider PaRSEC/Legion for load balancing
    ELSE:
        → NATIVE: Platform-specific implementation with expert tuning

IF cross_platform_PPC_variance > 0.3:
    → WARNING: High variance. Consider platform-specific branches.
```

### 13.3 Empirical Thresholds (To Be Determined)

These are hypotheses. The experiment campaign will determine the actual values.

| Threshold | Hypothetical Value | Determination Method |
|---|---|---|
| Memory regularity cutoff | 0.8 | Cluster analysis of stride variance in success vs. failure cases |
| Minimum kernel duration | 100 µs | Plot overhead fraction vs. kernel time; find inflection point |
| PPC success threshold | 0.80 | Survey of production HPC codes: what PPC do they tolerate? |
| Control divergence limit | 0.3 | Correlation analysis: PPC vs. warp divergence |

---

## 14. Validation Plan

### 14.1 Retrospective Validation

**Objective:** Test whether the decision framework would have predicted actual abstraction choices made in production ECP codes.

**Protocol:**
1. Select 5–8 applications from Evans et al.'s 62 ECP codes (open-source, with both native and abstraction implementations)
2. Profile hotspot kernels to extract workload characteristics
3. Apply decision framework; record recommendation
4. Compare to actual developer choice
5. Analyze discrepancies; refine framework

**Target case studies:**
- ExaSMR (reactor simulation) — uses Kokkos extensively
- QMCPACK (quantum Monte Carlo) — uses CUDA+OpenMP hybrid
- LAMMPS (molecular dynamics) — has both Kokkos and native CUDA versions

### 14.2 Prospective Validation

**Objective:** Work with active application teams to apply the framework during development.

**Protocol:**
1. Recruit 2–3 application teams actively porting to new architectures
2. Profile their current implementation
3. Apply decision framework; provide recommendation
4. Team implements following recommendations
5. Measure outcomes: time-to-first-performance, time-to-target-performance, final PPC, developer satisfaction

---

## 15. Tool Architecture

The `abstraction-advisor` CLI tool is the deliverable that makes the decision framework usable by the community.

### 15.1 Components

**Component 1 — Workload Profiler**
- Lightweight instrumentation extracting workload characteristics
- Integrates with `nsys`, `rocprof`, VTune
- Outputs: regularity score, arithmetic intensity, kernel duration distribution → `workload_profile.json`

**Component 2 — Taxonomy Database**
- Structured storage of failure/success patterns from experiments
- Queryable by workload characteristics and target platforms
- Returns: expected PPC, known issues, mitigation strategies

**Component 3 — Recommendation Engine**
- Implements decision logic from §13.2
- Ingests `workload_profile.json` + target platforms
- Outputs: abstraction strategy with confidence level and rationale

### 15.2 CLI Interface

```
# Step 1: Profile the application
abstraction-advisor profile --app ./my_simulation --args "input.dat"
# → Outputs: workload_profile.json

# Step 2: Get recommendation
abstraction-advisor recommend \
  --profile workload_profile.json \
  --targets nvidia_a100,amd_mi250x \
  --priority portability
# → Outputs: Recommendation with rationale and expected PPC per kernel
```

### 15.3 Output Format

For each detected kernel, the tool outputs:
- Recommended strategy (FULL_ABSTRACTION / HYBRID / NATIVE / TASK_RUNTIME)
- Expected PPC with recommended strategy
- Known failure patterns that apply
- Mitigation strategies if applicable
- Confidence level (HIGH / MEDIUM / LOW) based on database coverage

---

## 16. Paper Structure

| Section | Title | Status | Key Content |
|---|---|---|---|
| 1 | Introduction | ✓ Written (`introduction.tex`) | Motivation, gap, contribution, paper map |
| 2 | Background | ✓ Written (`section2_background.tex`) | PPC definition, abstraction taxonomy, hardware landscape |
| 3 | Related Work | ✓ Written (needs gap bridge) | 4 thematic subsections, gap statement |
| 4 | Methodology | → In progress | Measurement protocol, experimental design, statistical approach |
| 5 | Experimental Results | → Pending data | Per-kernel results, PPC tables, roofline plots |
| 6 | Taxonomy | → Pending data | Failure/success patterns with evidence |
| 7 | Decision Framework | → Pending data | Rules, thresholds, validation |
| 8 | Validation | → Pending data | Retrospective + prospective validation |
| 9 | Conclusion | → Pending | Contributions, limitations, future work |

### 16.1 Required Figures (Full List)

| Figure | Content | Source |
|---|---|---|
| Fig 1 | Three-dimensional experimental matrix | Methodology |
| Fig 2 | Abstraction spectrum (Native → Julia) | Background |
| Fig 3–9 | Per-kernel: bandwidth/throughput vs. abstraction (3 platforms) | E1–E7 |
| Fig 10–16 | Per-kernel: efficiency relative to native | E1–E7 |
| Fig 17 | PPC summary across all kernels and abstractions | Analysis |
| Fig 18 | Overhead attribution breakdown (flagged configurations) | Deep profiling |
| Fig 19 | Taxonomy map: failure modes × workload types × platforms | Taxonomy |
| Fig 20 | Decision framework flowchart | Framework |
| Fig 21 | Tool validation: predicted vs. actual PPC | Validation |

---

## 17. Timeline and Milestones

| Month | Phase | Activities | Deliverables |
|---|---|---|---|
| 1 | **Setup** | Secure platform access, install full software stack, clone all benchmark suites, verify hardware state control | Working environments on all 3 platforms; baseline native runs passing |
| 2 | **Baseline** | Run all native implementations on all platforms and sizes; establish performance ceilings | `performance.csv` with native rows; roofline utilization ≥ 80% confirmed |
| 3–4 | **Abstraction Comparison** | Run Kokkos, RAJA, SYCL, Julia versions for all kernels; collect 9,450 timed runs | Complete performance dataset; initial PPC scores |
| 5 | **Deep Profiling** | Profile all flagged configurations (< 0.85 × native) with vendor tools | `profiling_metrics.csv`; overhead attribution tables |
| 6 | **Root Cause Analysis** | Inspect generated code; attribute root causes; identify failure mechanisms | Initial taxonomy: 8–12 patterns documented in `taxonomy.json` |
| 7 | **Framework Development** | Synthesize taxonomy into decision rules; determine empirical thresholds | Decision framework prototype; `decision_framework.py` |
| 8 | **Tool Development** | Implement `abstraction-advisor` CLI; validate on held-out kernels | Working prototype tool |
| 9 | **Validation** | Retrospective validation on ECP codes; prospective case study with application team | Validation results; framework refinement |
| 10 | **Dissemination** | Write remaining paper sections; prepare artifact release; create documentation | Manuscript submission; artifact package with DOI |

### 17.1 Critical Path

```
Month 1 (Platform access) → Month 2 (Native baseline) → Months 3-4 (Abstraction data)
→ Month 5 (Deep profiling) → Month 6 (Root cause) → Month 7 (Framework)
→ Month 8 (Tool) → Month 9 (Validation) → Month 10 (Paper)
```

Each phase is a hard dependency on the previous. If Month 2 native baselines do not pass the ≥ 80% utilization gate, Months 3–4 cannot begin.

---

## 18. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Platform access delays | Medium | High | Apply for allocations 3+ months early; use cloud as backup; prioritize NVIDIA (easiest access) for initial results |
| Benchmark implementation gaps | Low | Medium | Pre-survey available implementations before finalizing kernel list; drop abstraction-kernel combinations without existing code |
| Profiling tool failures | Low | Medium | Test profiling tools in Month 1; establish manual instrumentation fallback |
| Too much data / scope creep | Medium | Medium | Minimum viable matrix first: 3 kernels × 3 abstractions × 2 platforms = 18 experiments; expand only if time permits |
| Julia oneAPI.jl instability | High | Low | Mark Intel Julia results as partial/experimental; do not block paper on them |
| SYCL on AMD underperformance | Medium | Low | Expected and scientifically interesting; document as taxonomy pattern |
| Reproducibility failures | Low | High | Lock hardware state; use containers; archive all scripts with DOI before submission |

---

## 19. Resource Requirements

### 19.1 Computational Resources

| Resource | Estimate | Notes |
|---|---|---|
| Total GPU-hours | ~500 hours across 3 platforms | Includes experiments + profiling + reruns |
| NVIDIA A100 | ~150 hours | University cluster primary; AWS backup |
| AMD MI250X | ~200 hours | OLCF Frontier allocation |
| Intel PVC | ~150 hours | ALCF Aurora allocation |
| Cloud backup budget | $8,000–$12,000 | Seek academic credits (AWS, Azure, Google) |

### 19.2 Software (All Open Source)

- Kokkos, RAJA, SYCL implementations: open source, no license cost
- Vendor compilers and profilers: free for academic use
- Julia and GPU packages: open source
- No commercial software licenses required

### 19.3 Human Resources

| Role | Commitment | Responsibilities |
|---|---|---|
| Primary researcher | Full-time, 10 months | All experiments, analysis, paper writing, tool development |
| Advisor / senior collaborator | Part-time | Experimental design guidance, paper review |
| Undergraduate assistant (optional) | Part-time | Data collection automation, script testing |
| Application team collaborators | 2–3 teams | Prospective validation phase (Month 9) |

---

## 20. Artifact Release Plan

All artifacts will be released open-source with DOI assignment before paper submission.

### 20.1 Open-Source Components

| Artifact | Repository | License |
|---|---|---|
| `abstraction-advisor` CLI tool | GitHub | BSD/MIT |
| Complete experimental dataset | Zenodo | CC BY 4.0 |
| Reproducibility package (containers + scripts) | GitHub + Zenodo | BSD/MIT |
| Interactive taxonomy browser | GitHub Pages | BSD/MIT |
| All paper figures (source data) | Zenodo | CC BY 4.0 |

### 20.2 Reproducibility Requirements

The reproducibility package must allow an independent researcher to:

1. Reproduce any single experiment result within 10% of reported values
2. Recompute all PPC scores from raw data
3. Regenerate all paper figures from raw data
4. Run the `abstraction-advisor` tool on a new workload

### 20.3 Community Contribution Pathway

- Accept pull requests for new workload characterizations
- Crowdsource additional failure/success patterns via GitHub Issues
- Maintain a living taxonomy database that grows with community contributions
- Publish updated dataset versions with DOI increments

---

## Appendix A — PPC Worked Example

Given:
- Abstraction: Kokkos
- Kernel: STREAM Triad, Large size
- Platforms: A100, MI250X, PVC

| Platform | Native GB/s | Kokkos GB/s | Efficiency |
|---|---|---|---|
| A100 | 1,500 | 1,450 | 0.967 |
| MI250X | 1,700 | 1,650 | 0.971 |
| PVC | 1,100 | 980 | 0.891 |

```
PPC = 3 / (1/0.967 + 1/0.971 + 1/0.891)
PPC = 3 / (1.034 + 1.030 + 1.122)
PPC = 3 / 3.186
PPC = 0.942  → Excellent portability ✓
```

---

## Appendix B — Deep Profiling Trigger Checklist

When a configuration is flagged (abstraction < 0.85 × native), work through this checklist in order:

- [ ] Confirm hardware state was locked during the run
- [ ] Rerun the configuration to confirm the gap is reproducible
- [ ] Run `nsys` / `rocprof` for timeline analysis
- [ ] Run `ncu` / `omniperf` for kernel-level counters
- [ ] Extract and inspect generated PTX / GCN code
- [ ] Compare generated code side-by-side with native
- [ ] Attribute root cause to one of four taxonomy categories
- [ ] Add entry to `data/taxonomy.json` with evidence reference
- [ ] Document mitigation strategy

---

## Appendix C — Experiment Readiness Checklist

Before starting any experiment group, verify:

- [ ] All software versions recorded in experiment log
- [ ] GPU clocks locked and verified
- [ ] Native implementation achieves ≥ 80% of theoretical peak (Large size)
- [ ] Warm-up protocol confirmed (10 runs before timing)
- [ ] 30 timed runs configured
- [ ] Output parsing script tested on sample output
- [ ] `validate_schema.py` passes on test CSV
- [ ] Results directory exists and is writable
- [ ] Git commit hash recorded for all benchmark source code

---

*End of Project Specification v1.0*
