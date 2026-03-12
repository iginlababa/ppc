# Reproducibility Guide

Step-by-step instructions for an independent researcher to reproduce all results.

## Prerequisites

- Linux system (Ubuntu 22.04+ or RHEL 8+)
- Access to at least one supported GPU (NVIDIA A100, AMD MI250X, or Intel PVC)
- Conda or Mamba installed
- Git with LFS support

---

## 1. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd measurement-driven-abstraction

# Create conda environment
conda env create -f environment.yml
conda activate hpc-abstraction

# Copy and configure environment variables
cp .env.example .env
# Edit .env to match your platform
```

## 2. Platform-Specific Software Stack

```bash
# NVIDIA
source scripts/env/setup_nvidia.sh

# AMD
source scripts/env/setup_amd.sh

# Intel
source scripts/env/setup_intel.sh
```

## 3. Clone Benchmark Suites

```bash
git clone https://github.com/UoB-HPC/BabelStream benchmarks/stream/BabelStream
git clone https://github.com/LLNL/RAJAPerf       benchmarks/dgemm/RAJAPerf
git clone https://github.com/sbeamer/gapbs        benchmarks/bfs/GAP
git clone https://github.com/ECP-copa/CoMD        benchmarks/nbody/CoMD
```

## 4. Build All Abstraction Variants

```bash
# Build a single experiment
./scripts/build/build_stream.sh --platform nvidia_a100

# Build all experiments
for exp in stream dgemm stencil spmv sptrsv bfs nbody; do
    ./scripts/build/build_${exp}.sh --platform nvidia_a100
done
```

## 5. Verify Native Baselines (≥ 80% peak required)

```bash
./scripts/run/run_stream.sh   --abstraction native --platform nvidia_a100 --size large
python analysis/compute_roofline.py --experiment E1 --platform nvidia_a100
# Must report utilization >= 0.80 before proceeding
```

## 6. Run Full Experiment Suite

```bash
# Single experiment, all abstractions, all sizes, 30 reps
./scripts/run/run_stream.sh --platform nvidia_a100 --reps 30

# Full sweep (all 7 experiments)
python scripts/run/run_experiments.py --platform nvidia_a100 --reps 30
```

## 7. Parse and Validate Results

```bash
python scripts/parse/parse_results.py \
    --results-dir results/nvidia_a100/ \
    --output data/performance.csv

python scripts/parse/validate_schema.py --input data/performance.csv
```

## 8. Reproduce Analysis

```bash
# Compute PPC scores
python analysis/compute_ppc.py \
    --input data/performance.csv \
    --output data/processed/ppc_results.csv

# Overhead attribution (requires profiling data)
python analysis/overhead_attribution.py \
    --experiment E1 --platform nvidia_a100

# Generate all paper figures
python analysis/generate_plots.py \
    --input data/processed/ \
    --output paper/figures/
```

## 9. Reproduce the Paper

```bash
cd paper && latexmk -pdf main.tex
```

---

## Tolerances

Results reproduced from this artifact should match published values within:
- Performance metrics (GB/s, GFLOP/s): ±10%
- PPC scores: ±0.05

Larger deviations indicate a hardware or configuration mismatch. Verify §5.3 hardware state control.

## Archived Dataset

The complete raw dataset is archived at: TODO — DOI assigned at submission
