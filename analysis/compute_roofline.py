#!/usr/bin/env python3
"""
Stage 3: Roofline normalization.

Computes hardware utilization = measured_throughput / theoretical_peak
for every configuration in performance.csv.

Gate rule: if native utilization < 0.80 on Large problem size,
           halt and report — environment setup is incorrect.

Usage:
    python analysis/compute_roofline.py \
        --input data/performance.csv \
        --output data/processed/roofline.csv \
        [--experiment E1] [--platform nvidia_a100]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


# Peak hardware values from project_spec.md §5.2 and §9.2
PEAK_BW_GBS = {
    "nvidia_rtx5060_laptop":        288.0,
    "nvidia_rtx5060_laptop_locked": 384.0,   # GDDR7 12001 MHz × 128-bit × 2
    "nvidia_a100":                  2039.0,
    "amd_mi250x":                   3277.0,
    "amd_mi300x":                   5300.0,   # HBM3 theoretical peak
    "intel_pvc":                    3276.0,
}

PEAK_FLOPS_GFLOPS = {
    "nvidia_rtx5060_laptop":        1700.0,   # FP64 ~1.7 TFLOP/s (RTX 5060 Laptop)
    "nvidia_rtx5060_laptop_locked": 1700.0,
    "nvidia_a100":                  19500.0,
    "amd_mi250x":                   47900.0,
    "amd_mi300x":                   163400.0, # FP64 ~163.4 TFLOP/s (MI300X)
    "intel_pvc":                    22200.0,
}

# Bandwidth-bound kernels — primary metric is GB/s
BW_BOUND_KERNELS = {"stream"}
# Compute-bound kernels — primary metric is GFLOP/s
COMPUTE_BOUND_KERNELS = {"dgemm"}
# Mixed — report both
MIXED_KERNELS = {"stencil", "spmv", "sptrsv", "bfs", "nbody"}

NATIVE_UTILIZATION_GATE = 0.80  # must achieve ≥ 80% on Large size


def peak_for_kernel(kernel: str, platform: str) -> float:
    """Return the relevant theoretical peak for this kernel/platform pair."""
    if kernel in BW_BOUND_KERNELS:
        return PEAK_BW_GBS.get(platform, float("nan"))
    if kernel in COMPUTE_BOUND_KERNELS:
        return PEAK_FLOPS_GFLOPS.get(platform, float("nan"))
    # Mixed — use bandwidth peak as conservative bound
    return PEAK_BW_GBS.get(platform, float("nan"))


def compute_utilization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["theoretical_peak"] = df.apply(
        lambda r: peak_for_kernel(r["kernel"], r["platform"]), axis=1
    )
    df["utilization"] = df["throughput"] / df["theoretical_peak"]
    return df


def check_native_gate(df: pd.DataFrame) -> list[str]:
    """Return list of (platform, kernel) pairs that fail the ≥ 80% gate."""
    native_large = df[
        (df["abstraction"] == "native") & (df["problem_size"] == "large")
    ]
    failures = native_large[native_large["utilization"] < NATIVE_UTILIZATION_GATE]
    return [
        f"  kernel={r.kernel}, platform={r.platform}, utilization={r.utilization:.3f}"
        for _, r in failures.iterrows()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--platform")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip the ≥ 80% native utilization gate check",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input), parse_dates=["timestamp"])

    if args.platform:
        df = df[df["platform"] == args.platform]

    # Aggregate to medians first
    group_keys = ["kernel", "abstraction", "platform", "problem_size"]
    agg = (
        df.groupby(group_keys)["throughput"]
        .median()
        .reset_index()
        .rename(columns={"throughput": "throughput_median"})
    )
    agg = agg.rename(columns={"throughput_median": "throughput"})

    result = compute_utilization(agg)

    if not args.no_gate:
        failures = check_native_gate(result)
        if failures:
            print("ERROR: Native baseline fails ≥ 80% utilization gate on Large size:")
            for line in failures:
                print(line)
            print("Fix environment setup before running abstraction experiments.")
            return 1
        else:
            print("Native utilization gate: PASSED (all native Large >= 80%)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Roofline data written to {out_path}  ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
