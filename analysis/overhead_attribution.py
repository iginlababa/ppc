#!/usr/bin/env python3
"""
Stage 4: Overhead attribution.

Decomposes total abstraction overhead into four categories:
  1. Kernel launch latency
  2. Synchronization overhead
  3. Host-side framework overhead
  4. Memory transfer overhead
  5. Compiler code quality (inferred from kernel_time delta)

Input: profiling_metrics.csv (flagged configurations only — output of compute_ppc.py)
Output: overhead_breakdown.csv

Usage:
    python analysis/overhead_attribution.py \
        --perf data/performance.csv \
        --profiling data/profiling_metrics.csv \
        --output data/processed/overhead_breakdown.csv \
        [--experiment E1] [--platform nvidia_a100]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


OVERHEAD_CATEGORIES = [
    "kernel_launch_ms",
    "synchronization_ms",
    "host_framework_ms",
    "memory_transfer_ms",
    "compiler_quality_ms",  # inferred: kernel_time_delta - accounted overheads
]


def load_and_merge(perf_path: Path, prof_path: Path) -> pd.DataFrame:
    perf = pd.read_csv(perf_path, parse_dates=["timestamp"])
    prof = pd.read_csv(prof_path)
    merged = perf.merge(prof, on="experiment_id", how="inner")
    return merged


def attribute_overhead(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overhead components for each flagged configuration.

    All times in milliseconds. Native rows serve as baseline.
    """
    results = []

    group_keys = ["kernel", "platform", "problem_size"]
    for group, grp in df.groupby(group_keys):
        native = grp[grp["abstraction"] == "native"]
        if native.empty:
            continue
        native_kernel_time = native["kernel_time_ms"].median()
        native_total_time = native["execution_time_ms"].median()

        for _, row in grp[grp["abstraction"] != "native"].iterrows():
            total_overhead = row["execution_time_ms"] - native_total_time
            kernel_delta = row["kernel_time_ms"] - native_kernel_time

            launch = row.get("launch_overhead_ms", 0.0)
            memory = max(0.0, row.get("memory_transfer_mb", 0.0) * 0.001)  # rough estimate

            # Host framework = total - kernel - launch - memory
            host_framework = max(0.0, total_overhead - kernel_delta - launch - memory)

            # Compiler quality = unaccounted kernel time delta
            compiler_quality = max(0.0, kernel_delta - launch)

            results.append({
                "kernel": row["kernel"],
                "abstraction": row["abstraction"],
                "platform": row["platform"],
                "problem_size": row["problem_size"],
                "total_overhead_ms": round(total_overhead, 4),
                "kernel_launch_ms": round(launch, 4),
                "host_framework_ms": round(host_framework, 4),
                "memory_transfer_ms": round(memory, 4),
                "compiler_quality_ms": round(compiler_quality, 4),
                "dominant_category": _dominant(launch, host_framework, memory, compiler_quality),
            })

    return pd.DataFrame(results)


def _dominant(launch, host, memory, compiler) -> str:
    vals = {
        "kernel_launch": launch,
        "host_framework": host,
        "memory_transfer": memory,
        "compiler_quality": compiler,
    }
    return max(vals, key=vals.get)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf", required=True, help="data/performance.csv")
    parser.add_argument("--profiling", required=True, help="data/profiling_metrics.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--platform")
    args = parser.parse_args()

    df = load_and_merge(Path(args.perf), Path(args.profiling))
    if args.platform:
        df = df[df["platform"] == args.platform]

    breakdown = attribute_overhead(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(out_path, index=False)
    print(f"Overhead breakdown written to {out_path}  ({len(breakdown)} rows)")

    if not breakdown.empty:
        print("\nDominant overhead categories:")
        print(breakdown["dominant_category"].value_counts())

    return 0


if __name__ == "__main__":
    sys.exit(main())
