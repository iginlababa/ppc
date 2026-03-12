#!/usr/bin/env python3
"""
Stage 5: Root cause analysis.

For every flagged configuration, assign one of the four taxonomy categories:
  1. Compiler Backend Failure
  2. Runtime Coordination Overhead
  3. Memory Model Mismatch
  4. API Limitation

Uses heuristic rules based on overhead breakdown + profiling counters.
Manual override via --annotations CSV is supported for cases requiring
PTX/GCN inspection.

Usage:
    python analysis/root_cause_analysis.py \
        --overhead data/processed/overhead_breakdown.csv \
        --profiling data/profiling_metrics.csv \
        --output data/processed/root_cause_labels.csv \
        [--annotations data/processed/manual_annotations.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Four taxonomy categories — project_spec.md §12.1
CATEGORIES = {
    "compiler_backend_failure": (
        "Poor code generation: missed optimizations, bad unrolling, "
        "suboptimal register allocation"
    ),
    "runtime_coordination_overhead": (
        "Extra synchronization, task scheduling costs, dispatch overhead"
    ),
    "memory_model_mismatch": (
        "Abstraction forces non-optimal layout or extra indirection"
    ),
    "api_limitation": (
        "Abstraction cannot express a required optimization"
    ),
}


def classify(row: pd.Series) -> str:
    """
    Heuristic classification. Covers common cases; manual annotation
    required for corner cases (see --annotations flag).

    Rules (applied in priority order):
    1. If memory_transfer_ms dominates AND L2 hit rate dropped → memory_model_mismatch
    2. If host_framework_ms dominates → runtime_coordination_overhead
    3. If compiler_quality_ms dominates → compiler_backend_failure
    4. If kernel_launch_ms dominates → runtime_coordination_overhead
    5. Default → compiler_backend_failure (inspect PTX to confirm)
    """
    dominant = row.get("dominant_category", "")
    l2_drop = row.get("l2_hit_rate_delta", 0.0)  # native - abstraction; positive = drop

    if dominant == "memory_transfer" or (l2_drop is not None and l2_drop > 0.15):
        return "memory_model_mismatch"
    if dominant == "host_framework":
        return "runtime_coordination_overhead"
    if dominant == "kernel_launch":
        return "runtime_coordination_overhead"
    if dominant == "compiler_quality":
        return "compiler_backend_failure"
    return "compiler_backend_failure"  # default — requires PTX confirmation


def apply_annotations(df: pd.DataFrame, ann_path: Path) -> pd.DataFrame:
    """Override heuristic labels with manually verified root causes."""
    ann = pd.read_csv(ann_path)
    key = ["kernel", "abstraction", "platform", "problem_size"]
    df = df.merge(ann[key + ["manual_root_cause"]], on=key, how="left")
    df["root_cause"] = df["manual_root_cause"].fillna(df["root_cause_heuristic"])
    df["root_cause_source"] = df["manual_root_cause"].notna().map(
        {True: "manual", False: "heuristic"}
    )
    df = df.drop(columns=["manual_root_cause"])
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overhead", required=True)
    parser.add_argument("--profiling", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", help="CSV with manual root cause overrides")
    args = parser.parse_args()

    overhead = pd.read_csv(Path(args.overhead))
    profiling = pd.read_csv(Path(args.profiling))

    key = ["kernel", "abstraction", "platform", "problem_size"]
    # Merge in L2 hit rate info if available
    # TODO: compute l2_hit_rate_delta = native_l2 - abstraction_l2

    overhead["root_cause_heuristic"] = overhead.apply(classify, axis=1)

    if args.annotations:
        result = apply_annotations(overhead, Path(args.annotations))
    else:
        overhead["root_cause"] = overhead["root_cause_heuristic"]
        overhead["root_cause_source"] = "heuristic"
        result = overhead

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Root cause labels written to {out_path}  ({len(result)} rows)")

    print("\nRoot cause distribution:")
    print(result["root_cause"].value_counts())

    heuristic_count = (result["root_cause_source"] == "heuristic").sum()
    print(f"\nManually verified: {len(result) - heuristic_count}/{len(result)}")
    if heuristic_count > 0:
        print("WARNING: Heuristic labels require PTX/GCN inspection to confirm.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
