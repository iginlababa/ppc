#!/usr/bin/env python3
"""
Schema validation for data/performance.csv.

Checks:
  1. All required columns present
  2. No null values in mandatory fields
  3. Valid enum values (abstraction, platform, problem_size)
  4. throughput > 0 for all rows
  5. run_id in 1–30 range
  6. Experiment IDs are unique

Run after every parse_results.py call.

Usage:
    python scripts/parse/validate_schema.py --input data/performance.csv
    python scripts/parse/validate_schema.py --input data/performance.csv --strict
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp", "experiment_id", "kernel", "abstraction", "platform",
    "problem_size", "problem_size_n", "run_id", "execution_time_ms",
    "throughput", "hardware_state_verified",
]

VALID_KERNELS      = {"stream", "dgemm", "stencil", "spmv", "sptrsv", "bfs", "nbody"}
VALID_ABSTRACTIONS = {"native", "kokkos", "raja", "sycl", "julia"}
VALID_PLATFORMS    = {"nvidia_a100", "amd_mi250x", "intel_pvc"}
VALID_SIZES        = {"small", "medium", "large"}


def validate(df: pd.DataFrame, strict: bool = False) -> list[str]:
    errors = []
    warnings = []

    # 1. Required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors  # can't continue without columns

    # 2. Null checks on mandatory fields
    for col in ["experiment_id", "kernel", "abstraction", "platform",
                "problem_size", "run_id", "throughput"]:
        nulls = df[col].isna().sum()
        if nulls > 0:
            errors.append(f"Column '{col}' has {nulls} null values")

    # 3. Enum validation
    bad_kernels = ~df["kernel"].isin(VALID_KERNELS)
    if bad_kernels.any():
        bad_vals = df.loc[bad_kernels, "kernel"].unique().tolist()
        errors.append(f"Invalid kernel values: {bad_vals}")

    bad_abs = ~df["abstraction"].isin(VALID_ABSTRACTIONS)
    if bad_abs.any():
        bad_vals = df.loc[bad_abs, "abstraction"].unique().tolist()
        errors.append(f"Invalid abstraction values: {bad_vals}")

    bad_plat = ~df["platform"].isin(VALID_PLATFORMS)
    if bad_plat.any():
        bad_vals = df.loc[bad_plat, "platform"].unique().tolist()
        errors.append(f"Invalid platform values: {bad_vals}")

    bad_size = ~df["problem_size"].isin(VALID_SIZES)
    if bad_size.any():
        bad_vals = df.loc[bad_size, "problem_size"].unique().tolist()
        errors.append(f"Invalid problem_size values: {bad_vals}")

    # 4. Throughput > 0
    non_pos = df["throughput"].le(0)
    if non_pos.any():
        errors.append(f"{non_pos.sum()} rows have throughput <= 0")

    # 5. run_id range
    if "run_id" in df.columns:
        bad_run = ~df["run_id"].between(1, 30)
        if bad_run.any():
            warnings.append(f"{bad_run.sum()} rows have run_id outside 1–30 range")

    # 6. Unique experiment IDs
    dupes = df["experiment_id"].duplicated().sum()
    if dupes > 0:
        errors.append(f"{dupes} duplicate experiment_id values")

    if strict:
        errors.extend(warnings)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return 1

    df = pd.read_csv(path)
    if df.empty:
        print(f"WARNING: {path} is empty (header only) — nothing to validate")
        return 0

    errors = validate(df, strict=args.strict)

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"VALIDATION PASSED — {len(df)} rows, {df['kernel'].nunique()} kernels, "
          f"{df['platform'].nunique()} platforms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
