#!/usr/bin/env python3
"""
Stage 2: Compute Performance Portability Coefficient (PPC).

PPC(a, p) = |H| / sum_{h in H} (1 / E_h)
where E_h = performance_abstraction_h / performance_native_h

Pennycook et al. (2019) formulation. Efficiency is always relative to
measured native performance, not theoretical peak.

Usage:
    python analysis/compute_ppc.py \
        --input data/performance.csv \
        --output data/processed/ppc_results.csv \
        [--experiment E1] [--abstraction kokkos]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# PPC thresholds from project_spec.md §9.4
PPC_EXCELLENT = 0.80
PPC_ACCEPTABLE = 0.60
DEEP_PROFILE_TRIGGER = 0.85  # flag if abstraction < 0.85 * native


def load_performance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required_cols = {
        "experiment_id", "kernel", "abstraction", "platform",
        "problem_size", "run_id", "throughput",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in performance CSV: {missing}")
    return df


def compute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30 per-run rows to one median row per configuration."""
    group_keys = ["kernel", "abstraction", "platform", "problem_size"]
    agg = (
        df.groupby(group_keys)["throughput"]
        .agg(
            throughput_median="median",
            throughput_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
            throughput_min="min",
            throughput_max="max",
            n_runs="count",
        )
        .reset_index()
    )
    return agg


def compute_efficiency(agg: pd.DataFrame) -> pd.DataFrame:
    """Divide each abstraction's throughput by the native baseline on the same platform."""
    native = agg[agg["abstraction"] == "native"][
        ["kernel", "platform", "problem_size", "throughput_median"]
    ].rename(columns={"throughput_median": "native_throughput"})

    merged = agg.merge(native, on=["kernel", "platform", "problem_size"], how="left")
    merged["efficiency"] = merged["throughput_median"] / merged["native_throughput"]
    return merged


def compute_ppc(eff: pd.DataFrame) -> pd.DataFrame:
    """Compute PPC for each abstraction × kernel × problem_size combination."""
    results = []
    for (kernel, abstraction, problem_size), grp in eff.groupby(
        ["kernel", "abstraction", "problem_size"]
    ):
        if abstraction == "native":
            continue

        valid = grp.dropna(subset=["efficiency"])
        if valid.empty:
            continue

        H = len(valid)
        ppc = H / (1.0 / valid["efficiency"]).sum()

        results.append(
            {
                "kernel": kernel,
                "abstraction": abstraction,
                "problem_size": problem_size,
                "ppc": round(ppc, 4),
                "n_platforms": H,
                "platforms": ",".join(sorted(valid["platform"].tolist())),
                "ppc_tier": _ppc_tier(ppc),
                "min_efficiency": round(valid["efficiency"].min(), 4),
                "max_efficiency": round(valid["efficiency"].max(), 4),
            }
        )

    return pd.DataFrame(results)


def _ppc_tier(ppc: float) -> str:
    if ppc >= PPC_EXCELLENT:
        return "excellent"
    if ppc >= PPC_ACCEPTABLE:
        return "acceptable"
    return "poor"


def flag_for_profiling(eff: pd.DataFrame) -> pd.DataFrame:
    """Return rows where abstraction < DEEP_PROFILE_TRIGGER * native."""
    flagged = eff[
        (eff["abstraction"] != "native")
        & (eff["efficiency"] < DEEP_PROFILE_TRIGGER)
    ].copy()
    flagged["profiling_reason"] = (
        "efficiency=" + flagged["efficiency"].round(3).astype(str)
        + " < " + str(DEEP_PROFILE_TRIGGER)
    )
    return flagged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to data/performance.csv")
    parser.add_argument("--output", required=True, help="Path for ppc_results.csv")
    parser.add_argument("--experiment", help="Filter to a single experiment ID (e.g. E1)")
    parser.add_argument("--abstraction", help="Filter to a single abstraction")
    parser.add_argument(
        "--profiling-queue",
        default="data/processed/profiling_queue.csv",
        help="Path to write configurations flagged for deep profiling",
    )
    args = parser.parse_args()

    df = load_performance(Path(args.input))

    if args.experiment:
        # Map experiment IDs to kernel names via config lookup
        # TODO: load from config.yaml; for now use a simple filter on kernel column
        pass

    if args.abstraction:
        df = df[df["abstraction"].isin(["native", args.abstraction])]

    agg = compute_medians(df)
    eff = compute_efficiency(agg)
    ppc_df = compute_ppc(eff)
    flagged = flag_for_profiling(eff)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ppc_df.to_csv(out_path, index=False)
    print(f"PPC results written to {out_path}  ({len(ppc_df)} rows)")

    queue_path = Path(args.profiling_queue)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(queue_path, index=False)
    print(f"Profiling queue written to {queue_path}  ({len(flagged)} configs flagged)")

    # Print summary to stdout
    if not ppc_df.empty:
        print("\nPPC Summary:")
        print(ppc_df.groupby(["abstraction", "ppc_tier"])["ppc"].agg(["mean", "min", "max"]).round(3))

    return 0


if __name__ == "__main__":
    sys.exit(main())
