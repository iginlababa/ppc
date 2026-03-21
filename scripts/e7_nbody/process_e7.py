#!/usr/bin/env python3
"""
process_e7.py — E7 N-Body: raw CSV → summary CSV.

Reads:  results/e7_nbody/raw/e7_nbody_*.csv  (all matching files)
Writes: data/processed/e7_nbody_summary.csv

Summary columns per (platform, abstraction, kernel, problem_size):
  n_atoms, n_nbrs_total, max_nbrs_per_atom, mean_nbrs_per_atom,
  median_gflops, p25_gflops, p75_gflops, iqr_gflops,
  median_time_ms, p25_time_ms, p75_time_ms,
  actual_flops, ai_flop_byte,
  efficiency_vs_native_notile,
  tier (excellent ≥0.95, acceptable ≥0.85, marginal ≥0.70, poor <0.70),
  hw_state_verified

Efficiency is computed as:
  median_gflops(abstraction) / median_gflops(native_notile, same size)
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR    = os.path.join(REPO_ROOT, "results", "e7_nbody", "raw")
PROC_DIR   = os.path.join(REPO_ROOT, "data", "processed")
OUT_CSV    = os.path.join(PROC_DIR, "e7_nbody_summary.csv")
os.makedirs(PROC_DIR, exist_ok=True)

FLOPS_PER_PAIR = 20

# Per-platform config: native abstraction label used as efficiency baseline
PLATFORM_CONFIGS = {
    # platform_prefix : native_abstraction_label
    "nvidia": "native_notile",
    "amd":    "hip",
    "intel":  "native_notile",
}


def _native_key(platform: str) -> str:
    for prefix, key in PLATFORM_CONFIGS.items():
        if platform.startswith(prefix):
            return key
    return "native_notile"

# Efficiency tiers (PPC-equivalent thresholds)
TIER_EXCELLENT  = 0.95
TIER_ACCEPTABLE = 0.85
TIER_MARGINAL   = 0.70


def assign_tier(eff: float) -> str:
    if eff >= TIER_EXCELLENT:   return "excellent"
    if eff >= TIER_ACCEPTABLE:  return "acceptable"
    if eff >= TIER_MARGINAL:    return "marginal"
    return "poor"


def compute_ai(row: pd.Series) -> float:
    """
    Arithmetic intensity: FLOP / byte.
    FLOPs  = n_nbrs_total × 20 (per rep, actual pairs processed)
    Bytes  = n_nbrs_total × 16 (read float4 pos_j) + n_atoms × 16 (read pos_i) + n_atoms × 12 (write force)
    """
    N = int(row["n_atoms"])
    T = int(row["n_nbrs_total"])
    flops = T * FLOPS_PER_PAIR
    bytes_ = T * 16.0 + N * 16.0 + N * 12.0
    return flops / bytes_ if bytes_ > 0 else float("nan")


def main():
    patterns = glob.glob(os.path.join(RAW_DIR, "e7_nbody_*.csv"))
    if not patterns:
        print(f"[process_e7] ERROR: no raw CSVs found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for p in sorted(patterns):
        print(f"[process_e7] Reading {p}")
        dfs.append(pd.read_csv(p))
    raw = pd.concat(dfs, ignore_index=True)
    print(f"[process_e7] Total rows: {len(raw)}")

    # ── Compute per-group stats ───────────────────────────────────────────────
    group_cols = ["platform", "abstraction", "kernel", "problem_size"]
    meta_cols  = ["n_atoms", "n_nbrs_total", "max_nbrs_per_atom", "mean_nbrs_per_atom"]

    records = []
    for key, grp in raw.groupby(group_cols):
        platform, abstraction, kernel, problem_size = key
        meta = {c: grp[c].iloc[0] for c in meta_cols}

        t   = grp["time_ms"].values.astype(float)
        gf  = grp["throughput_gflops"].values.astype(float)
        flops = grp["actual_flops"].iloc[0]
        hw    = int(grp["hw_state_verified"].iloc[0])

        p25_t, med_t, p75_t = np.percentile(t,  [25, 50, 75])
        p25_g, med_g, p75_g = np.percentile(gf, [25, 50, 75])

        ai = compute_ai(pd.Series(meta))

        rec = dict(
            platform          = platform,
            abstraction       = abstraction,
            kernel            = kernel,
            problem_size      = problem_size,
            **meta,
            median_gflops     = med_g,
            p25_gflops        = p25_g,
            p75_gflops        = p75_g,
            iqr_gflops        = p75_g - p25_g,
            median_time_ms    = med_t,
            p25_time_ms       = p25_t,
            p75_time_ms       = p75_t,
            actual_flops      = flops,
            ai_flop_byte      = ai,
            hw_state_verified = hw,
        )
        records.append(rec)

    summary = pd.DataFrame(records)

    # ── Efficiency relative to platform native baseline ───────────────────────
    # Build per-platform baseline (native_notile for NVIDIA, hip for AMD)
    baseline_rows = []
    for plat, grp in summary.groupby("platform"):
        nat = _native_key(plat)
        sub = grp[grp["abstraction"] == nat][["platform", "problem_size", "median_gflops"]]
        baseline_rows.append(sub)
    if baseline_rows:
        baseline = pd.concat(baseline_rows).rename(columns={"median_gflops": "baseline_gflops"})
    else:
        baseline = pd.DataFrame(columns=["platform", "problem_size", "baseline_gflops"])
    summary = summary.merge(baseline, on=["platform", "problem_size"], how="left")
    summary["efficiency_vs_native_notile"] = (
        summary["median_gflops"] / summary["baseline_gflops"]
    )
    summary["tier"] = summary["efficiency_vs_native_notile"].apply(assign_tier)
    summary.drop(columns=["baseline_gflops"], inplace=True)

    # ── Size ordering ─────────────────────────────────────────────────────────
    size_order = {"small": 0, "medium": 1, "large": 2}
    summary["_size_ord"] = summary["problem_size"].map(size_order).fillna(99)
    summary.sort_values(["platform", "abstraction", "kernel", "_size_ord"], inplace=True)
    summary.drop(columns=["_size_ord"], inplace=True)

    summary.to_csv(OUT_CSV, index=False)
    print(f"[process_e7] Summary rows: {len(summary)}")
    print(f"[process_e7] Written: {OUT_CSV}")

    # ── Quick table ───────────────────────────────────────────────────────────
    print("\n[process_e7] Performance summary (median GFLOP/s, efficiency):")
    print(f"{'abstraction':<18} {'kernel':<8} {'size':<8} "
          f"{'median GFLOP/s':>14}  {'eff':>6}  {'tier'}")
    print("-" * 72)
    for _, row in summary.iterrows():
        eff = row["efficiency_vs_native_notile"]
        print(f"{row['abstraction']:<18} {row['kernel']:<8} {row['problem_size']:<8} "
              f"{row['median_gflops']:>14.1f}  {eff:>6.3f}  {row['tier']}")


if __name__ == "__main__":
    main()
