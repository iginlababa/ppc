#!/usr/bin/env python3
"""
E1 STREAM Triad — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs, removes warm-up,
computes statistics and PPC, saves data/processed/e1_stream_summary.csv.

Native baseline note
--------------------
The native CSV contains two separate invocations of run_stream.sh:
  Batch 0 (00:38 UTC): GPU cold → boost clocks → ~350 GB/s (runs 1-28 clean)
  Batch 1 (01:41 UTC): GPU warm → throttled    → ~272 GB/s (runs 1-18 clean)

Kokkos / Julia / Numba ran right after batch 0 while the GPU was still in
the same boost-clock thermal state: their hw_state=1 runs cluster at ~350 GB/s.
We therefore use batch 0 as the canonical native baseline for PPC computation
so that all abstractions are compared within the same thermal regime.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(REPO_ROOT, "data", "raw")
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
PLATFORM     = "nvidia_rtx5060_laptop"
PEAK_BW_GBS  = 288.0   # GDDR7 128-bit theoretical peak (config.yaml)
WARMUP_DROP  = 5       # discard first N run_ids per batch
ABSTRACTIONS = ["native", "kokkos", "julia", "numba"]

# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e1_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"stream_{abs_name}_{PLATFORM}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no CSV found for abstraction={abs_name}", file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"  {abs_name:8s}: {len(df):3d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW path")
    return pd.concat(frames, ignore_index=True)


# ── Batch detection ───────────────────────────────────────────────────────────
# Sort by (abstraction, timestamp, run_id) so that runs within the same
# second are ordered by run_id.  A new batch begins whenever run_id
# resets to a value ≤ the previous run_id.
def assign_batch_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["abstraction", "timestamp", "run_id"]).reset_index(drop=True)
    batch_col = []
    for _, grp in df.groupby("abstraction", sort=False):
        bid = 0
        prev_rid = None
        for rid in grp["run_id"]:
            if prev_rid is not None and rid <= prev_rid:
                bid += 1
            batch_col.append(bid)
            prev_rid = rid
    df["batch_id"] = batch_col
    return df


# ── Native baseline selection ─────────────────────────────────────────────────
# Use the FIRST (batch_id=0) native batch as the canonical baseline.
# Batch 0 represents the GPU boost-clock state (~350 GB/s), which is the
# same thermal regime in which Kokkos/Julia/Numba were measured.
def select_native_baseline(df: pd.DataFrame) -> pd.DataFrame:
    native_mask = df["abstraction"] == "native"
    non_native  = df[~native_mask].copy()

    native_df    = df[native_mask].copy()
    native_batch0 = native_df[native_df["batch_id"] == 0].copy()

    n_dropped = native_mask.sum() - len(native_batch0)
    print(f"  Native: using batch_id=0 ({len(native_batch0)} rows, boost state); "
          f"dropped {n_dropped} rows from later batch(es)")

    return pd.concat([native_batch0, non_native], ignore_index=True)


# ── Filter and clean ──────────────────────────────────────────────────────────
def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)

    # 1. Retain only thermally stable runs
    df = df[df["hardware_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter:         {len(df):3d}/{n_total} rows kept")

    # 2. Drop first WARMUP_DROP run_ids within each (abstraction, batch)
    df = df[df["run_id"] > WARMUP_DROP].copy()
    print(f"  Drop run_id ≤ {WARMUP_DROP}:          {len(df):3d} rows remain")

    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for abs_name, grp in df.groupby("abstraction"):
        bw = grp["throughput"].dropna().to_numpy(dtype=float)
        q1, q3 = np.percentile(bw, [25, 75])
        rows.append({
            "abstraction":   abs_name,
            "n_runs":        len(bw),
            "median_bw_gbs": float(np.median(bw)),
            "mean_bw_gbs":   float(np.mean(bw)),
            "std_bw_gbs":    float(np.std(bw, ddof=1)),
            "iqr_bw_gbs":    float(q3 - q1),
            "cv_pct":        float(100.0 * np.std(bw, ddof=1) / np.mean(bw)),
            "min_bw_gbs":    float(np.min(bw)),
            "max_bw_gbs":    float(np.max(bw)),
            "q1_bw_gbs":     float(q1),
            "q3_bw_gbs":     float(q3),
        })
    return pd.DataFrame(rows)


# ── PPC (performance relative to native baseline) ─────────────────────────────
def compute_ppc(stats: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    native_row = stats[stats["abstraction"] == "native"]
    if native_row.empty:
        raise ValueError("Native baseline missing from stats — cannot compute PPC")
    native_median = float(native_row["median_bw_gbs"].iloc[0])
    print(f"  Native baseline (median):  {native_median:.2f} GB/s")

    stats = stats.copy()
    stats["ppc"]          = stats["median_bw_gbs"] / native_median
    stats["roofline_pct"] = 100.0 * stats["median_bw_gbs"] / PEAK_BW_GBS
    stats["ppc_tier"] = pd.cut(
        stats["ppc"],
        bins=[-np.inf, 0.60, 0.80, np.inf],
        labels=["poor", "acceptable", "excellent"],
    )
    return stats, native_median


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e1] Loading raw CSVs ...")
    raw = load_e1_csvs()

    print("[process_e1] Assigning batch IDs ...")
    raw = assign_batch_ids(raw)

    print("[process_e1] Selecting native baseline (last batch) ...")
    raw = select_native_baseline(raw)

    print("[process_e1] Filtering ...")
    clean = filter_and_clean(raw)

    print("[process_e1] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e1] Computing PPC ...")
    stats, _ = compute_ppc(stats)

    # Round floats for readability
    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e1_stream_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e1] Saved → {out_path}")
    print()
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
