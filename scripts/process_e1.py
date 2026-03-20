#!/usr/bin/env python3
"""
E1 STREAM Triad — data processing pipeline.

Loads raw per-abstraction CSVs for all configured platforms, filters clean
runs, removes warm-up, computes statistics and PPC, and saves a combined
data/processed/e1_stream_summary.csv with a platform column.

Measurement protocols
─────────────────────
NVIDIA RTX 5060 Laptop (locked-clock session, 2026-03-14)
  - SM clock locked at 2092 MHz; memory clock self-transitions 9001→12001 MHz
    after ~40 iterations of sustained bandwidth load
  - warmup=50 ensures the 12001 MHz state before any timed run
  - 30 s cooldown between abstractions; batch_id=0 (first batch = boost state)
    is the canonical native baseline
  - All 5 abstractions (native, kokkos, raja, julia, numba) converge ~345-350 GB/s

AMD MI300X (2026-03-20)
  - Single session per problem size; hardware_state_verified flag is the sole
    warmup gate (no clock-boost artifact, so no batch selection needed)
  - 3 problem sizes: small (2^20), medium (2^24), large (2^28)
  - 4 abstractions: native, kokkos, raja, julia
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

# ── Per-platform configuration ────────────────────────────────────────────────
PLATFORM_CONFIGS = {
    "nvidia_rtx5060_laptop_locked": {
        "peak_bw_gbs":         384.0,   # GDDR7 12001 MHz × 128-bit × 2
        "warmup_drop":         5,        # drop run_id ≤ 5 after hw_state filter
        "abstractions":        ["native", "kokkos", "raja", "julia", "numba"],
        "select_native_batch0": True,    # keep only clock-boost batch for native
    },
    "amd_mi300x": {
        "peak_bw_gbs":         5300.0,  # HBM3 theoretical peak
        "warmup_drop":         0,        # hw_state_verified is the sole warmup gate
        "abstractions":        ["native", "kokkos", "raja", "julia"],
        "select_native_batch0": False,
    },
}

# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_platform_csvs(platform: str, cfg: dict) -> pd.DataFrame:
    frames = []
    for abs_name in cfg["abstractions"]:
        pattern = os.path.join(DATA_RAW, f"stream_{abs_name}_{platform}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING [{platform}]: no CSV for abstraction={abs_name}",
                  file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"  {abs_name:8s}: {len(df):3d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSV files found for platform={platform}")
    combined = pd.concat(frames, ignore_index=True)
    combined["platform"] = platform
    return combined


# ── Batch detection ───────────────────────────────────────────────────────────
# Sort by (abstraction, problem_size, timestamp, run_id).  A new batch starts
# whenever run_id resets to a value ≤ the previous run_id within the same
# (abstraction, problem_size) group.
def assign_batch_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["abstraction", "problem_size", "timestamp", "run_id"]
    ).reset_index(drop=True)
    batch_col = []
    for _, grp in df.groupby(["abstraction", "problem_size"], sort=False):
        bid, prev_rid = 0, None
        for rid in grp["run_id"]:
            if prev_rid is not None and rid <= prev_rid:
                bid += 1
            batch_col.append(bid)
            prev_rid = rid
    df["batch_id"] = batch_col
    return df


# ── Native baseline selection (NVIDIA only) ───────────────────────────────────
# Keep only batch_id=0 for native — the clock-boost state that matches all
# other abstractions.  Per problem_size to stay general.
def select_native_batch0(df: pd.DataFrame) -> pd.DataFrame:
    native_mask   = df["abstraction"] == "native"
    non_native    = df[~native_mask].copy()
    native_df     = df[native_mask].copy()
    native_batch0 = native_df[native_df["batch_id"] == 0].copy()
    n_dropped     = native_mask.sum() - len(native_batch0)
    print(f"  Native: using batch_id=0 ({len(native_batch0)} rows, boost state); "
          f"dropped {n_dropped} rows from later batch(es)")
    return pd.concat([native_batch0, non_native], ignore_index=True)


# ── Filter and clean ──────────────────────────────────────────────────────────
def filter_and_clean(df: pd.DataFrame, warmup_drop: int) -> pd.DataFrame:
    n_total = len(df)
    df = df[df["hardware_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter: {len(df):3d}/{n_total} rows kept")
    if warmup_drop > 0:
        df = df[df["run_id"] > warmup_drop].copy()
        print(f"  Drop run_id ≤ {warmup_drop}: {len(df):3d} rows remain")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    rows = []
    for (abs_name, size), grp in df.groupby(["abstraction", "problem_size"]):
        bw = grp["throughput"].dropna().to_numpy(dtype=float)
        if len(bw) == 0:
            continue
        q1, q3 = np.percentile(bw, [25, 75])
        # problem_size_n: take the unique value; warn if mixed
        n_vals = grp["problem_size_n"].dropna().unique()
        prob_n = int(n_vals[0]) if len(n_vals) == 1 else int(n_vals[0])
        rows.append({
            "platform":      platform,
            "abstraction":   abs_name,
            "problem_size":  size,
            "problem_size_n": prob_n,
            "n_runs":        len(bw),
            "median_bw_gbs": float(np.median(bw)),
            "mean_bw_gbs":   float(np.mean(bw)),
            "std_bw_gbs":    float(np.std(bw, ddof=1)) if len(bw) > 1 else 0.0,
            "iqr_bw_gbs":    float(q3 - q1),
            "cv_pct":        float(100.0 * np.std(bw, ddof=1) / np.mean(bw))
                             if len(bw) > 1 else 0.0,
            "min_bw_gbs":    float(np.min(bw)),
            "max_bw_gbs":    float(np.max(bw)),
            "q1_bw_gbs":     float(q1),
            "q3_bw_gbs":     float(q3),
        })
    return pd.DataFrame(rows)


# ── PPC (performance relative to native baseline within platform+size) ────────
def compute_ppc(stats: pd.DataFrame, peak_bw_gbs: float) -> pd.DataFrame:
    stats = stats.copy()
    ppc_vals, roofline_vals = [], []
    for (_, size), grp in stats.groupby(["platform", "problem_size"]):
        native_row = grp[grp["abstraction"] == "native"]
        if native_row.empty:
            raise ValueError(f"Native baseline missing for size={size}")
        native_median = float(native_row["median_bw_gbs"].iloc[0])
        print(f"    [{size}] native median: {native_median:.2f} GB/s")
        for idx in grp.index:
            med = stats.at[idx, "median_bw_gbs"]
            ppc_vals.append((idx, med / native_median))
            roofline_vals.append((idx, 100.0 * med / peak_bw_gbs))

    for idx, v in ppc_vals:
        stats.at[idx, "ppc"] = v
    for idx, v in roofline_vals:
        stats.at[idx, "roofline_pct"] = v

    stats["ppc_tier"] = pd.cut(
        stats["ppc"],
        bins=[-np.inf, 0.60, 0.80, np.inf],
        labels=["poor", "acceptable", "excellent"],
    )
    return stats


# ── Process one platform ───────────────────────────────────────────────────────
def process_platform(platform: str, cfg: dict) -> pd.DataFrame:
    print(f"\n[process_e1] ── {platform} ──")

    print("  Loading CSVs ...")
    raw = load_platform_csvs(platform, cfg)

    print("  Assigning batch IDs ...")
    raw = assign_batch_ids(raw)

    if cfg["select_native_batch0"]:
        print("  Selecting native baseline (batch_id=0) ...")
        raw = select_native_batch0(raw)

    print("  Filtering ...")
    clean = filter_and_clean(raw, cfg["warmup_drop"])

    print("  Computing statistics ...")
    stats = compute_stats(clean, platform)

    print("  Computing PPC ...")
    stats = compute_ppc(stats, cfg["peak_bw_gbs"])

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_frames = []
    for platform, cfg in PLATFORM_CONFIGS.items():
        df = process_platform(platform, cfg)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    float_cols = combined.select_dtypes(include="float").columns
    combined[float_cols] = combined[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e1_stream_summary.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n[process_e1] Saved → {out_path}  ({len(combined)} rows)")
    print()
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
