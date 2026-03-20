#!/usr/bin/env python3
"""
E4 SpMV — data processing pipeline.

Loads raw per-abstraction CSVs for all platforms, filters clean runs
(hw_state_verified=1), computes per-(platform, matrix_type, size) statistics
and efficiency relative to native baseline, flags configurations for deep
profiling, saves data/processed/e4_spmv_summary.csv.

E4 DESIGN DECISIONS
[D1] Problem sizes: small=1024 rows, medium=8192 rows, large=32768 rows.
     Actual nrows may differ slightly for laplacian_2d (grid rounding).
[D2] Matrix types: laplacian_2d, random_sparse, power_law.
     load imbalance (CoV of row lengths) is highest for power_law.
[D4] Primary metric: GFLOP/s = 2*nnz / time_s / 1e9 (1 mul + 1 add per nnz).
[D7] Amended warmup: adaptive CV<2% — no fixed WARMUP_DROP needed since the
     kernel itself discards the warmup phase before emitting run_id rows.
[D6] experiment_id: spmv_{abstraction}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}
[D8] Multi-platform: PLATFORM_CONFIGS dict mirrors E3 process_e3.py structure.
Measurement protocol: no locked clocks (sudo unavailable); adaptive warmup (§9.1 amended).
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

# ── Platform configurations (D8) ──────────────────────────────────────────────
PLATFORM_CONFIGS = {
    "nvidia_rtx5060": {
        "abstractions": ["native", "kokkos", "raja", "sycl", "julia", "numba"],
        "unsupported":  {"numba"},          # CC 12.0 PTX mismatch
        "peak_bw_gbs":  270.0,              # GB/s — E1 STREAM measured
        "peak_fp64_gflops": 261.0,          # GFLOP/s
    },
    "amd_mi300x": {
        "abstractions": ["native", "kokkos", "raja", "sycl", "julia"],
        "unsupported":  set(),
        "peak_bw_gbs":  4010.0,             # GB/s — E1 STREAM measured
        "peak_fp64_gflops": 163400.0,       # GFLOP/s (163.4 TFLOP/s datasheet)
    },
}

MATRIX_TYPES = ["laplacian_2d", "random_sparse", "power_law"]

PROBLEM_SIZES = {
    "small":  1024,
    "medium": 8192,
    "large":  32768,
}


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e4_csvs() -> pd.DataFrame:
    frames = []
    for platform, cfg in PLATFORM_CONFIGS.items():
        found_any = False
        for abs_name in cfg["abstractions"]:
            pattern = os.path.join(DATA_RAW, f"spmv_{abs_name}_{platform}_*.csv")
            files = sorted(glob.glob(pattern))
            if not files:
                if abs_name in cfg["unsupported"]:
                    print(f"  {platform}/{abs_name}: UNSUPPORTED — skipped",
                          file=sys.stderr)
                else:
                    print(f"  WARNING: no CSV for {platform}/{abs_name}", file=sys.stderr)
                continue
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            if len(df) == 0:
                if abs_name in cfg["unsupported"]:
                    print(f"  {platform}/{abs_name}: UNSUPPORTED (empty) — skipped",
                          file=sys.stderr)
                continue
            df["platform"] = platform
            print(f"  {platform}/{abs_name:16s}: {len(df):4d} rows from {len(files)} file(s)")
            frames.append(df)
            found_any = True
        if not found_any:
            print(f"  WARNING: no CSVs found for platform={platform}", file=sys.stderr)

    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW path and run run_spmv.sh first")
    return pd.concat(frames, ignore_index=True)


# ── Filter ────────────────────────────────────────────────────────────────────
def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter:   {len(df):4d}/{n_total} rows kept")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_abs_global = ["native", "kokkos", "raja", "sycl", "julia", "numba"]
    plat_order = list(PLATFORM_CONFIGS.keys())

    for (platform, abs_name, matrix_type, size_label), grp in df.groupby(
            ["platform", "abstraction", "matrix_type", "problem_size"]):
        gflops = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gflops) == 0:
            continue
        q1, q3 = np.percentile(gflops, [25, 75])
        n_rows = int(grp["n_rows"].iloc[0]) if "n_rows" in grp.columns else PROBLEM_SIZES.get(size_label, 0)
        nnz    = int(grp["nnz"].iloc[0])    if "nnz"    in grp.columns else 0
        rows.append({
            "platform":       platform,
            "abstraction":    abs_name,
            "matrix_type":    matrix_type,
            "problem_size":   size_label,
            "n_rows":         n_rows,
            "nnz":            nnz,
            "n_runs":         len(gflops),
            "median_gflops":  float(np.median(gflops)),
            "mean_gflops":    float(np.mean(gflops)),
            "std_gflops":     float(np.std(gflops, ddof=1)) if len(gflops) > 1 else 0.0,
            "iqr_gflops":     float(q3 - q1),
            "cv_pct":         float(100.0 * np.std(gflops, ddof=1) / np.mean(gflops))
                               if len(gflops) > 1 and np.mean(gflops) > 0 else 0.0,
            "min_gflops":     float(np.min(gflops)),
            "max_gflops":     float(np.max(gflops)),
            "q1_gflops":      float(q1),
            "q3_gflops":      float(q3),
        })

    abs_order  = {a: i for i, a in enumerate(all_abs_global)}
    mtyp_order = {m: i for i, m in enumerate(MATRIX_TYPES)}
    size_order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_ord_plat"] = result["platform"].apply(
        lambda p: plat_order.index(p) if p in plat_order else 999)
    result["_ord_abs"]  = result["abstraction"].map(abs_order).fillna(999)
    result["_ord_mtyp"] = result["matrix_type"].map(mtyp_order).fillna(999)
    result["_ord_size"] = result["problem_size"].map(size_order).fillna(999)
    result = result.sort_values(["_ord_plat", "_ord_mtyp", "_ord_size", "_ord_abs"]) \
                   .drop(columns=["_ord_plat", "_ord_abs", "_ord_mtyp", "_ord_size"]) \
                   .reset_index(drop=True)
    return result


# ── Efficiency and PPC ────────────────────────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    # Native baseline keyed by (platform, matrix_type, problem_size)
    native_rows = stats[stats["abstraction"] == "native"].set_index(
        ["platform", "matrix_type", "problem_size"])

    if native_rows.empty:
        print("  WARNING: native baseline not found — efficiency not computed",
              file=sys.stderr)
        stats["efficiency"] = np.nan
        stats["flag_deep_profiling"] = False
        stats["ppc_tier"] = "unknown"
        return stats

    def native_median(platform: str, mtype: str, size_label: str) -> float:
        key = (platform, mtype, size_label)
        if key in native_rows.index:
            return float(native_rows.loc[key, "median_gflops"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        nm = native_median(row["platform"], row["matrix_type"], row["problem_size"])
        if row["abstraction"] == "native" or np.isnan(nm) or nm == 0:
            eff = 1.0 if row["abstraction"] == "native" else np.nan
        else:
            eff = row["median_gflops"] / nm
        efficiencies.append(eff)

    stats["efficiency"] = efficiencies

    # Deep profiling flag: efficiency < 0.85 (§9.5 protocol)
    stats["flag_deep_profiling"] = (
        (stats["abstraction"] != "native") &
        (stats["efficiency"] < 0.85)
    )

    # PPC tier (project_spec §9.4)
    def tier(eff):
        if np.isnan(eff):
            return "unknown"
        if eff >= 0.80:
            return "excellent"
        if eff >= 0.60:
            return "acceptable"
        return "poor"

    stats["ppc_tier"] = [
        "native" if row["abstraction"] == "native" else tier(row["efficiency"])
        for _, row in stats.iterrows()
    ]
    return stats


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E4 SpMV Summary (median GFLOP/s, efficiency vs native):")
    for platform in PLATFORM_CONFIGS:
        sub_plat = stats[stats["platform"] == platform]
        if sub_plat.empty:
            continue
        print(f"\n{'='*84}")
        print(f"  Platform: {platform}")
        print(f"{'='*84}")
        for matrix_type in MATRIX_TYPES:
            sub_m = sub_plat[sub_plat["matrix_type"] == matrix_type]
            if sub_m.empty:
                continue
            print()
            print(f"  Matrix type: {matrix_type}")
            print("-" * 90)
            for size_label in ["small", "medium", "large"]:
                sub = sub_m[sub_m["problem_size"] == size_label]
                if sub.empty:
                    continue
                n_rows = PROBLEM_SIZES[size_label]
                print(f"\n  Problem size: {size_label} (N≈{n_rows} rows)")
                print(f"  {'Abstraction':18s} {'Median GFLOP/s':>15s} {'IQR':>8s} {'Eff':>7s} "
                      f"{'CV%':>6s} {'Tier':>12s} {'Flag':>5s}")
                print(f"  {'-'*18} {'-'*15} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*5}")
                for _, row in sub.iterrows():
                    flag    = "⚑" if row.get("flag_deep_profiling", False) else ""
                    eff_str = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) else "  n/a "
                    print(f"  {row['abstraction']:18s} {row['median_gflops']:>15.4f} "
                          f"{row['iqr_gflops']:>8.4f} {eff_str:>7s} "
                          f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s} {flag}")

    flagged = stats[stats.get("flag_deep_profiling", pd.Series(dtype=bool)).astype(bool)]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['matrix_type']:16s} {row['problem_size']:8s} "
                  f"eff={row['efficiency']:.4f}")
    else:
        print()
        print("  No configurations flagged for deep profiling.")
        print("  Expected: memory-bound SpMV (AI≈0.13 FLOP/byte) shows high")
        print("  efficiency across all abstractions — bandwidth bottleneck dominates.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e4] Loading raw CSVs ...")
    raw = load_e4_csvs()

    print("[process_e4] Filtering ...")
    clean = filter_clean(raw)

    print("[process_e4] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e4] Computing efficiency vs native baseline ...")
    stats = compute_efficiency(stats)

    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e4_spmv_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e4] Saved → {out_path}")

    print_report(stats)

    n_flagged = int(stats.get("flag_deep_profiling", pd.Series(dtype=bool)).sum())
    if n_flagged > 0:
        print(f"\n[process_e4] {n_flagged} configuration(s) flagged for deep profiling.")
        print("[process_e4] Run nsys/ncu on flagged configs per §9.5 protocol.")
        print("[process_e4] Expected candidates: power_law (load imbalance),")
        print("[process_e4] julia small/medium (P001 launch overhead),")
        print("[process_e4] any abstraction with eff<0.85 at large N.")
    else:
        print("\n[process_e4] No deep profiling required.")
        print("[process_e4] Hypothesis: memory-bound AI≈0.13 masks abstraction overhead.")


if __name__ == "__main__":
    main()
