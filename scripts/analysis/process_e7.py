#!/usr/bin/env python3
"""
E7 N-Body — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs (hw_state_verified=1),
computes per-(kernel, size) statistics, efficiency relative to native_notile
baseline, eff_gt1_flag, and deep_profiling flag.
Saves data/processed/e7_nbody_summary.csv.

E7 DESIGN DECISIONS
[D1] Problem sizes: small=4000, medium=32000, large=256000 atoms (FCC, 4M^3).
[D2] Kernel variants: notile (neighbor-list), tile (all-pairs shared-mem P006).
[D3] Baseline: native_notile for efficiency computation.
[D4] Metric: GFLOP/s = actual_flops / time_s / 1e9.
     notile flops = total_neighbor_pairs × 20.
     tile flops   = N×(N-1)×20 (all pairs).
[D5] PPC tiers (§9.4 amended for E7):
     ≥0.95 excellent, 0.85–0.95 good, 0.70–0.85 acceptable, <0.70 poor.
[D6] AI = actual_flops_per_particle / bytes_per_particle.
     bytes = 16(pos_i) + n_nbrs_mean×(16+4)(pos_j+idx) + 12(force) + 4(count).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW  = os.path.join(REPO_ROOT, "data", "raw")
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

PLATFORM = "nvidia_rtx5060_laptop"
ALL_ABSTRACTIONS = ["native", "julia"]
PROBLEM_SIZES    = {"small": 4000, "medium": 32000, "large": 256000}
KERNEL_VARIANTS  = ["notile", "tile"]
FLOPS_PER_PAIR   = 20


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e7_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ALL_ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"nbody_{abs_name}_{PLATFORM}_*.csv")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(df) == 0:
            print(f"  WARNING: empty CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        print(f"  {abs_name:16s}: {len(df):5d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW and run run_nbody.sh first")
    return pd.concat(frames, ignore_index=True)


# ── Filter clean runs ─────────────────────────────────────────────────────────
def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter: {len(df):5d}/{n_total} rows kept")
    return df


# ── Summary statistics ─────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (abs_name, kernel, size_label), grp in df.groupby(
            ["abstraction", "kernel", "problem_size"]):
        gflops = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gflops) == 0:
            continue
        q1, q3 = np.percentile(gflops, [25, 75])
        n_atoms      = int(grp["n_atoms"].iloc[0]) \
                       if "n_atoms"      in grp.columns else PROBLEM_SIZES.get(size_label, 0)
        n_nbrs_mean  = float(grp["n_nbrs_mean"].iloc[0]) \
                       if "n_nbrs_mean"  in grp.columns else 0.0
        n_nbrs_max   = int(grp["n_nbrs_max"].iloc[0]) \
                       if "n_nbrs_max"   in grp.columns else 0
        actual_flops = int(grp["actual_flops"].iloc[0]) \
                       if "actual_flops" in grp.columns else 0

        # Arithmetic intensity estimate (notile only; tile is all-pairs so different)
        ai = np.nan
        if kernel == "notile" and n_nbrs_mean > 0:
            bytes_pp  = 16.0 + n_nbrs_mean * (16.0 + 4.0) + 12.0 + 4.0
            flops_pp  = n_nbrs_mean * FLOPS_PER_PAIR
            ai = flops_pp / bytes_pp

        rows.append({
            "abstraction":    abs_name,
            "kernel":         kernel,
            "problem_size":   size_label,
            "n_atoms":        n_atoms,
            "n_nbrs_mean":    n_nbrs_mean,
            "n_nbrs_max":     n_nbrs_max,
            "actual_flops":   actual_flops,
            "ai_flop_byte":   ai,
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

    abs_order  = {a: i for i, a in enumerate(ALL_ABSTRACTIONS)}
    kv_order   = {k: i for i, k in enumerate(KERNEL_VARIANTS)}
    sz_order   = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_oa"] = result["abstraction"].map(abs_order)
    result["_ok"] = result["kernel"].map(kv_order)
    result["_os"] = result["problem_size"].map(sz_order)
    result = result.sort_values(["_ok", "_os", "_oa"]) \
                   .drop(columns=["_oa", "_ok", "_os"]) \
                   .reset_index(drop=True)
    return result


# ── Efficiency and flags ───────────────────────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    # Baseline: native_notile per problem_size
    baseline = stats[
        (stats["abstraction"] == "native") & (stats["kernel"] == "notile")
    ].set_index("problem_size")

    if baseline.empty:
        print("  WARNING: native_notile baseline not found — efficiency not computed",
              file=sys.stderr)
        stats["efficiency"]          = np.nan
        stats["eff_gt1_flag"]        = False
        stats["flag_deep_profiling"] = False
        stats["ppc_tier"]            = "unknown"
        return stats

    def native_notile_median(sz: str) -> float:
        if sz in baseline.index:
            return float(baseline.loc[sz, "median_gflops"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        if row["abstraction"] == "native" and row["kernel"] == "notile":
            efficiencies.append(1.0)
        else:
            nm = native_notile_median(row["problem_size"])
            efficiencies.append(row["median_gflops"] / nm
                                if not np.isnan(nm) and nm > 0 else np.nan)
    stats["efficiency"] = efficiencies

    stats["eff_gt1_flag"] = (
        ~((stats["abstraction"] == "native") & (stats["kernel"] == "notile")) &
        (stats["efficiency"] > 1.0)
    )

    stats["flag_deep_profiling"] = (
        ~((stats["abstraction"] == "native") & (stats["kernel"] == "notile")) &
        (stats["efficiency"] < 0.85)
    )

    def tier(eff):
        if np.isnan(eff): return "unknown"
        if eff >= 0.95:   return "excellent"
        if eff >= 0.85:   return "good"
        if eff >= 0.70:   return "acceptable"
        return "poor"

    stats["ppc_tier"] = [
        "native" if (r["abstraction"] == "native" and r["kernel"] == "notile")
        else tier(r["efficiency"])
        for _, r in stats.iterrows()
    ]
    return stats


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E7 N-Body Summary (median GFLOP/s, efficiency vs native_notile):")
    print("Note: 'tile' uses all-pairs O(N²) tiling (P006 test); 'notile' uses neighbor-list.")
    for kv in KERNEL_VARIANTS:
        sub_k = stats[stats["kernel"] == kv]
        if sub_k.empty:
            continue
        print()
        print(f"  Kernel variant: {kv}")
        print("-" * 100)
        for sz in ["small", "medium", "large"]:
            sub = sub_k[sub_k["problem_size"] == sz]
            if sub.empty:
                continue
            meta = sub.iloc[0]
            ai_str = f"{meta['ai_flop_byte']:.3f}" if not np.isnan(meta["ai_flop_byte"]) else "n/a"
            print(f"\n  Size: {sz} (N={meta['n_atoms']}, n_nbrs_mean={meta['n_nbrs_mean']:.1f}, "
                  f"n_nbrs_max={meta['n_nbrs_max']}, AI={ai_str} FLOP/byte)")
            print(f"  {'Abstraction':18s} {'Median GFLOP/s':>14s} {'IQR':>8s} "
                  f"{'Eff':>7s} {'CV%':>6s} {'Tier':>12s} {'Flags':>6s}")
            print(f"  {'-'*18} {'-'*14} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*6}")
            for _, row in sub.iterrows():
                flags = ""
                if row.get("flag_deep_profiling", False): flags += "⚑"
                if row.get("eff_gt1_flag",        False): flags += ">1"
                eff_s = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) else "  n/a "
                print(f"  {row['abstraction']:18s} {row['median_gflops']:>14.4f} "
                      f"{row['iqr_gflops']:>8.4f} {eff_s:>7s} "
                      f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s} {flags:>6s}")

    flagged = stats[stats.get("flag_deep_profiling", pd.Series(dtype=bool)).fillna(False)]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['kernel']:10s} "
                  f"{row['problem_size']:8s}  eff={row['efficiency']:.4f}")

    gt1 = stats[stats.get("eff_gt1_flag", pd.Series(dtype=bool)).fillna(False)]
    if not gt1.empty:
        print()
        print("  >1 Abstraction faster than native_notile:")
        for _, row in gt1.iterrows():
            print(f"    {row['abstraction']:18s} {row['kernel']:10s} "
                  f"{row['problem_size']:8s}  eff={row['efficiency']:.4f}")

    # Tile vs notile speedup (P006 evidence)
    print()
    print("  P006 Tiling speedup (tile / notile, native only):")
    for sz in ["small", "medium", "large"]:
        nt = stats[(stats["abstraction"] == "native") &
                   (stats["kernel"] == "notile") &
                   (stats["problem_size"] == sz)]
        tl = stats[(stats["abstraction"] == "native") &
                   (stats["kernel"] == "tile") &
                   (stats["problem_size"] == sz)]
        if nt.empty or tl.empty:
            continue
        speedup = float(tl["median_gflops"].iloc[0]) / float(nt["median_gflops"].iloc[0])
        print(f"    {sz:8s}: notile={float(nt['median_gflops'].iloc[0]):.4f} GFLOP/s  "
              f"tile={float(tl['median_gflops'].iloc[0]):.4f} GFLOP/s  "
              f"speedup={speedup:.3f}x")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("[process_e7] Loading raw CSVs ...")
    raw = load_e7_csvs()

    print("[process_e7] Filtering ...")
    clean = filter_clean(raw)

    print("[process_e7] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e7] Computing efficiency vs native_notile ...")
    stats = compute_efficiency(stats)

    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(6)

    out_path = os.path.join(DATA_PROC, "e7_nbody_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e7] Saved → {out_path}")

    print_report(stats)

    n_flag = int(stats.get("flag_deep_profiling", pd.Series(dtype=bool)).sum())
    n_gt1  = int(stats.get("eff_gt1_flag",        pd.Series(dtype=bool)).sum())
    print(f"\n[process_e7] {n_flag} configuration(s) flagged for deep profiling.")
    print(f"[process_e7] {n_gt1} configuration(s) with efficiency > 1.0.")


if __name__ == "__main__":
    main()
