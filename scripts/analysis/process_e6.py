#!/usr/bin/env python3
"""
E6 BFS — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs (hw_state_verified=1),
computes per-(graph_type, size) statistics, efficiency relative to native CUDA
baseline, frontier_irregularity, eff_gt1_flag, and deep_profiling flag.
Saves data/processed/e6_bfs_summary.csv.

Note: throughput_gflops in the raw CSV is GTEPS = n_edges / time_s / 1e9.
      Column names use "gflops" for CSV schema consistency with E2-E5.
      The processed summary also uses "median_gflops" = median GTEPS.

E6 DESIGN DECISIONS
[D1] Problem sizes: small=1024, medium=16384, large=65536 vertices.
[D2] Graph types: erdos_renyi, 2d_grid.
[D5] Metric: GTEPS stored as "gflops" columns.
[D9] frontier_irregularity = std(frontier_widths) / mean(frontier_widths) —
     CV of per-level frontier widths.  High = irregular BFS profile.
     Loaded from bfs_profile_* CSV if available; otherwise NaN.
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

# ── Configuration ─────────────────────────────────────────────────────────────
PLATFORM = "nvidia_rtx5060_laptop"

ALL_ABSTRACTIONS = ["native", "kokkos", "raja", "julia"]
# numba: UNSUPPORTED_CC120; sycl: NO_COMPILER

GRAPH_TYPES = ["erdos_renyi", "2d_grid"]

PROBLEM_SIZES = {
    "small":  1024,
    "medium": 16384,
    "large":  65536,
}


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e6_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ALL_ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"bfs_{abs_name}_{PLATFORM}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(df) == 0:
            print(f"  WARNING: empty CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        print(f"  {abs_name:16s}: {len(df):4d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW path and run run_bfs.sh first")
    return pd.concat(frames, ignore_index=True)


# ── Load frontier profiles ────────────────────────────────────────────────────
def load_frontier_profiles() -> dict:
    """
    Returns dict keyed by (graph_type, problem_size) →
        {'n_levels': int, 'frontier_widths': list[int],
         'irregularity': float}
    """
    profiles = {}
    pattern = os.path.join(DATA_RAW, f"bfs_profile_{PLATFORM}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  WARNING: no profile CSVs found; frontier_irregularity will be NaN",
              file=sys.stderr)
        return profiles
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Keep first profile per (graph_type, problem_size) — all reps identical
    for _, row in df.drop_duplicates(subset=["graph_type", "problem_size"]).iterrows():
        gt  = row["graph_type"]
        sz  = row["problem_size"]
        try:
            widths = [int(x) for x in str(row["frontier_widths"]).split(",") if x]
        except Exception:
            widths = []
        irr = np.nan
        if len(widths) > 1:
            mean_w = np.mean(widths)
            if mean_w > 0:
                irr = float(np.std(widths, ddof=0) / mean_w)
        profiles[(gt, sz)] = {
            "n_levels":         int(row["n_levels"]),
            "frontier_widths":  widths,
            "irregularity":     irr,
        }
    print(f"  Loaded {len(profiles)} frontier profiles from {len(files)} file(s)")
    return profiles


# ── Filter clean runs ─────────────────────────────────────────────────────────
def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter: {len(df):4d}/{n_total} rows kept")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (abs_name, graph_type, size_label), grp in df.groupby(
            ["abstraction", "graph_type", "problem_size"]):
        gflops = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gflops) == 0:
            continue
        q1, q3   = np.percentile(gflops, [25, 75])
        n_verts  = int(grp["n_vertices"].iloc[0]) if "n_vertices" in grp.columns \
                   else PROBLEM_SIZES.get(size_label, 0)
        n_edges  = int(grp["n_edges"].iloc[0])    if "n_edges"   in grp.columns else 0
        n_levels = int(grp["n_levels"].iloc[0])   if "n_levels"  in grp.columns else 0
        max_fw   = int(grp["max_frontier_width"].iloc[0]) \
                   if "max_frontier_width" in grp.columns else 0
        min_fw   = int(grp["min_frontier_width"].iloc[0]) \
                   if "min_frontier_width" in grp.columns else 0
        peak_ff  = float(grp["peak_frontier_fraction"].iloc[0]) \
                   if "peak_frontier_fraction" in grp.columns else 0.0

        rows.append({
            "abstraction":           abs_name,
            "graph_type":            graph_type,
            "problem_size":          size_label,
            "n_vertices":            n_verts,
            "n_edges":               n_edges,
            "n_levels":              n_levels,
            "max_frontier_width":    max_fw,
            "min_frontier_width":    min_fw,
            "peak_frontier_fraction": peak_ff,
            "n_runs":                len(gflops),
            "median_gflops":         float(np.median(gflops)),
            "mean_gflops":           float(np.mean(gflops)),
            "std_gflops":            float(np.std(gflops, ddof=1)) if len(gflops) > 1 else 0.0,
            "iqr_gflops":            float(q3 - q1),
            "cv_pct":                float(100.0 * np.std(gflops, ddof=1) / np.mean(gflops))
                                     if len(gflops) > 1 and np.mean(gflops) > 0 else 0.0,
            "min_gflops":            float(np.min(gflops)),
            "max_gflops":            float(np.max(gflops)),
            "q1_gflops":             float(q1),
            "q3_gflops":             float(q3),
        })

    abs_order   = {a: i for i, a in enumerate(ALL_ABSTRACTIONS)}
    graph_order = {g: i for i, g in enumerate(GRAPH_TYPES)}
    size_order  = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_ord_abs"]   = result["abstraction"].map(abs_order)
    result["_ord_graph"] = result["graph_type"].map(graph_order)
    result["_ord_size"]  = result["problem_size"].map(size_order)
    result = result.sort_values(["_ord_graph", "_ord_size", "_ord_abs"]) \
                   .drop(columns=["_ord_abs", "_ord_graph", "_ord_size"]) \
                   .reset_index(drop=True)
    return result


# ── Efficiency + flags ────────────────────────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame,
                        profiles: dict) -> pd.DataFrame:
    stats = stats.copy()
    native_rows = stats[stats["abstraction"] == "native"].set_index(
        ["graph_type", "problem_size"])

    if native_rows.empty:
        print("  WARNING: native baseline not found — efficiency not computed",
              file=sys.stderr)
        stats["efficiency"]           = np.nan
        stats["eff_gt1_flag"]         = False
        stats["flag_deep_profiling"]  = False
        stats["ppc_tier"]             = "unknown"
        stats["frontier_irregularity"] = np.nan
        return stats

    def native_median(gt: str, sz: str) -> float:
        key = (gt, sz)
        if key in native_rows.index:
            return float(native_rows.loc[key, "median_gflops"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        nm = native_median(row["graph_type"], row["problem_size"])
        if row["abstraction"] == "native" or np.isnan(nm) or nm == 0:
            eff = 1.0 if row["abstraction"] == "native" else np.nan
        else:
            eff = row["median_gflops"] / nm
        efficiencies.append(eff)
    stats["efficiency"] = efficiencies

    # eff_gt1_flag: abstraction outperforms native
    stats["eff_gt1_flag"] = (
        (stats["abstraction"] != "native") &
        (stats["efficiency"] > 1.0)
    )

    # deep_profiling flag: efficiency < 0.85
    stats["flag_deep_profiling"] = (
        (stats["abstraction"] != "native") &
        (stats["efficiency"] < 0.85)
    )

    # PPC tier (§9.4)
    def tier(eff):
        if np.isnan(eff):  return "unknown"
        if eff >= 0.80:    return "excellent"
        if eff >= 0.60:    return "acceptable"
        return "poor"

    stats["ppc_tier"] = [
        "native" if row["abstraction"] == "native" else tier(row["efficiency"])
        for _, row in stats.iterrows()
    ]

    # frontier_irregularity from profiles (CV of frontier widths across levels)
    stats["frontier_irregularity"] = stats.apply(
        lambda r: profiles.get((r["graph_type"], r["problem_size"]),
                               {}).get("irregularity", np.nan),
        axis=1
    )

    return stats


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E6 BFS Summary (median GTEPS, efficiency vs native):")
    print("Note: throughput stored as GTEPS = n_edges / time_s / 1e9 in 'gflops' columns.")
    for graph_type in GRAPH_TYPES:
        sub_g = stats[stats["graph_type"] == graph_type]
        if sub_g.empty:
            continue
        print()
        print(f"  Graph type: {graph_type}")
        print("-" * 115)
        for size_label in ["small", "medium", "large"]:
            sub = sub_g[sub_g["problem_size"] == size_label]
            if sub.empty:
                continue
            meta = sub.iloc[0]
            print(f"\n  Problem size: {size_label} (N={meta['n_vertices']}, "
                  f"n_edges={meta['n_edges']}, n_levels={meta['n_levels']}, "
                  f"max_fw={meta['max_frontier_width']}, "
                  f"peak_ff={meta['peak_frontier_fraction']:.4f}, "
                  f"irregularity={meta['frontier_irregularity']:.3f})")
            print(f"  {'Abstraction':18s} {'Median GTEPS':>14s} {'IQR':>8s} "
                  f"{'Eff':>7s} {'CV%':>6s} {'Tier':>12s} {'Flags':>6s}")
            print(f"  {'-'*18} {'-'*14} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*6}")
            for _, row in sub.iterrows():
                flags = ""
                if row.get("flag_deep_profiling", False): flags += "⚑"
                if row.get("eff_gt1_flag", False):        flags += ">1"
                eff_str = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) \
                          else "  n/a "
                print(f"  {row['abstraction']:18s} {row['median_gflops']:>14.6f} "
                      f"{row['iqr_gflops']:>8.6f} {eff_str:>7s} "
                      f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s} {flags:>6s}")

    flagged = stats[stats.get("flag_deep_profiling", pd.Series(dtype=bool)).astype(bool)]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['graph_type']:16s} "
                  f"{row['problem_size']:8s} eff={row['efficiency']:.4f}  "
                  f"n_levels={row['n_levels']}  irr={row['frontier_irregularity']:.3f}")

    gt1 = stats[stats.get("eff_gt1_flag", pd.Series(dtype=bool)).astype(bool)]
    if not gt1.empty:
        print()
        print("  >1 Abstraction faster than native baseline:")
        for _, row in gt1.iterrows():
            print(f"    {row['abstraction']:18s} {row['graph_type']:16s} "
                  f"{row['problem_size']:8s} eff={row['efficiency']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e6] Loading raw CSVs ...")
    raw = load_e6_csvs()

    print("[process_e6] Loading frontier profiles ...")
    profiles = load_frontier_profiles()

    print("[process_e6] Filtering ...")
    clean = filter_clean(raw)

    print("[process_e6] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e6] Computing efficiency vs native baseline ...")
    stats = compute_efficiency(stats, profiles)

    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(6)

    out_path = os.path.join(DATA_PROC, "e6_bfs_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e6] Saved → {out_path}")

    print_report(stats)

    n_flagged = int(stats.get("flag_deep_profiling", pd.Series(dtype=bool)).sum())
    n_gt1     = int(stats.get("eff_gt1_flag",        pd.Series(dtype=bool)).sum())
    print(f"\n[process_e6] {n_flagged} configuration(s) flagged for deep profiling.")
    print(f"[process_e6] {n_gt1} configuration(s) with efficiency > 1.0.")


if __name__ == "__main__":
    main()
