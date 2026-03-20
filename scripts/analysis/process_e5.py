#!/usr/bin/env python3
"""
E5 SpTRSV — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs (hw_state_verified=1),
computes per-(matrix_type, size) statistics, efficiency relative to native CUDA
baseline, parallelism_ratio, eff_gt1_flag, and deep_profiling flag.
Saves data/processed/e5_sptrsv_summary.csv.

E5 DESIGN DECISIONS
[D1] Problem sizes: small=256, medium=2048, large=8192 rows.
[D2] Matrix types: lower_triangular_laplacian, lower_triangular_random.
[D4] Primary metric: GFLOP/s = 2*nnz / time_s / 1e9 (consistent with E4 SpMV).
     SpTRSV is latency-bound (serial dependency chain), not bandwidth-bound.
     GFLOP/s will be low — the binding constraint is level-set depth (n_levels),
     not compute or bandwidth.
[D7] Amended warmup: adaptive CV<2% — x reset included in warmup, excluded from
     timed region.
[D6] experiment_id: sptrsv_{abstraction}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}

Key additional columns vs E4:
  n_levels          — depth of the dependency DAG (serial synchronisation count)
  max_level_width   — maximum rows in any single level (peak parallelism)
  min_level_width   — minimum rows in any level (parallelism bottleneck)
  parallelism_ratio — max_level_width / n_rows: fraction of rows in widest wavefront
  eff_gt1_flag      — efficiency > 1.0 (abstraction outperforms native baseline)
  flag_deep_profiling — efficiency < 0.85 (trigger nsys/ncu analysis)
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
PLATFORM = "nvidia_rtx5060"

ALL_ABSTRACTIONS = ["native", "kokkos", "raja", "julia"]
# numba: UNSUPPORTED_CC120; sycl: NO_COMPILER

MATRIX_TYPES = ["lower_triangular_laplacian", "lower_triangular_random"]

PROBLEM_SIZES = {
    "small":  256,
    "medium": 2048,
    "large":  8192,
}


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e5_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ALL_ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"sptrsv_{abs_name}_{PLATFORM}_*.csv")
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
        raise RuntimeError("No CSV files found — check DATA_RAW path and run run_sptrsv.sh first")
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
    for (abs_name, matrix_type, size_label), grp in df.groupby(
            ["abstraction", "matrix_type", "problem_size"]):
        gflops = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gflops) == 0:
            continue
        q1, q3 = np.percentile(gflops, [25, 75])
        n_rows = int(grp["n_rows"].iloc[0]) if "n_rows" in grp.columns else PROBLEM_SIZES.get(size_label, 0)
        nnz    = int(grp["nnz"].iloc[0])    if "nnz"    in grp.columns else 0

        # Level-set metadata: constant per (matrix_type, n_rows), take first row
        n_levels  = int(grp["n_levels"].iloc[0])       if "n_levels"       in grp.columns else 0
        max_lw    = int(grp["max_level_width"].iloc[0]) if "max_level_width" in grp.columns else 0
        min_lw    = int(grp["min_level_width"].iloc[0]) if "min_level_width" in grp.columns else 0

        rows.append({
            "abstraction":      abs_name,
            "matrix_type":      matrix_type,
            "problem_size":     size_label,
            "n_rows":           n_rows,
            "nnz":              nnz,
            "n_levels":         n_levels,
            "max_level_width":  max_lw,
            "min_level_width":  min_lw,
            "n_runs":           len(gflops),
            "median_gflops":    float(np.median(gflops)),
            "mean_gflops":      float(np.mean(gflops)),
            "std_gflops":       float(np.std(gflops, ddof=1)) if len(gflops) > 1 else 0.0,
            "iqr_gflops":       float(q3 - q1),
            "cv_pct":           float(100.0 * np.std(gflops, ddof=1) / np.mean(gflops))
                                 if len(gflops) > 1 and np.mean(gflops) > 0 else 0.0,
            "min_gflops":       float(np.min(gflops)),
            "max_gflops":       float(np.max(gflops)),
            "q1_gflops":        float(q1),
            "q3_gflops":        float(q3),
        })

    abs_order  = {a: i for i, a in enumerate(ALL_ABSTRACTIONS)}
    mtyp_order = {m: i for i, m in enumerate(MATRIX_TYPES)}
    size_order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_ord_abs"]  = result["abstraction"].map(abs_order)
    result["_ord_mtyp"] = result["matrix_type"].map(mtyp_order)
    result["_ord_size"] = result["problem_size"].map(size_order)
    result = result.sort_values(["_ord_mtyp", "_ord_size", "_ord_abs"]) \
                   .drop(columns=["_ord_abs", "_ord_mtyp", "_ord_size"]) \
                   .reset_index(drop=True)
    return result


# ── Efficiency, parallelism_ratio, flags ──────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    native_rows = stats[stats["abstraction"] == "native"].set_index(
        ["matrix_type", "problem_size"])

    if native_rows.empty:
        print("  WARNING: native baseline not found — efficiency not computed",
              file=sys.stderr)
        stats["efficiency"]         = np.nan
        stats["eff_gt1_flag"]       = False
        stats["flag_deep_profiling"] = False
        stats["ppc_tier"]           = "unknown"
        stats["parallelism_ratio"]  = np.nan
        return stats

    def native_median(mtype: str, size_label: str) -> float:
        key = (mtype, size_label)
        if key in native_rows.index:
            return float(native_rows.loc[key, "median_gflops"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        nm = native_median(row["matrix_type"], row["problem_size"])
        if row["abstraction"] == "native" or np.isnan(nm) or nm == 0:
            eff = 1.0 if row["abstraction"] == "native" else np.nan
        else:
            eff = row["median_gflops"] / nm
        efficiencies.append(eff)

    stats["efficiency"] = efficiencies

    # eff_gt1_flag: abstraction faster than native baseline (unexpected, warrants investigation)
    stats["eff_gt1_flag"] = (
        (stats["abstraction"] != "native") &
        (stats["efficiency"] > 1.0)
    )

    # deep_profiling flag: efficiency < 0.85 (§9.5 protocol)
    # For SpTRSV: expected triggers are abstractions with high per-level dispatch cost
    # (P001 Launch Overhead, multiplied by n_levels) and small level widths.
    stats["flag_deep_profiling"] = (
        (stats["abstraction"] != "native") &
        (stats["efficiency"] < 0.85)
    )

    # PPC tier (project_spec §9.4)
    def tier(eff):
        if np.isnan(eff):   return "unknown"
        if eff >= 0.80:     return "excellent"
        if eff >= 0.60:     return "acceptable"
        return "poor"

    stats["ppc_tier"] = [
        "native" if row["abstraction"] == "native" else tier(row["efficiency"])
        for _, row in stats.iterrows()
    ]

    # parallelism_ratio: max_level_width / n_rows
    # Tests whether low parallelism ratio (shallow DAG with few parallel rows per level)
    # predicts low efficiency. Key scatter plot axis for fig21.
    stats["parallelism_ratio"] = stats.apply(
        lambda r: float(r["max_level_width"]) / float(r["n_rows"])
                  if r["n_rows"] > 0 else np.nan,
        axis=1
    )

    return stats


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E5 SpTRSV Summary (median GFLOP/s, efficiency vs native):")
    print("Note: SpTRSV is latency-bound — binding constraint is n_levels (serial depth),")
    print("      not bandwidth. GFLOP/s will be << bandwidth-limited peak.")
    for matrix_type in MATRIX_TYPES:
        sub_m = stats[stats["matrix_type"] == matrix_type]
        if sub_m.empty:
            continue
        print()
        print(f"  Matrix type: {matrix_type}")
        print("-" * 110)
        for size_label in ["small", "medium", "large"]:
            sub = sub_m[sub_m["problem_size"] == size_label]
            if sub.empty:
                continue
            # Print level-set metadata (same for all abstractions at this size)
            nl_row = sub.iloc[0]
            print(f"\n  Problem size: {size_label} (N={nl_row['n_rows']} rows, "
                  f"n_levels={nl_row['n_levels']}, "
                  f"max_lw={nl_row['max_level_width']}, "
                  f"par_ratio={nl_row['parallelism_ratio']:.3f})")
            print(f"  {'Abstraction':18s} {'Median GFLOP/s':>15s} {'IQR':>8s} {'Eff':>7s} "
                  f"{'CV%':>6s} {'Tier':>12s} {'Flags':>8s}")
            print(f"  {'-'*18} {'-'*15} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*8}")
            for _, row in sub.iterrows():
                flags = ""
                if row.get("flag_deep_profiling", False): flags += "⚑"
                if row.get("eff_gt1_flag", False):        flags += ">1"
                eff_str = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) else "  n/a "
                print(f"  {row['abstraction']:18s} {row['median_gflops']:>15.4f} "
                      f"{row['iqr_gflops']:>8.4f} {eff_str:>7s} "
                      f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s} {flags:>8s}")

    flagged = stats[stats.get("flag_deep_profiling", pd.Series(dtype=bool)).astype(bool)]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['matrix_type']:35s} {row['problem_size']:8s} "
                  f"eff={row['efficiency']:.4f}  n_levels={row['n_levels']}")
        print()
        print("  Expected root causes:")
        print("    julia: P001 Launch Overhead × n_levels (n_levels launches per solve)")
        print("    kokkos/raja: per-level dispatch overhead (Kokkos::fence / RAJA::synchronize)")
        print("    lower_triangular_random: irregular level widths → load imbalance (P007)")

    gt1 = stats[stats.get("eff_gt1_flag", pd.Series(dtype=bool)).astype(bool)]
    if not gt1.empty:
        print()
        print("  >1 Abstraction faster than native baseline:")
        for _, row in gt1.iterrows():
            print(f"    {row['abstraction']:18s} {row['matrix_type']:35s} {row['problem_size']:8s} "
                  f"eff={row['efficiency']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e5] Loading raw CSVs ...")
    raw = load_e5_csvs()

    print("[process_e5] Filtering ...")
    clean = filter_clean(raw)

    print("[process_e5] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e5] Computing efficiency vs native baseline ...")
    stats = compute_efficiency(stats)

    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e5_sptrsv_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e5] Saved → {out_path}")

    print_report(stats)

    n_flagged = int(stats.get("flag_deep_profiling", pd.Series(dtype=bool)).sum())
    n_gt1     = int(stats.get("eff_gt1_flag", pd.Series(dtype=bool)).sum())
    print(f"\n[process_e5] {n_flagged} configuration(s) flagged for deep profiling.")
    print(f"[process_e5] {n_gt1} configuration(s) with efficiency > 1.0 (eff_gt1_flag).")
    if n_flagged > 0:
        print("[process_e5] Run nsys/ncu on flagged configs per §9.5 protocol.")
        print("[process_e5] Key: compare per-level launch overhead across abstractions.")
        print("[process_e5]      julia overhead = n_levels × @cuda_dispatch (~0.3ms each)")
        print("[process_e5]      kokkos overhead = n_levels × Kokkos::fence")
        print("[process_e5]      raja overhead = n_levels × RAJA::synchronize")


if __name__ == "__main__":
    main()
