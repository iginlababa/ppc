#!/usr/bin/env python3
"""
E3 3D Stencil — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs (hw_state_verified=1),
computes per-size statistics and efficiency relative to native baseline,
flags configurations for deep profiling, saves data/processed/e3_stencil_summary.csv.

E3 DESIGN DECISIONS
[D1] Problem sizes: small=32³, medium=128³, large=256³.
[D3] Primary metric: GB/s (memory-bound kernel, AI≈0.2 FLOP/byte).
[D7] Amended warmup: adaptive CV<2% — no fixed WARMUP_DROP needed since the
     kernel itself discards the warmup phase before emitting run_id rows.
     run_id in raw CSV starts at 1 (first timed rep), so all rows are clean.
[D6] experiment_id: stencil_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
Measurement protocol: locked-clock session, adaptive warmup (§9.1 amended).
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
PLATFORM = "nvidia_rtx5060_laptop"

ALL_ABSTRACTIONS = [
    "native",
    "kokkos",
    "raja",
    "sycl",
    "julia",
    "numba",
]

# No ceiling references for E3 (unlike E2 where cuBLAS was a ceiling).
CEILING_REFS: set = set()

# Abstractions with hard CC 12.0 incompatibility on RTX 5060 Laptop
UNSUPPORTED_CC120 = {"numba"}

PROBLEM_SIZES = {
    "small":  32,
    "medium": 128,
    "large":  256,
}


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e3_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ALL_ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"stencil_{abs_name}_{PLATFORM}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(df) == 0 and abs_name in UNSUPPORTED_CC120:
            print(f"  UNSUPPORTED_CC120: {abs_name} — Numba 0.64.0 does not support"
                  f" CC 12.0 (Blackwell); PTX 9.2 rejected by driver (max PTX 9.1)."
                  f" Platform limitation, not a SKIP.",
                  file=sys.stderr)
            continue
        print(f"  {abs_name:16s}: {len(df):4d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW path and run run_stencil.sh first")
    return pd.concat(frames, ignore_index=True)


# ── Filter ────────────────────────────────────────────────────────────────────
def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    # hw_state_verified == 1 only
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter:   {len(df):4d}/{n_total} rows kept")
    # No run_id drop needed: adaptive warmup is handled inside the binary;
    # only timed reps are written to CSV (run_id starts at 1 = first timed rep).
    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (abs_name, size_label), grp in df.groupby(["abstraction", "problem_size"]):
        gbs = grp["throughput_gbs"].dropna().to_numpy(dtype=float)
        if len(gbs) == 0:
            continue
        q1, q3 = np.percentile(gbs, [25, 75])
        n_val = PROBLEM_SIZES.get(size_label, 0)
        rows.append({
            "abstraction":    abs_name,
            "problem_size":   size_label,
            "n_grid":         n_val,
            "n_runs":         len(gbs),
            "median_gbs":     float(np.median(gbs)),
            "mean_gbs":       float(np.mean(gbs)),
            "std_gbs":        float(np.std(gbs, ddof=1)),
            "iqr_gbs":        float(q3 - q1),
            "cv_pct":         float(100.0 * np.std(gbs, ddof=1) / np.mean(gbs)),
            "min_gbs":        float(np.min(gbs)),
            "max_gbs":        float(np.max(gbs)),
            "q1_gbs":         float(q1),
            "q3_gbs":         float(q3),
            "is_ceiling_ref": abs_name in CEILING_REFS,
        })

    order      = {a: i for i, a in enumerate(ALL_ABSTRACTIONS)}
    size_order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_ord_abs"]  = result["abstraction"].map(order)
    result["_ord_size"] = result["problem_size"].map(size_order)
    result = result.sort_values(["_ord_size", "_ord_abs"]) \
                   .drop(columns=["_ord_abs", "_ord_size"]) \
                   .reset_index(drop=True)
    return result


# ── Efficiency and PPC ────────────────────────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    native_rows = stats[stats["abstraction"] == "native"].set_index("problem_size")
    if native_rows.empty:
        print("  WARNING: native baseline not found — efficiency not computed",
              file=sys.stderr)
        stats["efficiency"] = np.nan
        stats["flag_deep_profiling"] = False
        stats["ppc_tier"] = "unknown"
        return stats

    def native_median(size_label: str) -> float:
        if size_label in native_rows.index:
            return float(native_rows.loc[size_label, "median_gbs"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        nm = native_median(row["problem_size"])
        if row["abstraction"] == "native" or np.isnan(nm) or nm == 0:
            eff = 1.0 if row["abstraction"] == "native" else np.nan
        else:
            eff = row["median_gbs"] / nm
        efficiencies.append(eff)

    stats["efficiency"] = efficiencies

    # Deep profiling flag: non-ceiling with efficiency < 0.85
    stats["flag_deep_profiling"] = (
        (~stats["is_ceiling_ref"]) &
        (stats["efficiency"] < 0.85)
    )

    # PPC tier (project_spec §9.4)
    def tier(eff, is_ceiling):
        if is_ceiling:
            return "ceiling_ref"
        if np.isnan(eff):
            return "unknown"
        if eff >= 0.80:
            return "excellent"
        if eff >= 0.60:
            return "acceptable"
        return "poor"

    stats["ppc_tier"] = [
        tier(row["efficiency"], row["is_ceiling_ref"])
        for _, row in stats.iterrows()
    ]
    return stats


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E3 3D Stencil Summary (median GB/s, efficiency vs native):")
    print("-" * 84)
    for size_label in ["small", "medium", "large"]:
        sub = stats[stats["problem_size"] == size_label]
        if sub.empty:
            continue
        N = PROBLEM_SIZES[size_label]
        print(f"\n  Problem size: {size_label} (N={N}, grid={N}³)")
        print(f"  {'Abstraction':18s} {'Median GB/s':>12s} {'IQR':>8s} {'Eff':>7s} "
              f"{'CV%':>6s} {'Tier':>12s} {'Flag':>5s}")
        print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*5}")
        for _, row in sub.iterrows():
            flag     = "⚑" if row["flag_deep_profiling"] else ""
            ceil     = " [ceil]" if row["is_ceiling_ref"] else ""
            eff_str  = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) else "  n/a "
            print(f"  {row['abstraction']:18s} {row['median_gbs']:>12.2f} "
                  f"{row['iqr_gbs']:>8.3f} {eff_str:>7s} "
                  f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s}{ceil} {flag}")

    flagged = stats[stats["flag_deep_profiling"]]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['problem_size']:8s} "
                  f"eff={row['efficiency']:.4f}")
    else:
        print()
        print("  No configurations flagged for deep profiling.")
        print("  Expected: memory-bound stencil shows high efficiency across all")
        print("  abstractions — bandwidth bottleneck independent of abstraction layer.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e3] Loading raw CSVs ...")
    raw = load_e3_csvs()

    print("[process_e3] Filtering ...")
    clean = filter_clean(raw)

    print("[process_e3] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e3] Computing efficiency vs native baseline ...")
    stats = compute_efficiency(stats)

    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e3_stencil_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e3] Saved → {out_path}")

    print_report(stats)

    n_flagged = int(stats["flag_deep_profiling"].sum())
    if n_flagged > 0:
        print(f"\n[process_e3] {n_flagged} configuration(s) flagged for deep profiling.")
        print("[process_e3] Run nsys/ncu on flagged configs per §9.5 protocol.")
    else:
        print("\n[process_e3] No deep profiling required.")
        print("[process_e3] Hypothesis: memory-bound AI≈0.2 masks abstraction overhead.")


if __name__ == "__main__":
    main()
