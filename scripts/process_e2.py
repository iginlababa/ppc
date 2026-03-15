#!/usr/bin/env python3
"""
E2 DGEMM — data processing pipeline.

Loads raw per-abstraction CSVs, filters clean runs (hw_state_verified=1),
removes warmup tail (first WARMUP_DROP run_ids), computes per-size statistics
and efficiency relative to native baseline, flags configurations for deep
profiling, saves data/processed/e2_dgemm_summary.csv.

E2 DESIGN DECISIONS
[D1] Large size: N=8192 (not 16384 from original spec — VRAM headroom on RTX 5060).
[D5] Ceiling references (native_cublas, julia_cublas) are excluded from PPC
     computation but included in summary CSV with is_ceiling_ref=True.
[D6] raja_naive is a PPC abstraction (not a ceiling); its expected low
     efficiency vs native is the API Limitation finding.
[D7] experiment_id: dgemm_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
Measurement protocol: locked-clock session, warmup-50 (§9.1), §5.5 RTX 5060.
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
PLATFORM    = "nvidia_rtx5060_laptop"
WARMUP_DROP = 5      # discard first N run_ids per (abstraction, size, batch)

# All abstractions in display order
ALL_ABSTRACTIONS = [
    "native",
    "native_cublas",   # ceiling reference
    "kokkos",
    "raja_naive",
    "sycl",
    "julia_naive",
    "julia_cublas",    # ceiling reference
    "numba",
]

# Ceiling references — excluded from PPC, shown separately in figures
CEILING_REFS = {"native_cublas", "julia_cublas"}

# Abstractions that participate in PPC computation (non-ceiling, non-native)
PPC_ABSTRACTIONS = [a for a in ALL_ABSTRACTIONS
                    if a != "native" and a not in CEILING_REFS]

# SKIP              = missing environment dependency (binary not built, missing
#                     package, etc.) — fixable by rebuilding or installing.
# UNSUPPORTED_CC120 = hard platform incompatibility — not fixable with current
#                     tooling. Specifically: Numba 0.64.0 predates Blackwell
#                     CC 12.0; libnvvm generates PTX 9.2 which driver 590.48.01
#                     rejects (max PTX 9.1). No pip-installable fix exists.
UNSUPPORTED_CC120 = {"numba"}   # abstractions with hard CC 12.0 incompatibility

PROBLEM_SIZES = {
    "small":  1024,
    "medium": 4096,
    "large":  8192,
}

# ── Load raw CSVs ─────────────────────────────────────────────────────────────
def load_e2_csvs() -> pd.DataFrame:
    frames = []
    for abs_name in ALL_ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"dgemm_{abs_name}_{PLATFORM}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no CSV for abstraction={abs_name}", file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(df) == 0 and abs_name in UNSUPPORTED_CC120:
            print(f"  UNSUPPORTED_CC120: {abs_name} — Numba 0.64.0 does not support"
                  f" CC 12.0 (Blackwell); PTX 9.2 rejected by driver (max PTX 9.1)."
                  f" No pip-installable fix. Platform limitation, not a SKIP.",
                  file=sys.stderr)
            continue   # no data rows — do not append empty frame
        print(f"  {abs_name:16s}: {len(df):4d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files found — check DATA_RAW path and run run_dgemm.sh first")
    return pd.concat(frames, ignore_index=True)


# ── Batch detection ───────────────────────────────────────────────────────────
def assign_batch_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["abstraction", "problem_size", "timestamp", "run_id"]) \
           .reset_index(drop=True)
    batch_col = []
    for _, grp in df.groupby(["abstraction", "problem_size"], sort=False):
        bid = 0
        prev_rid = None
        for rid in grp["run_id"]:
            if prev_rid is not None and rid <= prev_rid:
                bid += 1
            batch_col.append(bid)
            prev_rid = rid
    df["batch_id"] = batch_col
    return df


# ── Filter ────────────────────────────────────────────────────────────────────
def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)

    # 1. hw_state_verified == 1 only
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter:   {len(df):4d}/{n_total} rows kept")

    # 2. Drop first WARMUP_DROP run_ids per (abstraction, problem_size, batch)
    df = df[df["run_id"] > WARMUP_DROP].copy()
    print(f"  Drop run_id ≤ {WARMUP_DROP}:     {len(df):4d} rows remain")

    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (abs_name, size_label), grp in df.groupby(["abstraction", "problem_size"]):
        gf = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gf) == 0:
            continue
        q1, q3 = np.percentile(gf, [25, 75])
        n_val = PROBLEM_SIZES.get(size_label, 0)
        rows.append({
            "abstraction":      abs_name,
            "problem_size":     size_label,
            "n_matrix":         n_val,
            "n_runs":           len(gf),
            "median_gflops":    float(np.median(gf)),
            "mean_gflops":      float(np.mean(gf)),
            "std_gflops":       float(np.std(gf, ddof=1)),
            "iqr_gflops":       float(q3 - q1),
            "cv_pct":           float(100.0 * np.std(gf, ddof=1) / np.mean(gf)),
            "min_gflops":       float(np.min(gf)),
            "max_gflops":       float(np.max(gf)),
            "q1_gflops":        float(q1),
            "q3_gflops":        float(q3),
            "is_ceiling_ref":   abs_name in CEILING_REFS,
        })

    # Enforce display order
    order = {a: i for i, a in enumerate(ALL_ABSTRACTIONS)}
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

    # Native baseline median per size
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
            return float(native_rows.loc[size_label, "median_gflops"])
        return np.nan

    efficiencies = []
    for _, row in stats.iterrows():
        nm = native_median(row["problem_size"])
        if row["abstraction"] == "native" or np.isnan(nm) or nm == 0:
            eff = 1.0 if row["abstraction"] == "native" else np.nan
        else:
            eff = row["median_gflops"] / nm
        efficiencies.append(eff)

    stats["efficiency"] = efficiencies

    # Deep profiling flag: any non-ceiling abstraction with efficiency < 0.85
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
    print("E2 DGEMM Summary (median GFLOP/s, efficiency vs native):")
    print("-" * 80)
    for size_label in ["small", "medium", "large"]:
        sub = stats[stats["problem_size"] == size_label]
        if sub.empty:
            continue
        N = PROBLEM_SIZES[size_label]
        print(f"\n  Problem size: {size_label} (N={N})")
        print(f"  {'Abstraction':18s} {'Median':>10s} {'IQR':>8s} {'Eff':>7s} {'CV%':>6s} {'Tier':>12s} {'Flag':>5s}")
        print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*5}")
        for _, row in sub.iterrows():
            flag = "⚑" if row["flag_deep_profiling"] else ""
            ceil = " [ceil]" if row["is_ceiling_ref"] else ""
            eff_str = f"{row['efficiency']:.4f}" if not np.isnan(row["efficiency"]) else "  n/a "
            print(f"  {row['abstraction']:18s} {row['median_gflops']:>10.2f} "
                  f"{row['iqr_gflops']:>8.3f} {eff_str:>7s} "
                  f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s}{ceil} {flag}")

    flagged = stats[stats["flag_deep_profiling"]]
    if not flagged.empty:
        print()
        print("  ⚑ Configurations flagged for deep profiling (efficiency < 0.85):")
        for _, row in flagged.iterrows():
            print(f"    {row['abstraction']:18s} {row['problem_size']:8s} "
                  f"eff={row['efficiency']:.4f} — root cause: API Limitation (expected for raja_naive)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[process_e2] Loading raw CSVs ...")
    raw = load_e2_csvs()

    print("[process_e2] Assigning batch IDs ...")
    raw = assign_batch_ids(raw)

    print("[process_e2] Filtering ...")
    clean = filter_and_clean(raw)

    print("[process_e2] Computing statistics ...")
    stats = compute_stats(clean)

    print("[process_e2] Computing efficiency vs native baseline ...")
    stats = compute_efficiency(stats)

    # Round floats for readability
    float_cols = stats.select_dtypes(include="float").columns
    stats[float_cols] = stats[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e2_dgemm_summary.csv")
    stats.to_csv(out_path, index=False)
    print(f"[process_e2] Saved → {out_path}")

    print_report(stats)

    # Flag summary
    n_flagged = int(stats["flag_deep_profiling"].sum())
    if n_flagged > 0:
        print(f"\n[process_e2] {n_flagged} configuration(s) flagged for deep profiling.")
        print("[process_e2] Run nsys/ncu on flagged configs per §9.5 protocol.")


if __name__ == "__main__":
    main()
