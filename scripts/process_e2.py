#!/usr/bin/env python3
"""
E2 DGEMM — data processing pipeline.

Loads raw per-abstraction CSVs for all configured platforms, filters clean
runs (hw_state_verified=1), computes per-size statistics and efficiency
relative to native baseline, flags configurations for deep profiling, and
saves data/processed/e2_dgemm_summary.csv.

E2 DESIGN DECISIONS
[D1] Large size: N=8192 (not 16384 from original spec — VRAM headroom on RTX 5060).
[D5] Ceiling references (native_cublas, native_rocblas, julia_cublas,
     julia_rocblas) are excluded from PPC computation but included in summary
     CSV with is_ceiling_ref=True.
[D6] raja_naive is a PPC abstraction (not a ceiling); its expected low
     efficiency vs native is the API Limitation finding.
[D7] experiment_id: dgemm_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}

Measurement protocols
─────────────────────
NVIDIA RTX 5060 Laptop (2026-03-14)
  - warmup=50; hw_state_verified is the sole clean-run gate.
  - Abstractions: native, native_cublas, raja_naive, julia_naive, julia_cublas, numba
  - numba: UNSUPPORTED_CC120 — no clean rows (PTX 9.2 rejected by driver)

AMD MI300X (2026-03-20)
  - warmup=50; hw_state_verified gate only (no clock-boost artifact).
  - Abstractions: native, native_rocblas, kokkos, raja_naive, sycl,
                  julia_naive, julia_rocblas
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

# ── Ceiling-reference abstractions ────────────────────────────────────────────
ALL_CEILING_REFS = {"native_cublas", "native_rocblas", "julia_cublas", "julia_rocblas"}

PROBLEM_SIZES = {"small": 1024, "medium": 4096, "large": 8192}

# ── Per-platform configuration ────────────────────────────────────────────────
PLATFORM_CONFIGS = {
    "nvidia_rtx5060_laptop": {
        "abstractions": [
            "native", "native_cublas",
            "kokkos", "raja_naive", "sycl",
            "julia_naive", "julia_cublas", "numba",
        ],
        "unsupported": {"numba"},   # UNSUPPORTED_CC120: Blackwell CC 12.0
        "warmup_drop": 5,
    },
    "amd_mi300x": {
        "abstractions": [
            "native", "native_rocblas",
            "kokkos", "raja_naive", "sycl",
            "julia_naive", "julia_rocblas",
        ],
        "unsupported": set(),
        "warmup_drop": 0,
    },
}


# ── Load raw CSVs for one platform ────────────────────────────────────────────
def load_platform_csvs(platform: str, cfg: dict) -> pd.DataFrame:
    frames = []
    for abs_name in cfg["abstractions"]:
        pattern = os.path.join(DATA_RAW, f"dgemm_{abs_name}_{platform}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            if abs_name not in cfg["unsupported"]:
                print(f"  WARNING [{platform}]: no CSV for {abs_name}",
                      file=sys.stderr)
            else:
                print(f"  UNSUPPORTED [{platform}]: {abs_name} — skipped",
                      file=sys.stderr)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(df) == 0 and abs_name in cfg["unsupported"]:
            print(f"  UNSUPPORTED_CC120 [{platform}]: {abs_name} — 0 rows",
                  file=sys.stderr)
            continue
        print(f"  {abs_name:20s}: {len(df):4d} rows from {len(files)} file(s)")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSV files for platform={platform}")
    combined = pd.concat(frames, ignore_index=True)
    combined["platform"] = platform
    return combined


# ── Filter ────────────────────────────────────────────────────────────────────
def filter_and_clean(df: pd.DataFrame, warmup_drop: int) -> pd.DataFrame:
    n_total = len(df)
    df = df[df["hw_state_verified"] == 1].copy()
    print(f"  hw_state=1 filter: {len(df):4d}/{n_total} rows kept")
    if warmup_drop > 0:
        df = df[df["run_id"] > warmup_drop].copy()
        print(f"  Drop run_id ≤ {warmup_drop}:  {len(df):4d} rows remain")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, platform: str, cfg: dict) -> pd.DataFrame:
    ceiling_refs = ALL_CEILING_REFS
    rows = []
    for (abs_name, size_label), grp in df.groupby(["abstraction", "problem_size"],
                                                   sort=False):
        gf = grp["throughput_gflops"].dropna().to_numpy(dtype=float)
        if len(gf) == 0:
            continue
        q1, q3 = np.percentile(gf, [25, 75])
        n_val = PROBLEM_SIZES.get(size_label, 0)
        rows.append({
            "platform":       platform,
            "abstraction":    abs_name,
            "problem_size":   size_label,
            "n_matrix":       n_val,
            "n_runs":         len(gf),
            "median_gflops":  float(np.median(gf)),
            "mean_gflops":    float(np.mean(gf)),
            "std_gflops":     float(np.std(gf, ddof=1)) if len(gf) > 1 else 0.0,
            "iqr_gflops":     float(q3 - q1),
            "cv_pct":         float(100.0 * np.std(gf, ddof=1) / np.mean(gf))
                              if len(gf) > 1 else 0.0,
            "min_gflops":     float(np.min(gf)),
            "max_gflops":     float(np.max(gf)),
            "q1_gflops":      float(q1),
            "q3_gflops":      float(q3),
            "is_ceiling_ref": abs_name in ceiling_refs,
        })

    abs_order  = {a: i for i, a in enumerate(cfg["abstractions"])}
    size_order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_oa"] = result["abstraction"].map(abs_order).fillna(99)
    result["_os"] = result["problem_size"].map(size_order)
    result = (result.sort_values(["_os", "_oa"])
                    .drop(columns=["_oa", "_os"])
                    .reset_index(drop=True))
    return result


# ── Efficiency and flags ───────────────────────────────────────────────────────
def compute_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    efficiencies, deep_flags, tiers = [], [], []

    def _tier(eff, is_ceiling):
        if is_ceiling:        return "ceiling_ref"
        if np.isnan(eff):     return "unknown"
        if eff >= 0.80:       return "excellent"
        if eff >= 0.60:       return "acceptable"
        return "poor"

    for (plat, size), grp in stats.groupby(["platform", "problem_size"]):
        native_row = grp[grp["abstraction"] == "native"]
        native_med = float(native_row["median_gflops"].iloc[0]) \
                     if not native_row.empty else np.nan
        if not native_row.empty:
            print(f"    [{plat}/{size}] native: {native_med:.1f} GFLOP/s")
        else:
            print(f"    [{plat}/{size}] WARNING: no native baseline",
                  file=sys.stderr)

        for idx in grp.index:
            row = stats.loc[idx]
            if row["abstraction"] == "native":
                eff = 1.0
            elif np.isnan(native_med) or native_med == 0:
                eff = np.nan
            else:
                eff = row["median_gflops"] / native_med
            is_ceil  = bool(row["is_ceiling_ref"])
            deep     = (not is_ceil) and (not np.isnan(eff)) and (eff < 0.85)
            efficiencies.append((idx, eff))
            deep_flags.append((idx, deep))
            tiers.append((idx, _tier(eff, is_ceil)))

    for idx, v in efficiencies:  stats.at[idx, "efficiency"]          = v
    for idx, v in deep_flags:    stats.at[idx, "flag_deep_profiling"] = v
    for idx, v in tiers:         stats.at[idx, "ppc_tier"]            = v
    return stats


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(stats: pd.DataFrame):
    print()
    print("E2 DGEMM Summary (median GFLOP/s, efficiency vs native):")
    for plat in stats["platform"].unique():
        print(f"\n  Platform: {plat}")
        print("-" * 90)
        psub = stats[stats["platform"] == plat]
        for size_label in ["small", "medium", "large"]:
            sub = psub[psub["problem_size"] == size_label]
            if sub.empty:
                continue
            N = PROBLEM_SIZES.get(size_label, 0)
            print(f"\n    [{size_label}] N={N}")
            hdr = f"    {'Abstraction':20s} {'Median':>10s} {'IQR':>8s} " \
                  f"{'Eff':>7s} {'CV%':>6s} {'Tier':>12s}"
            print(hdr)
            print(f"    {'-'*20} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*12}")
            for _, row in sub.iterrows():
                ceil_tag = " [ceil]" if row["is_ceiling_ref"] else ""
                flag_tag = " ⚑"     if row["flag_deep_profiling"] else ""
                eff_str  = f"{row['efficiency']:.4f}" \
                           if not np.isnan(row["efficiency"]) else "  n/a "
                print(f"    {row['abstraction']:20s} {row['median_gflops']:>10.2f} "
                      f"{row['iqr_gflops']:>8.3f} {eff_str:>7s} "
                      f"{row['cv_pct']:>6.2f} {row['ppc_tier']:>12s}"
                      f"{ceil_tag}{flag_tag}")


# ── Process one platform ───────────────────────────────────────────────────────
def process_platform(platform: str, cfg: dict) -> pd.DataFrame:
    print(f"\n[process_e2] ── {platform} ──")
    raw   = load_platform_csvs(platform, cfg)
    clean = filter_and_clean(raw, cfg["warmup_drop"])
    stats = compute_stats(clean, platform, cfg)
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_frames = []
    for platform, cfg in PLATFORM_CONFIGS.items():
        df = process_platform(platform, cfg)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    print("\n[process_e2] Computing efficiency ...")
    combined = compute_efficiency(combined)

    float_cols = combined.select_dtypes(include="float").columns
    combined[float_cols] = combined[float_cols].round(4)

    out_path = os.path.join(DATA_PROC, "e2_dgemm_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n[process_e2] Saved → {out_path}  ({len(combined)} rows)")

    print_report(combined)

    n_flagged = int(combined["flag_deep_profiling"].sum())
    if n_flagged > 0:
        print(f"\n[process_e2] {n_flagged} config(s) flagged for deep profiling.")


if __name__ == "__main__":
    main()
