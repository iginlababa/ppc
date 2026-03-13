#!/usr/bin/env python3
"""
Stage 2: Compute Performance Portability Coefficient (PPC).

PPC(a, P) = |P| / sum_{p in P} (1 / E(a,p))
where E(a,p) = throughput(abstraction, p) / throughput(native, p)

Pennycook et al. (2019) formulation.  Efficiency is always relative to
measured native performance on each platform, not theoretical peak.

Single-platform note:
    When data covers only one platform, PPC(a, {p}) = E(a, p) by definition
    (H=1 harmonic mean).  This is numerically correct but does not capture
    portability across platforms.  The report flags this case explicitly.

Usage:
    python analysis/compute_ppc.py \\
        --input  data/raw/stream_nvidia_a100_20240115.csv \\
        --output data/processed/ppc_e1_nvidia_a100.csv \\
        --experiment E1 \\
        [--problem-size large] \\
        [--abstraction kokkos] \\
        [--profiling-queue data/processed/profiling_queue.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── PPC thresholds (project_spec.md §9.4) ─────────────────────────────────────
PPC_EXCELLENT        = 0.80
PPC_ACCEPTABLE       = 0.60
DEEP_PROFILE_TRIGGER = 0.85   # flag if abstraction efficiency < 0.85 * native

# ── Experiment → kernel name mapping ──────────────────────────────────────────
# Populated from benchmarks/*/config.yaml at load time when available;
# hard-coded fallback ensures the script works without PyYAML installed.
_EXPERIMENT_TO_KERNEL: dict[str, str] = {
    "E1": "stream",
    "E2": "dgemm",
    "E3": "stencil",
    "E4": "spmv",
    "E5": "sptrsv",
    "E6": "bfs",
    "E7": "nbody",
}

# Peak bandwidth (GB/s) from benchmarks/stream/config.yaml, used for roofline %
_PEAK_BW_GBS: dict[str, float] = {
    "nvidia_rtx5060_laptop": 288.0,
    "nvidia_a100":           2039.0,
    "amd_mi250x":            3277.0,
    "intel_pvc":             3276.0,
}

# ── Lazy config loader ─────────────────────────────────────────────────────────

def _load_experiment_map() -> dict[str, str]:
    """
    Try to enrich _EXPERIMENT_TO_KERNEL from benchmarks/*/config.yaml.
    Falls back silently if PyYAML is not installed or files are missing.
    """
    mapping = dict(_EXPERIMENT_TO_KERNEL)
    try:
        import yaml
        repo_root = Path(__file__).parent.parent
        for cfg_path in sorted(repo_root.glob("benchmarks/*/config.yaml")):
            try:
                cfg = yaml.safe_load(cfg_path.read_text())
                exp_id  = cfg.get("experiment_id", "")
                kernel  = cfg.get("kernel", "")
                if exp_id and kernel:
                    mapping[exp_id.upper()] = kernel
                    # Also load peak bandwidth if present
                    for plat, bw in cfg.get("peak_bandwidth_gbs", {}).items():
                        _PEAK_BW_GBS.setdefault(plat, float(bw))
            except Exception:
                pass
    except ImportError:
        pass
    return mapping


# ── CSV loader ─────────────────────────────────────────────────────────────────

def load_performance(path: Path) -> pd.DataFrame:
    """
    Load the performance CSV written by run_stream.sh or parse_results.py.

    Accepts two column-name variants for throughput:
      - "throughput"      (parse_results.py canonical)
      - "throughput_gbs"  (older run scripts)
    """
    df = pd.read_csv(path)

    # Normalise throughput column name
    if "throughput" not in df.columns and "throughput_gbs" in df.columns:
        df = df.rename(columns={"throughput_gbs": "throughput"})

    # Normalise execution_time column name
    if "execution_time_ms" not in df.columns and "time_ms" in df.columns:
        df = df.rename(columns={"time_ms": "execution_time_ms"})

    required = {"kernel", "abstraction", "platform", "problem_size",
                "run_id", "throughput"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # Drop rows where throughput is not a positive finite number
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    n_before = len(df)
    df = df[df["throughput"].notna() & (df["throughput"] > 0)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  WARNING: dropped {n_dropped} rows with invalid throughput")

    return df


# ── Aggregation ────────────────────────────────────────────────────────────────

def compute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30 per-run rows to one median row per (kernel, abs, platform, size)."""
    group_keys = ["kernel", "abstraction", "platform", "problem_size"]
    agg = (
        df.groupby(group_keys)["throughput"]
        .agg(
            throughput_median="median",
            throughput_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
            throughput_min="min",
            throughput_max="max",
            n_runs="count",
        )
        .reset_index()
    )
    return agg


# ── Efficiency ─────────────────────────────────────────────────────────────────

def compute_efficiency(agg: pd.DataFrame) -> pd.DataFrame:
    """Divide each abstraction's throughput by the native baseline on the same platform."""
    native = (
        agg[agg["abstraction"] == "native"]
        [["kernel", "platform", "problem_size", "throughput_median"]]
        .rename(columns={"throughput_median": "native_throughput"})
    )
    merged = agg.merge(native, on=["kernel", "platform", "problem_size"], how="left")
    merged["efficiency"] = merged["throughput_median"] / merged["native_throughput"]
    return merged


# ── PPC ────────────────────────────────────────────────────────────────────────

def compute_ppc(eff: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PPC for each (kernel, abstraction, problem_size) across all platforms.

    PPC(a, P) = |P| / Σ_{p∈P} (1 / E(a,p))

    When |P| == 1 the result equals E(a,p) — the single-platform efficiency.
    """
    results = []
    for (kernel, abstraction, problem_size), grp in eff.groupby(
        ["kernel", "abstraction", "problem_size"]
    ):
        if abstraction == "native":
            continue

        valid = grp.dropna(subset=["efficiency"])
        if valid.empty:
            continue

        H   = len(valid)
        ppc = H / (1.0 / valid["efficiency"]).sum()

        results.append(
            {
                "kernel":         kernel,
                "abstraction":    abstraction,
                "problem_size":   problem_size,
                "ppc":            round(ppc, 4),
                "n_platforms":    H,
                "platforms":      ",".join(sorted(valid["platform"].tolist())),
                "ppc_tier":       _ppc_tier(ppc),
                "min_efficiency": round(valid["efficiency"].min(), 4),
                "max_efficiency": round(valid["efficiency"].max(), 4),
            }
        )

    return pd.DataFrame(results)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ppc_tier(ppc: float) -> str:
    if ppc >= PPC_EXCELLENT:
        return "excellent"
    if ppc >= PPC_ACCEPTABLE:
        return "acceptable"
    return "poor"


def flag_for_profiling(eff: pd.DataFrame) -> pd.DataFrame:
    """Return rows where abstraction efficiency < DEEP_PROFILE_TRIGGER."""
    flagged = eff[
        (eff["abstraction"] != "native")
        & (eff["efficiency"] < DEEP_PROFILE_TRIGGER)
    ].copy()
    flagged["profiling_reason"] = (
        "efficiency=" + flagged["efficiency"].round(3).astype(str)
        + " < " + str(DEEP_PROFILE_TRIGGER)
    )
    return flagged


# ── Per-platform efficiency table ──────────────────────────────────────────────

def _efficiency_table(eff: pd.DataFrame, problem_size: str | None) -> None:
    """Print a per-platform, per-abstraction efficiency breakdown."""
    df = eff[eff["abstraction"] != "native"].copy()
    if problem_size:
        df = df[df["problem_size"] == problem_size]
    if df.empty:
        return

    platforms = sorted(df["platform"].unique())
    abstractions = sorted(df["abstraction"].unique())
    print(f"\n{'Abstraction':<14}", end="")
    for p in platforms:
        pct_label = f"  {p} (eff%)"
        print(f"{pct_label:>22}", end="")
    print()
    print("-" * (14 + 22 * len(platforms)))

    for abs_name in abstractions:
        print(f"{abs_name:<14}", end="")
        for p in platforms:
            row = df[(df["abstraction"] == abs_name) & (df["platform"] == p)]
            if row.empty:
                print(f"{'—':>22}", end="")
            else:
                e = row["efficiency"].iloc[0]
                bw = row["throughput_median"].iloc[0]
                peak = _PEAK_BW_GBS.get(p)
                if peak:
                    cell = f"{e:.3f}  {bw/peak*100:.1f}% pk"
                else:
                    cell = f"{e:.3f}  {bw:.0f} GB/s"
                print(f"{cell:>22}", end="")
        print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    exp_map = _load_experiment_map()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True,
                        help="Path to performance CSV (data/raw/stream_*.csv "
                             "or data/performance.csv)")
    parser.add_argument("--output", required=True,
                        help="Destination for ppc_results.csv")
    parser.add_argument("--experiment",
                        help="Filter to one experiment (e.g. E1).  "
                             f"Known: {', '.join(sorted(exp_map))}")
    parser.add_argument("--abstraction",
                        help="Filter to one abstraction (e.g. kokkos, raja)")
    parser.add_argument("--problem-size",
                        choices=["small", "medium", "large"],
                        help="Filter to one problem size.  "
                             "Omit to compute PPC for all sizes.")
    parser.add_argument("--profiling-queue",
                        default="data/processed/profiling_queue.csv",
                        help="Path to write configurations flagged for deep profiling "
                             "(default: data/processed/profiling_queue.csv)")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_performance(Path(args.input))
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"  Platforms:    {sorted(df['platform'].unique())}")
    print(f"  Abstractions: {sorted(df['abstraction'].unique())}")
    print(f"  Problem sizes:{sorted(df['problem_size'].unique())}")

    # ── Experiment filter → kernel filter ─────────────────────────────────────
    if args.experiment:
        exp_upper = args.experiment.upper()
        if exp_upper not in exp_map:
            print(f"ERROR: Unknown experiment '{args.experiment}'.  "
                  f"Known: {sorted(exp_map)}", file=sys.stderr)
            return 1
        kernel_name = exp_map[exp_upper]
        df = df[df["kernel"] == kernel_name]
        if df.empty:
            print(f"ERROR: No rows with kernel='{kernel_name}' "
                  f"after filtering for experiment {exp_upper}.", file=sys.stderr)
            return 1
        print(f"  Experiment {exp_upper} → kernel={kernel_name} "
              f"({len(df)} rows remain)")

    # ── Abstraction filter — always keep native as efficiency denominator ──────
    if args.abstraction:
        df = df[df["abstraction"].isin(["native", args.abstraction])]
        print(f"  Abstraction filter: {args.abstraction} ({len(df)} rows remain)")

    # ── Problem-size filter ───────────────────────────────────────────────────
    if args.problem_size:
        df = df[df["problem_size"] == args.problem_size]
        print(f"  Problem-size filter: {args.problem_size} ({len(df)} rows remain)")

    if df.empty:
        print("ERROR: No data rows remain after filtering.", file=sys.stderr)
        return 1

    # ── Compute ───────────────────────────────────────────────────────────────
    agg     = compute_medians(df)
    eff     = compute_efficiency(agg)
    ppc_df  = compute_ppc(eff)
    flagged = flag_for_profiling(eff)

    # ── Write PPC CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ppc_df.to_csv(out_path, index=False)
    print(f"\nPPC results → {out_path}  ({len(ppc_df)} rows)")

    # ── Write profiling queue ─────────────────────────────────────────────────
    queue_path = Path(args.profiling_queue)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(queue_path, index=False)
    print(f"Profiling queue → {queue_path}  ({len(flagged)} configs flagged)")

    # ── Print summary ─────────────────────────────────────────────────────────
    if ppc_df.empty:
        print("\nNo PPC rows computed — need at least one non-native abstraction "
              "with a matching native baseline on the same platform.")
        return 0

    n_platforms = eff["platform"].nunique()
    if n_platforms == 1:
        plat = eff["platform"].iloc[0]
        print(f"\nNOTE: Data covers only one platform ({plat}).  "
              f"PPC = efficiency (H=1).  Run on additional platforms for "
              f"meaningful portability scores.")

    # Per-size PPC table
    for sz in sorted(ppc_df["problem_size"].unique()):
        sub = ppc_df[ppc_df["problem_size"] == sz].sort_values("ppc", ascending=False)
        print(f"\nPPC — problem_size={sz}  "
              f"({'primary E1 result' if sz == 'large' else sz}):")
        print(f"  {'Abstraction':<14}  {'PPC':>6}  {'Tier':<10}  "
              f"{'Min eff':>8}  {'Max eff':>8}  {'Platforms'}")
        print("  " + "-" * 68)
        for _, row in sub.iterrows():
            print(f"  {row['abstraction']:<14}  {row['ppc']:>6.4f}  "
                  f"{row['ppc_tier']:<10}  {row['min_efficiency']:>8.4f}  "
                  f"{row['max_efficiency']:>8.4f}  {row['platforms']}")

    # Per-platform efficiency breakdown
    primary_sz = "large" if "large" in ppc_df["problem_size"].values else None
    print("\nPer-platform efficiency breakdown"
          + (f" (size={primary_sz}):" if primary_sz else ":"))
    _efficiency_table(eff, primary_sz)

    # Profiling candidates
    if not flagged.empty:
        print(f"\nConfigurations flagged for deep profiling "
              f"(efficiency < {DEEP_PROFILE_TRIGGER}):")
        cols = ["abstraction", "platform", "problem_size", "efficiency", "profiling_reason"]
        available = [c for c in cols if c in flagged.columns]
        print(flagged[available].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
