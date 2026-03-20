#!/usr/bin/env python3
"""
plot_e7.py — E7 N-Body figures.

fig13_e7_efficiency.png — grouped bar chart: efficiency vs native_notile
  x-axis: abstraction × kernel (native_notile, native_tile, kokkos, raja, julia)
  groups: problem size (small=4K, medium=32K, large=256K)
  y-axis: efficiency (median GFLOP/s / native_notile median GFLOP/s)
  Horizontal thresholds: excellent (0.95), acceptable (0.85), marginal (0.70)

fig14_e7_gflops.png — absolute GFLOP/s bar chart
  Same layout as fig13 but y-axis = median GFLOP/s.

Output: figures/e7/
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e7")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e7_nbody_summary.csv")

# ── Visual style ───────────────────────────────────────────────────────────────
SIZE_COLORS = {
    "small":  "#4878d0",
    "medium": "#ee854a",
    "large":  "#6acc65",
}
SIZE_LABELS = {"small": "S(4K)", "medium": "M(32K)", "large": "L(256K)"}

# Canonical abstraction order and display labels
ABS_ORDER = ["native_notile", "native_tile", "kokkos", "raja", "julia"]
ABS_LABELS = {
    "native_notile": "native\nnotile",
    "native_tile":   "native\ntile",
    "kokkos":        "Kokkos",
    "raja":          "RAJA",
    "julia":         "Julia",
}

# Tier thresholds
T_EXCELLENT  = 0.95
T_ACCEPTABLE = 0.85
T_MARGINAL   = 0.70


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot so index = abstraction (ordered), columns = size."""
    df2 = df.copy()
    # Use abstraction as row key (single kernel per abstraction for E7)
    df2["abs_key"] = df2["abstraction"]
    pivot = df2.pivot_table(index="abs_key", columns="problem_size",
                            values=value_col, aggfunc="first")
    # Reindex to canonical order (skip missing)
    present = [a for a in ABS_ORDER if a in pivot.index]
    return pivot.reindex(present)


def fig13_efficiency(df: pd.DataFrame):
    pivot = build_pivot(df, "efficiency_vs_native_notile")
    sizes = [s for s in ["small", "medium", "large"] if s in pivot.columns]

    n_abs   = len(pivot)
    n_sizes = len(sizes)
    width   = 0.22
    x       = np.arange(n_abs)
    offsets = np.linspace(-(n_sizes-1)*width/2, (n_sizes-1)*width/2, n_sizes)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, sz in enumerate(sizes):
        vals = pivot[sz].values.astype(float)
        bars = ax.bar(x + offsets[i], vals, width,
                      color=SIZE_COLORS[sz], label=SIZE_LABELS[sz],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    # Tier thresholds
    for thresh, label, color in [
        (T_EXCELLENT,  "excellent (0.95)", "#2ca02c"),
        (T_ACCEPTABLE, "acceptable (0.85)", "#ff7f0e"),
        (T_MARGINAL,   "marginal (0.70)",  "#d62728"),
    ]:
        ax.axhline(thresh, color=color, linewidth=1.0, linestyle="--", alpha=0.7,
                   label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([ABS_LABELS.get(a, a) for a in pivot.index], fontsize=9)
    ax.set_ylabel("Efficiency vs native_notile", fontsize=11)
    ax.set_ylim(0, 1.35)
    ax.set_title(
        "E7 N-Body — Abstraction Efficiency (RTX 5060 Laptop)\n"
        "Efficiency = median GFLOP/s / native_notile GFLOP/s",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    out = os.path.join(FIG_DIR, "fig13_e7_efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig14_gflops(df: pd.DataFrame):
    pivot = build_pivot(df, "median_gflops")
    pivot_iqr = build_pivot(df, "iqr_gflops")
    sizes = [s for s in ["small", "medium", "large"] if s in pivot.columns]

    n_abs   = len(pivot)
    n_sizes = len(sizes)
    width   = 0.22
    x       = np.arange(n_abs)
    offsets = np.linspace(-(n_sizes-1)*width/2, (n_sizes-1)*width/2, n_sizes)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, sz in enumerate(sizes):
        vals = pivot[sz].values.astype(float)
        iqrs = pivot_iqr[sz].values.astype(float)
        bars = ax.bar(x + offsets[i], vals, width,
                      yerr=iqrs/2, error_kw={"ecolor": "gray", "capsize": 3},
                      color=SIZE_COLORS[sz], label=SIZE_LABELS[sz],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + max(iqrs)*0.02 + 5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([ABS_LABELS.get(a, a) for a in pivot.index], fontsize=9)
    ax.set_ylabel("Median GFLOP/s", fontsize=11)
    ax.set_title(
        "E7 N-Body — Absolute Throughput (RTX 5060 Laptop)\n"
        "Error bars = IQR/2 over 30 timed reps",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    out = os.path.join(FIG_DIR, "fig14_e7_gflops.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e7] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e7] Run process_e7.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e7] Loaded {len(df)} rows")

    print("[plot_e7] Generating fig13 (efficiency) ...")
    fig13_efficiency(df)

    print("[plot_e7] Generating fig14 (GFLOP/s) ...")
    fig14_gflops(df)

    print(f"[plot_e7] Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
