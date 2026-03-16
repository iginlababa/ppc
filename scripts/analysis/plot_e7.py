#!/usr/bin/env python3
"""
E7 N-Body — efficiency and tiling figures.

fig13_e7_efficiency_by_size.png  — grouped bar: efficiency vs native_notile
                                   per (kernel, size); shows julia vs tile vs notile
fig14_e7_tile_vs_notile.png      — grouped bar: native tile / notile GFLOP/s
                                   by size (direct P006 evidence)

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

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e7")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e7_nbody_summary.csv")

SIZE_ORDER    = ["small", "medium", "large"]
SIZE_LABELS   = ["small\n(N=4K)", "medium\n(N=32K)", "large\n(N=256K)"]

# Bars shown in fig13: all rows except native_notile (baseline = 1.0)
# Groups: (kernel, abstraction) combos that are not the baseline
GROUPS = [
    ("notile", "julia"),   # Julia vs native_notile
    ("tile",   "native"),  # tile shows P006 benefit (may be > or < 1.0)
]
GROUP_LABELS = {
    ("notile", "julia"):  "julia/notile",
    ("tile",   "native"): "native/tile",
}
COLORS = {
    ("notile", "julia"):  "#d62728",
    ("tile",   "native"): "#e87722",
}


# ── Figure 13: efficiency grouped bar chart ────────────────────────────────────
def fig13_efficiency_bars(df: pd.DataFrame):
    x = np.arange(len(SIZE_ORDER))
    width = 0.30
    offsets = [-0.15, 0.15]   # one offset per group

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for gi, (kv, abs_name) in enumerate(GROUPS):
        vals, errs = [], []
        for sz in SIZE_ORDER:
            row = df[(df["kernel"] == kv) & (df["abstraction"] == abs_name) &
                     (df["problem_size"] == sz)]
            if row.empty:
                vals.append(0.0); errs.append(0.0)
                continue
            eff = float(row["efficiency"].iloc[0])
            # Error bar: IQR / native_median
            native = df[(df["kernel"] == "notile") & (df["abstraction"] == "native") &
                        (df["problem_size"] == sz)]
            nm  = float(native["median_gflops"].iloc[0]) if not native.empty else 1.0
            err = float(row["iqr_gflops"].iloc[0]) / nm if nm > 0 else 0.0
            vals.append(eff); errs.append(err)

        rects = ax.bar(
            x + offsets[gi], vals, width,
            color=COLORS[(kv, abs_name)], alpha=0.85,
            yerr=errs, capsize=3,
            error_kw={"linewidth": 0.8, "color": "dimgray"},
            label=GROUP_LABELS[(kv, abs_name)]
        )
        for rect, eff in zip(rects, vals):
            if eff > 0 and not np.isnan(eff):
                ax.text(rect.get_x() + rect.get_width() / 2.0, eff + 0.02,
                        f"{eff:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    ax.axhline(1.0,  color="black",   linewidth=0.9, linestyle="--", label="native_notile (1.0)")
    ax.axhline(0.85, color="#888888", linewidth=0.7, linestyle=":",  label="deep-profile (0.85)")

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_LABELS, fontsize=9)
    ax.set_xlabel("Problem size", fontsize=10)
    ax.set_ylabel("Efficiency vs native_notile", fontsize=10)
    max_val = max((r for vals_list in [
        [df[(df["kernel"]==kv) & (df["abstraction"]==a) &
            (df["problem_size"]==sz)]["efficiency"].iloc[0]
         for sz in SIZE_ORDER
         if not df[(df["kernel"]==kv) & (df["abstraction"]==a) &
                   (df["problem_size"]==sz)].empty]
        for kv, a in GROUPS
    ] for r in vals_list), default=1.4)
    ax.set_ylim(0, max(1.4, max_val + 0.25))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "E7 N-Body — Abstraction & Tiling Efficiency vs native_notile\n"
        "(LJ force, r_cut=2.5σ, FCC lattice; notile=neighbor-list, tile=all-pairs shared-mem)",
        fontsize=10, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig13_e7_efficiency_by_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 14: tile vs notile native GFLOP/s comparison ───────────────────────
def fig14_tile_vs_notile(df: pd.DataFrame):
    native = df[df["abstraction"] == "native"]
    notile = native[native["kernel"] == "notile"]
    tile   = native[native["kernel"] == "tile"]

    if notile.empty or tile.empty:
        print("  fig14: missing notile or tile data — skipping", file=sys.stderr)
        return

    x = np.arange(len(SIZE_ORDER))
    width = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"wspace": 0.35})

    # Left: absolute GFLOP/s
    ax = axes[0]
    nt_vals, ti_vals = [], []
    nt_errs, ti_errs = [], []
    for sz in SIZE_ORDER:
        r_nt = notile[notile["problem_size"] == sz]
        r_ti = tile[tile["problem_size"] == sz]
        nt_vals.append(float(r_nt["median_gflops"].iloc[0]) if not r_nt.empty else 0.0)
        ti_vals.append(float(r_ti["median_gflops"].iloc[0]) if not r_ti.empty else 0.0)
        nt_errs.append(float(r_nt["iqr_gflops"].iloc[0])    if not r_nt.empty else 0.0)
        ti_errs.append(float(r_ti["iqr_gflops"].iloc[0])    if not r_ti.empty else 0.0)

    ax.bar(x - 0.15, nt_vals, width, color="#2b4590", alpha=0.85,
           yerr=nt_errs, capsize=3, error_kw={"linewidth": 0.8},
           label="notile (neighbor-list)")
    ax.bar(x + 0.15, ti_vals, width, color="#e87722", alpha=0.85,
           yerr=ti_errs, capsize=3, error_kw={"linewidth": 0.8},
           label="tile (all-pairs, P006)")

    ax.set_xticks(x); ax.set_xticklabels(SIZE_LABELS, fontsize=9)
    ax.set_ylabel("GFLOP/s (median ± IQR)", fontsize=10)
    ax.set_xlabel("Problem size", fontsize=10)
    ax.set_title("Absolute GFLOP/s: tile vs notile", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Right: speedup tile/notile
    ax2 = axes[1]
    speedups = []
    for nt, ti in zip(nt_vals, ti_vals):
        speedups.append(ti / nt if nt > 0 else 0.0)
    bars = ax2.bar(x, speedups, 0.5, color=["#2b4590", "#e87722", "#2ca02c"],
                   alpha=0.85, edgecolor="white")
    ax2.axhline(1.0, color="black", linewidth=0.9, linestyle="--", label="parity (1.0×)")
    for rect, sp in zip(bars, speedups):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, sp + 0.01,
                 f"{sp:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(SIZE_LABELS, fontsize=9)
    ax2.set_ylabel("tile GFLOP/s / notile GFLOP/s", fontsize=10)
    ax2.set_xlabel("Problem size", fontsize=10)
    ax2.set_title("P006 Tiling Speedup (tile / notile)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "E7 N-Body — P006 Shared-Memory Tiling Benefit (native CUDA)\n"
        "(tile=cooperative warp loading, TILE_SIZE=32; notile=neighbor-list global-mem reads)",
        fontsize=10, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig14_e7_tile_vs_notile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e7] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e7] Run process_e7.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e7] Loaded {len(df)} rows from {SUMMARY_CSV}")

    print("[plot_e7] Generating fig13 (efficiency bar chart) ...")
    fig13_efficiency_bars(df)

    print("[plot_e7] Generating fig14 (tile vs notile P006) ...")
    fig14_tile_vs_notile(df)

    print(f"[plot_e7] All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
