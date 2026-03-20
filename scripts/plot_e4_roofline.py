#!/usr/bin/env python3
"""
E4 SpMV — multi-platform roofline plot.

fig18_e4_roofline.png — two-panel roofline (NVIDIA RTX 5060 | AMD MI300X).
Each panel: roofline model with all (abstraction, matrix_type, size) overlaid.

Hardware:
  NVIDIA RTX 5060: Peak BW ≈ 270 GB/s (E1 STREAM), Peak FP64 ≈ 261 GFLOP/s.
  AMD MI300X:      Peak BW ≈ 4010 GB/s (E1 STREAM), Peak FP64 ≈ 163,400 GFLOP/s.

AI per point = 2*nnz / bytes_moved (computed from actual nnz/nrows).
SpMV AI ≈ 0.13 FLOP/byte — firmly memory-bound on both platforms.

Output: figures/e4/fig18_e4_roofline.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e4")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e4_spmv_summary.csv")

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]
MATRIX_TYPES      = ["laplacian_2d", "random_sparse", "power_law"]

HW = {
    "nvidia_rtx5060": {
        "label":             "NVIDIA RTX 5060",
        "peak_bw_gbs":       270.0,
        "peak_fp64_gflops":  261.0,
    },
    "amd_mi300x": {
        "label":             "AMD MI300X",
        "peak_bw_gbs":       4010.0,
        "peak_fp64_gflops":  163400.0,
    },
}

PLATFORM_ORDER = ["nvidia_rtx5060", "amd_mi300x"]

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "sycl":   "#9467bd",
    "julia":  "#d62728",
    "numba":  "#8c564b",
}

MARKERS = {
    "laplacian_2d":  "o",
    "random_sparse": "s",
    "power_law":     "^",
}

SIZE_SIZES = {
    "small":  60,
    "medium": 90,
    "large":  130,
}

STYLE = {
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "font.size":        10,
}


def compute_ai(n_rows: int, nnz: int) -> float:
    """Arithmetic intensity for CSR SpMV: 2*nnz FLOP / bytes_moved."""
    if n_rows == 0 or nnz == 0:
        return 0.0
    bytes_moved = (nnz * 8.0          # values (FP64)
                   + nnz * 4.0        # col_idx (int32)
                   + (n_rows + 1) * 4.0  # row_ptr (int32)
                   + n_rows * 8.0     # x reads (FP64)
                   + n_rows * 8.0)    # y writes (FP64)
    return (2.0 * nnz) / bytes_moved


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found. Run process_e4.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(SUMMARY_CSV)


def plot_platform_panel(ax, df_plat: pd.DataFrame, hw: dict):
    peak_bw   = hw["peak_bw_gbs"]
    peak_fp64 = hw["peak_fp64_gflops"]

    ai_range   = np.logspace(-2, 3, 500)
    mem_bound  = peak_bw * ai_range
    comp_bound = np.full_like(ai_range, peak_fp64)
    roofline   = np.minimum(mem_bound, comp_bound)

    ax.plot(ai_range, roofline, "k-", linewidth=2.0, label="Roofline (HW bound)", zorder=5)
    ax.axvline(peak_fp64 / peak_bw, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)

    ai_vals = df_plat["ai"].dropna()
    if len(ai_vals) > 0:
        ai_min, ai_max = float(ai_vals.min()), float(ai_vals.max())
        ax.axvspan(ai_min * 0.9, ai_max * 1.1, alpha=0.07, color="steelblue",
                   label=f"SpMV AI [{ai_min:.3f}–{ai_max:.3f}]")

    legend_abs  = {}
    legend_mtyp = {}
    for _, row in df_plat.iterrows():
        abs_name   = row["abstraction"]
        mtype      = row["matrix_type"]
        size_label = row["problem_size"]
        gflops     = row["median_gflops"]
        ai         = row["ai"]
        if pd.isna(gflops) or gflops == 0 or pd.isna(ai) or ai == 0:
            continue
        color  = COLORS.get(abs_name, "grey")
        marker = MARKERS.get(mtype, "o")
        msize  = SIZE_SIZES.get(size_label, 80)
        ax.scatter(ai, gflops, color=color, marker=marker,
                   s=msize, zorder=10, edgecolors="white", linewidths=0.5, alpha=0.85)
        if abs_name not in legend_abs:
            legend_abs[abs_name] = plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markersize=9, label=abs_name)
        if mtype not in legend_mtyp:
            legend_mtyp[mtype] = plt.Line2D(
                [0], [0], marker=marker, color="grey",
                markersize=9, linestyle="None", label=mtype.replace("_", " "))

    size_handles = [
        plt.Line2D([0], [0], marker="o", color="grey",
                   markersize=int(np.sqrt(s) * 0.6), linestyle="None", label=sz)
        for sz, s in SIZE_SIZES.items()
    ]
    handles = list(legend_abs.values()) + list(legend_mtyp.values()) + size_handles
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"{hw['label']}\n"
                 f"Peak BW={peak_bw:.0f} GB/s, Peak FP64={peak_fp64:.0f} GFLOP/s")
    ax.annotate(f"Peak BW\n({peak_bw:.0f} GB/s)",
                xy=(0.05, peak_bw * 0.05),
                xytext=(0.1, peak_bw * 0.15),
                fontsize=7, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))


def main():
    df = load_summary()

    # Compute AI per row from actual n_rows and nnz
    if "n_rows" in df.columns and "nnz" in df.columns:
        df["ai"] = df.apply(
            lambda r: compute_ai(int(r["n_rows"]), int(r["nnz"])), axis=1)
    else:
        df["ai"] = 0.125
        print("  WARNING: n_rows/nnz columns not found; using approximate AI=0.125",
              file=sys.stderr)

    available = [p for p in PLATFORM_ORDER if p in df.get("platform", pd.Series()).unique()]
    if not available:
        # Legacy single-platform CSV (no platform column)
        available = ["nvidia_rtx5060"]
        df["platform"] = "nvidia_rtx5060"

    n_panels = len(available)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 6), squeeze=False)
        axes = axes[0]

        for ax, platform in zip(axes, available):
            df_plat = df[df["platform"] == platform].copy()
            hw = HW.get(platform, {
                "label": platform, "peak_bw_gbs": 1.0, "peak_fp64_gflops": 1.0})
            plot_platform_panel(ax, df_plat, hw)

        fig.suptitle("E4 SpMV — Roofline Analysis (AI ≈ 0.13 FLOP/byte, memory-bound)",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig18_e4_roofline.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_e4_roofline] Saved {out}")


if __name__ == "__main__":
    main()
