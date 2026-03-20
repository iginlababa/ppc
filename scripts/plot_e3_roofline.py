#!/usr/bin/env python3
"""
E3 3D Stencil — multi-platform roofline plot.

Two-panel figure: left = NVIDIA RTX 5060 Laptop, right = AMD MI300X.
Each panel shows the roofline model with all abstractions overlaid.

Hardware parameters:
  NVIDIA RTX 5060 Laptop:
    Peak BW:   ~270 GB/s (GDDR7, 128-bit bus; measured ~270 GB/s from E1 STREAM)
    Peak FP64: ~261 GFLOP/s (FP32/64 ≈ 16.7 TFLOP/s / 64)
  AMD MI300X:
    Peak BW:   ~4010 GB/s (HBM3, measured from E1 STREAM native)
    Peak FP64: ~163,400 GFLOP/s (163.4 TFLOP/s datasheet, gfx942)

AI = STENCIL_FLOP_PER_CELL / STENCIL_BYTES_PER_CELL = 13/64 ≈ 0.203 FLOP/byte.
Ridgeline: NVIDIA ≈ 261/270 ≈ 0.97 FLOP/byte; AMD ≈ 163400/4010 ≈ 40.7 FLOP/byte.
Both platforms are firmly memory-bound at AI ≈ 0.203.

Output: figures/e3/fig_e3_roofline_multiplatform.pdf
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
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e3")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e3_stencil_summary.csv")

# ── Hardware parameters ───────────────────────────────────────────────────────
HW = {
    "nvidia_rtx5060_laptop": {
        "label":         "NVIDIA RTX 5060 Laptop",
        "peak_bw_gbs":   270.0,     # GB/s — E1 STREAM native measurement
        "peak_fp64_gflops": 261.0,  # GFLOP/s — FP32/64 ≈ 16700/64
    },
    "amd_mi300x": {
        "label":         "AMD MI300X",
        "peak_bw_gbs":   4010.0,      # GB/s — E1 STREAM native measurement
        "peak_fp64_gflops": 163400.0, # GFLOP/s — datasheet gfx942
    },
}

# Arithmetic intensity for 7-point stencil (D3): 13 FLOP / 64 bytes
AI_STENCIL = 13.0 / 64.0  # ≈ 0.203 FLOP/byte

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "sycl":   "#9467bd",
    "julia":  "#d62728",
    "numba":  "#8c564b",
}
MARKERS = {
    "small":  "o",
    "medium": "s",
    "large":  "^",
}

SIZE_ORDER        = ["small", "medium", "large"]
ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]
PLATFORM_ORDER    = ["nvidia_rtx5060_laptop", "amd_mi300x"]

STYLE = {
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.size":         10,
}


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found. Run process_e3.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(SUMMARY_CSV)


def plot_platform_panel(ax, df_plat: pd.DataFrame, hw: dict):
    peak_bw    = hw["peak_bw_gbs"]
    peak_fp64  = hw["peak_fp64_gflops"]

    # ── Roofline boundaries ───────────────────────────────────────────────────
    ai_range  = np.logspace(-2, 3, 500)
    mem_bound = peak_bw * ai_range
    comp_ceil = np.full_like(ai_range, peak_fp64)
    roofline  = np.minimum(mem_bound, comp_ceil)

    ax.plot(ai_range, roofline, "k-", linewidth=2.0,
            label="Roofline (HW bound)", zorder=5)
    ax.axvline(peak_fp64 / peak_bw, color="grey",
               linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axvline(AI_STENCIL, color="steelblue", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Stencil AI ≈ {AI_STENCIL:.3f} FLOP/byte")

    # ── Plot each (abstraction, size) point ───────────────────────────────────
    legend_abs: dict = {}
    legend_sz:  dict = {}

    for _, row in df_plat.iterrows():
        abs_name   = row["abstraction"]
        size_label = row["problem_size"]
        gflops     = row["median_gbs"] * AI_STENCIL
        if pd.isna(gflops) or gflops == 0:
            continue
        color  = COLORS.get(abs_name, "grey")
        marker = MARKERS.get(size_label, "o")
        ax.scatter(AI_STENCIL, gflops, color=color, marker=marker,
                   s=90, zorder=10, edgecolors="white", linewidths=0.5)
        if abs_name not in legend_abs:
            legend_abs[abs_name] = plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markersize=9, label=abs_name)
        if size_label not in legend_sz:
            legend_sz[size_label] = plt.Line2D(
                [0], [0], marker=marker, color="grey",
                markersize=9, linestyle="None", label=size_label)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"{hw['label']}\n"
                 f"Peak BW={peak_bw:.0f} GB/s, Peak FP64={peak_fp64:.0f} GFLOP/s")

    # Annotate roofline slopes
    ax.annotate(f"Peak BW\n({peak_bw:.0f} GB/s)",
                xy=(0.05, peak_bw * 0.05),
                xytext=(0.1, peak_bw * 0.15),
                fontsize=7, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))

    # Two-column legend: abstractions + sizes
    handles = list(legend_abs.values()) + list(legend_sz.values())
    ax.legend(handles=handles, loc="upper left", framealpha=0.9,
              fontsize=8, ncol=2)

    return legend_abs, legend_sz


def main():
    df = load_summary()

    # Only plot platforms we have data for, in defined order
    available = [p for p in PLATFORM_ORDER if p in df["platform"].unique()]
    if not available:
        print("ERROR: no platform data in summary CSV.", file=sys.stderr)
        sys.exit(1)

    # ── Single-platform fallback (NVIDIA-only data is still valid) ────────────
    n_panels = len(available)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(8 * n_panels, 5.5),
                                 squeeze=False)
        axes = axes[0]  # shape: (n_panels,)

        for ax, platform in zip(axes, available):
            df_plat = df[df["platform"] == platform].copy()
            hw = HW.get(platform, {
                "label": platform,
                "peak_bw_gbs": 1.0,
                "peak_fp64_gflops": 1.0,
            })
            plot_platform_panel(ax, df_plat, hw)

        fig.suptitle("E3 3D Stencil — Roofline Analysis", fontsize=13, y=1.01)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig_e3_roofline_multiplatform.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_e3_roofline] Saved {out}")

    # ── Also regenerate the original single-panel NVIDIA figure ──────────────
    nvidia_key = "nvidia_rtx5060_laptop"
    if nvidia_key in available:
        df_nv = df[df["platform"] == nvidia_key].copy()
        hw_nv = HW[nvidia_key]
        with plt.rc_context(STYLE):
            fig2, ax2 = plt.subplots(figsize=(8, 5.5))
            plot_platform_panel(ax2, df_nv, hw_nv)
            ax2.set_title(f"E3 3D Stencil — Roofline Analysis\n{hw_nv['label']}")
            fig2.tight_layout()
            out2 = os.path.join(FIG_DIR, "fig12_e3_roofline.png")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"[plot_e3_roofline] Saved {out2}")


if __name__ == "__main__":
    main()
