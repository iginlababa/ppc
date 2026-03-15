#!/usr/bin/env python3
"""
E3 3D Stencil — roofline plot.

fig12_e3_roofline.png — roofline model with all abstractions overlaid.
  - Hardware: NVIDIA RTX 5060 Laptop
    - Peak FP64 TFLOP/s: 0.261 TFLOP/s (Blackwell, 4096 FP64 CUDA cores × boost)
      Note: RTX 5060 Laptop has heavily limited FP64 (1/64 of FP32).
      FP32 peak ≈ 16.7 TFLOP/s → FP64 peak ≈ 0.261 TFLOP/s.
    - Peak BW: 272 GB/s (GDDR7, 128-bit bus, effective measured from E1 STREAM ~270 GB/s)
  - Each abstraction plotted as a point: (AI, measured GFLOP/s) per problem size.
  - AI = STENCIL_FLOP_PER_CELL / STENCIL_BYTES_PER_CELL = 13/64 ≈ 0.203 FLOP/byte.
    All abstractions share the same AI — points differ only in measured performance.
  - Roofline ridgeline at AI = peak_gflops / peak_gbs ≈ 0.261e3 / 272 ≈ 0.96 FLOP/byte.
    Since AI ≈ 0.203 < 0.96, stencil is firmly memory-bound — expected to reach
    a fraction of peak BW, not peak compute.

Output: figures/e3/fig12_e3_roofline.png
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

# ── Hardware parameters (RTX 5060 Laptop) ─────────────────────────────────────
# Peak BW from E1 STREAM native: ~270 GB/s (locked-clock measurement).
# Peak FP64: RTX 5060 Laptop FP64 rate = FP32/64 ≈ 16.7e3/64 ≈ 261 GFLOP/s.
PEAK_BW_GBS    = 270.0   # GB/s — measured upper bound from E1
PEAK_FP64_GFLOPS = 261.0  # GFLOP/s — hardware ceiling

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

SIZE_ORDER = ["small", "medium", "large"]
ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]

STYLE = {
    "figure.dpi":     150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "font.size":        10,
}


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found. Run process_e3.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(SUMMARY_CSV)


def main():
    df = load_summary()

    # Derive GFLOP/s from GB/s using fixed AI
    # GFLOP/s = GB/s × AI (FLOP/byte)
    df["median_gflops"] = df["median_gbs"] * AI_STENCIL

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5.5))

        # ── Roofline boundaries ───────────────────────────────────────────────
        ai_range = np.logspace(-2, 2, 400)

        # Memory-bound slope: GFLOP/s = BW × AI
        mem_bound = PEAK_BW_GBS * ai_range
        # Compute-bound ceiling
        comp_bound = np.full_like(ai_range, PEAK_FP64_GFLOPS)
        # Roofline = min of the two
        roofline = np.minimum(mem_bound, comp_bound)

        ax.plot(ai_range, roofline, "k-", linewidth=2.0, label="Roofline (HW bound)", zorder=5)
        ax.axvline(PEAK_FP64_GFLOPS / PEAK_BW_GBS, color="grey",
                   linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(AI_STENCIL, color="steelblue", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Stencil AI ≈ {AI_STENCIL:.3f} FLOP/byte")

        # ── Plot each (abstraction, size) point ───────────────────────────────
        legend_abs = {}
        legend_sz  = {}
        for _, row in df.iterrows():
            abs_name  = row["abstraction"]
            size_label = row["problem_size"]
            gflops    = row["median_gflops"]
            if pd.isna(gflops) or gflops == 0:
                continue
            color  = COLORS.get(abs_name, "grey")
            marker = MARKERS.get(size_label, "o")
            sc = ax.scatter(AI_STENCIL, gflops, color=color, marker=marker,
                            s=90, zorder=10, edgecolors="white", linewidths=0.5)
            if abs_name not in legend_abs:
                legend_abs[abs_name] = plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=color, markersize=9, label=abs_name)
            if size_label not in legend_sz:
                legend_sz[size_label] = plt.Line2D(
                    [0], [0], marker=marker, color="grey",
                    markersize=9, linestyle="None",
                    label=size_label)

        # ── Labels ────────────────────────────────────────────────────────────
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title("E3 3D Stencil — Roofline Analysis\n"
                     f"RTX 5060 Laptop (peak BW={PEAK_BW_GBS} GB/s, "
                     f"peak FP64={PEAK_FP64_GFLOPS} GFLOP/s)")

        # Two-column legend: abstractions + sizes
        handles = list(legend_abs.values()) + list(legend_sz.values())
        ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=8,
                  ncol=2)

        # Annotate peak lines
        ax.annotate(f"Peak BW slope\n({PEAK_BW_GBS} GB/s)",
                    xy=(0.05, PEAK_BW_GBS * 0.05),
                    xytext=(0.08, PEAK_BW_GBS * 0.12),
                    fontsize=7, color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))
        ax.annotate(f"FP64 ceiling\n({PEAK_FP64_GFLOPS} GFLOP/s)",
                    xy=(10, PEAK_FP64_GFLOPS),
                    xytext=(5, PEAK_FP64_GFLOPS * 0.6),
                    fontsize=7, color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))

        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig12_e3_roofline.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_e3_roofline] Saved {out}")


if __name__ == "__main__":
    main()
