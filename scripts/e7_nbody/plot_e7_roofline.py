#!/usr/bin/env python3
"""
plot_e7_roofline.py — E7 N-Body roofline figure.

fig15_e7_roofline.png — roofline model: GFLOP/s vs arithmetic intensity (FLOP/byte)
  Plots measured (AI, GFLOP/s) for each (abstraction, problem_size) configuration.
  Overlays the roofline ridge for nvidia_rtx5060_laptop.
  AI computed from CSR neighbor statistics:
    FLOPs = n_nbrs_total × 20
    Bytes = n_nbrs_total × 16 + n_atoms × 16 + n_atoms × 12

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

# ── RTX 5060 Laptop roofline parameters ───────────────────────────────────────
PEAK_GFLOPS   = 10_000.0   # 10 TFLOP/s sustained FP32
PEAK_BW_GBS   = 272.0      # 272 GB/s GDDR7
RIDGE_AI      = PEAK_GFLOPS / PEAK_BW_GBS   # ≈ 36.8 FLOP/byte

COLORS = {
    "native_notile": "#2b4590",
    "native_tile":   "#e87722",
    "kokkos":        "#2ca02c",
    "raja":          "#9467bd",
    "julia":         "#d62728",
}
MARKERS = {
    "small":  "o",
    "medium": "s",
    "large":  "^",
}
SIZE_LABELS = {"small": "S(4K)", "medium": "M(32K)", "large": "L(256K)"}


def fig15_roofline(df: pd.DataFrame):
    ai_range = np.logspace(-2, 3, 500)
    roof = np.minimum(ai_range * PEAK_BW_GBS, PEAK_GFLOPS)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(ai_range, roof, "k-", linewidth=1.8, label="Roofline (RTX 5060L)")
    ax.axvline(RIDGE_AI, color="gray", linewidth=0.8, linestyle="--",
               label=f"Ridge = {RIDGE_AI:.1f} FLOP/byte")

    ax.text(RIDGE_AI * 0.3, PEAK_GFLOPS * 0.08,
            "memory-bound", fontsize=8, color="gray", ha="center")
    ax.text(RIDGE_AI * 3.5, PEAK_GFLOPS * 0.5,
            "compute-bound", fontsize=8, color="gray", ha="center")

    legend_handles = []

    for _, row in df.iterrows():
        abs_ = str(row["abstraction"])
        sz   = str(row["problem_size"])
        if abs_ not in COLORS or sz not in MARKERS:
            continue

        ai     = float(row["ai_flop_byte"]) if not np.isnan(row["ai_flop_byte"]) else np.nan
        gflops = float(row["median_gflops"])
        if np.isnan(ai) or ai <= 0:
            continue

        color  = COLORS[abs_]
        marker = MARKERS[sz]
        ax.scatter(ai, gflops, c=color, marker=marker,
                   s=80, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(
            SIZE_LABELS[sz],
            (ai, gflops), xytext=(5, 4), textcoords="offset points",
            fontsize=7, color=color
        )

    # Legend
    for abs_, color in COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=abs_))
    for sz, marker in MARKERS.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="gray",
                       linestyle="None", markersize=7, label=SIZE_LABELS[sz])
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=1.8, label="Roofline")
    )
    legend_handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Ridge ({RIDGE_AI:.1f})")
    )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("GFLOP/s (median)", fontsize=11)
    ax.set_title(
        "E7 N-Body — Roofline (RTX 5060 Laptop)\n"
        "CSR AI = n_nbrs×20 / (n_nbrs×16 + N×16 + N×12) ≈ 0.975 FLOP/byte\n"
        "All kernels are memory-bound (AI << ridge ≈ 36.8 FLOP/byte)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(handles=legend_handles, fontsize=7.5, ncol=3,
              loc="lower right", framealpha=0.9)
    ax.grid(which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(0.05, 200)
    ax.set_ylim(1, PEAK_GFLOPS * 2)

    out = os.path.join(FIG_DIR, "fig15_e7_roofline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e7_roofline] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e7_roofline] Run process_e7.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e7_roofline] Loaded {len(df)} rows")

    print("[plot_e7_roofline] Generating fig15 (roofline) ...")
    fig15_roofline(df)

    print(f"[plot_e7_roofline] Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
