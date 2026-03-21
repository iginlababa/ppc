#!/usr/bin/env python3
"""
plot_e7_roofline.py — E7 N-Body two-panel roofline figure.

fig15_e7_roofline.png — two-panel: left = NVIDIA RTX 5060, right = AMD MI300X
  Plots measured (AI, GFLOP/s) for each (abstraction, problem_size) per platform.
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

# ── Per-platform roofline parameters ───────────────────────────────────────────
PLATFORMS = {
    "nvidia_rtx5060": {
        "label":      "NVIDIA RTX 5060",
        "peak_gflops": 10_000.0,   # 10 TFLOP/s sustained FP32
        "peak_bw":     272.0,      # 272 GB/s GDDR7
        "native_abs":  "native_notile",
    },
    "amd_mi300x": {
        "label":      "AMD MI300X",
        "peak_gflops": 181_000.0,  # ~181 TFLOP/s FP32
        "peak_bw":     4_010.0,    # 4010 GB/s HBM3
        "native_abs":  "hip",
    },
}

COLORS = {
    "native_notile": "#2b4590",
    "native_tile":   "#e87722",
    "hip":           "#2b4590",   # same blue — native equivalent for AMD
    "kokkos":        "#2ca02c",
    "raja":          "#9467bd",
    "sycl":          "#ff7f0e",
    "julia":         "#d62728",
}
MARKERS = {
    "small":  "o",
    "medium": "s",
    "large":  "^",
}
SIZE_LABELS = {"small": "S(4K)", "medium": "M(32K)", "large": "L(256K)"}


def _plot_panel(ax, df_plat: pd.DataFrame, cfg: dict):
    peak_gflops = cfg["peak_gflops"]
    peak_bw     = cfg["peak_bw"]
    ridge_ai    = peak_gflops / peak_bw

    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(ai_range * peak_bw, peak_gflops)

    ax.loglog(ai_range, roof, "k-", linewidth=1.8)
    ax.axvline(ridge_ai, color="gray", linewidth=0.8, linestyle="--")
    ax.text(ridge_ai * 0.3, peak_gflops * 0.05,
            "memory-bound", fontsize=7, color="gray", ha="center")
    ax.text(ridge_ai * 3.5, peak_gflops * 0.4,
            "compute-bound", fontsize=7, color="gray", ha="center")

    for _, row in df_plat.iterrows():
        abs_ = str(row["abstraction"])
        sz   = str(row["problem_size"])
        if abs_ not in COLORS or sz not in MARKERS:
            continue
        ai     = float(row["ai_flop_byte"]) if not np.isnan(row["ai_flop_byte"]) else np.nan
        gflops = float(row["median_gflops"])
        if np.isnan(ai) or ai <= 0 or gflops <= 0:
            continue
        ax.scatter(ai, gflops, c=COLORS[abs_], marker=MARKERS[sz],
                   s=80, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(SIZE_LABELS[sz], (ai, gflops),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=6.5, color=COLORS[abs_])

    ridge_label = f"Ridge={ridge_ai:.0f}" if ridge_ai >= 10 else f"Ridge={ridge_ai:.1f}"
    ax.set_title(
        f"{cfg['label']}\n"
        f"Peak {peak_gflops/1000:.0f} TFLOP/s  |  BW {peak_bw:.0f} GB/s  |  {ridge_label} FLOP/byte",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=9)
    ax.grid(which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(0.05, 200)
    ax.set_ylim(1, peak_gflops * 2)


def fig15_roofline(df: pd.DataFrame):
    # Determine which platforms are present in data
    present = [p for p in PLATFORMS if p in df["platform"].unique()]
    if not present:
        print("[plot_e7_roofline] WARNING: no known platforms found in summary CSV")
        return

    ncols = len(present)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6), squeeze=False)

    for col, plat in enumerate(present):
        ax = axes[0][col]
        df_p = df[df["platform"] == plat]
        _plot_panel(ax, df_p, PLATFORMS[plat])
        if col == 0:
            ax.set_ylabel("GFLOP/s (median)", fontsize=9)

    # Shared legend on right panel
    legend_handles = []
    # Collect abstraction colors present in data
    for abs_, color in COLORS.items():
        if abs_ in df["abstraction"].unique():
            lbl = "native (hip)" if abs_ == "hip" else abs_
            legend_handles.append(mpatches.Patch(color=color, label=lbl))
    for sz, marker in MARKERS.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="gray",
                       linestyle="None", markersize=7, label=SIZE_LABELS[sz])
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=1.8, label="Roofline")
    )
    axes[0][-1].legend(handles=legend_handles, fontsize=7.5, ncol=1,
                       loc="lower right", framealpha=0.9)

    fig.suptitle(
        "E7 N-Body — Roofline  (CSR AI ≈ 0.975 FLOP/byte, memory-bound on all platforms)",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()

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

    print("[plot_e7_roofline] Generating fig15 (two-panel roofline) ...")
    fig15_roofline(df)

    print(f"[plot_e7_roofline] Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
