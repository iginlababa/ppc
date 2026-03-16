#!/usr/bin/env python3
"""
E5 SpTRSV — roofline plot.

fig22_e5_roofline.png — roofline model with all (abstraction, matrix_type, size) overlaid.

Hardware: NVIDIA RTX 5060 Laptop
  - Peak FP64: 261 GFLOP/s (Blackwell, FP64 = FP32/64 ≈ 16.7e3/64)
  - Peak BW:   272 GB/s (GDDR7 128-bit; measured from E1 STREAM ~270 GB/s)
  - Roofline ridgeline: AI_ridge = 261/272 ≈ 0.96 FLOP/byte

SpTRSV arithmetic intensity (AI):
  AI ≈ 2*nnz / (nnz*12 + n_rows*28 + 4) FLOP/byte ≈ 0.13 FLOP/byte
  (same ballpark as SpMV — low AI, well below ridgeline)

IMPORTANT NOTE ON INTERPRETATION:
  SpTRSV is NOT bandwidth-bound despite its low AI. The binding constraint is
  the level-set depth (n_levels synchronisation barriers), which imposes a
  serial latency floor that no amount of memory bandwidth can overcome.
  The roofline model assumes fully parallel execution and does NOT capture
  the level-set latency floor. The roofline ceiling is NOT the binding
  constraint for SpTRSV.

  The figure is included for consistency with E2–E4 and to make the contrast
  visible: all points fall well below the bandwidth-bound roofline line, but
  the gap between abstractions is NOT explained by bandwidth — it is explained
  by the per-level dispatch overhead (fig21).

  See the plot subtitle for this explanation.

Output: figures/e5/fig22_e5_roofline.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e5")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e5_sptrsv_summary.csv")

# ── Hardware parameters (RTX 5060 Laptop) ─────────────────────────────────────
PEAK_GFLOPS = 261.0    # FP64 GFLOP/s
PEAK_GBS    = 272.0    # GB/s (from E1 STREAM)
AI_RIDGE    = PEAK_GFLOPS / PEAK_GBS  # ≈ 0.96 FLOP/byte

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "julia":  "#d62728",
}

MATRIX_MARKERS = {
    "lower_triangular_laplacian": "o",
    "lower_triangular_random":    "^",
}

SIZE_SIZES = {"small": 50, "medium": 100, "large": 180}

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "julia"]


def compute_ai(n_rows: int, nnz: int) -> float:
    """Approximate SpTRSV arithmetic intensity (same model as SpMV + b reads)."""
    bytes_ = (nnz * 8.0          # values FP64
              + nnz * 4.0        # col_idx int32
              + (n_rows + 1) * 4.0  # row_ptr int32
              + n_rows * 8.0     # x reads (~nnz, but approximated as n_rows avg)
              + n_rows * 8.0     # b reads
              + n_rows * 8.0)    # x writes
    return (2.0 * nnz) / bytes_


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e5_roofline] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e5_roofline] Run process_e5.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e5_roofline] Loaded {len(df)} rows")

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Roofline ceiling ──────────────────────────────────────────────────────
    ai_range = np.logspace(-2.5, 1.5, 300)
    roofline  = np.minimum(PEAK_GFLOPS, ai_range * PEAK_GBS)
    ax.loglog(ai_range, roofline, "k-", linewidth=1.8, label="Roofline ceiling", zorder=2)

    # Ridgeline annotation
    ax.axvline(AI_RIDGE, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(AI_RIDGE * 1.05, PEAK_GFLOPS * 0.92,
            f"ridge\n({AI_RIDGE:.2f} FLOP/byte)",
            fontsize=7, color="gray", va="top")

    # Peak annotations
    ax.axhline(PEAK_GFLOPS, color="dimgray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(ai_range[-1] * 0.7, PEAK_GFLOPS * 1.02,
            f"{PEAK_GFLOPS:.0f} GFLOP/s (FP64 peak)",
            fontsize=7, color="dimgray", ha="right")

    # ── Data points ───────────────────────────────────────────────────────────
    plotted_labels = set()
    for _, row in df.iterrows():
        a     = row["abstraction"]
        mtype = row["matrix_type"]
        sz    = row["problem_size"]
        if a not in ABSTRACTION_ORDER:
            continue
        gf = float(row["median_gflops"])
        if np.isnan(gf) or gf <= 0:
            continue
        ai = compute_ai(int(row["n_rows"]), int(row["nnz"]))

        marker = MATRIX_MARKERS.get(mtype, "s")
        msize  = SIZE_SIZES.get(sz, 80)
        label  = f"{a} ({mtype.split('_')[2]})"  # e.g. "kokkos (laplacian)"
        do_label = label not in plotted_labels

        ax.scatter(ai, gf, c=COLORS[a], marker=marker, s=msize,
                   alpha=0.85, edgecolors="white", linewidths=0.5,
                   label=label if do_label else None, zorder=3)
        plotted_labels.add(label)

        ax.annotate(sz[0].upper(),  # S/M/L
                    (ai, gf), xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=COLORS[a])

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=10)

    subtitle = (
        "⚠ SpTRSV is latency-bound, NOT bandwidth-bound.\n"
        "Binding constraint = n_levels synchronisation barriers (serial dependency depth),\n"
        "not memory bandwidth. Roofline ceiling shown for consistency with E2–E4 only.\n"
        "Use fig21 (efficiency vs parallelism_ratio) for the correct diagnostic."
    )
    ax.set_title(
        f"E5 SpTRSV — Roofline (RTX 5060 Laptop, FP64)\n{subtitle}",
        fontsize=8, fontweight="bold", loc="left"
    )

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-2, PEAK_GFLOPS * 2)
    ax.grid(which="both", linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper left",
              framealpha=0.9, title="abstraction (matrix shape)", title_fontsize=7)

    out = os.path.join(FIG_DIR, "fig22_e5_roofline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_e5_roofline] Saved: {out}")


if __name__ == "__main__":
    main()
