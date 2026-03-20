#!/usr/bin/env python3
"""
E4 SpMV — roofline plot.

fig18_e4_roofline.png — roofline model with all (abstraction, matrix_type, size) overlaid.
  - Hardware: NVIDIA RTX 5060
    - Peak FP64 TFLOP/s: 0.261 TFLOP/s (Blackwell, FP64 = FP32/64 ≈ 261 GFLOP/s)
    - Peak BW: 272 GB/s (GDDR7, 128-bit bus; measured from E1 STREAM native ~270 GB/s)
  - Each (abstraction, matrix_type, size) plotted as a point: (AI, GFLOP/s).
  - AI per point = 2*nnz / (nnz*12 + nrows*20 + 4) FLOP/byte (computed from actual nnz/nrows).
    All matrix types have different AI due to different nnz structures:
      laplacian_2d:  ~4.8 nnz/row → AI ≈ 0.131 FLOP/byte
      random_sparse: exactly 5 nnz/row → AI ≈ 0.133 FLOP/byte
      power_law:     variable nnz (heavy tail) → AI similar range, higher variance
  - Roofline ridgeline at AI = peak_gflops / peak_gbs ≈ 261/272 ≈ 0.96 FLOP/byte.
    Since all SpMV AI values ≈ 0.13 << 0.96, SpMV is firmly memory-bound.

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

# ── Hardware parameters (RTX 5060) ─────────────────────────────────────
# Peak BW from E1 STREAM native: ~270 GB/s (locked-clock measurement).
# Peak FP64: RTX 5060 FP64 rate = FP32/64 ≈ 16.7e3/64 ≈ 261 GFLOP/s.
PEAK_BW_GBS      = 270.0   # GB/s — measured upper bound from E1
PEAK_FP64_GFLOPS = 261.0   # GFLOP/s — hardware ceiling

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]
MATRIX_TYPES      = ["laplacian_2d", "random_sparse", "power_law"]

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
    "figure.dpi":      150,
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


def main():
    df = load_summary()

    # Compute AI per row from actual n_rows and nnz
    if "n_rows" in df.columns and "nnz" in df.columns:
        df["ai"] = df.apply(
            lambda r: compute_ai(int(r["n_rows"]), int(r["nnz"])), axis=1)
    else:
        # Fallback: theoretical AI for 5-nnz/row CSR FP64
        # bytes = nnz*12 + nrows*20 + 4 ≈ (5*12 + 20)*N = 80N → AI = 10N/80N = 0.125
        df["ai"] = 0.125
        print("  WARNING: n_rows/nnz columns not found; using approximate AI=0.125",
              file=sys.stderr)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))

        # ── Roofline boundaries ────────────────────────────────────────────────
        ai_range = np.logspace(-2, 2, 400)
        mem_bound  = PEAK_BW_GBS * ai_range
        comp_bound = np.full_like(ai_range, PEAK_FP64_GFLOPS)
        roofline   = np.minimum(mem_bound, comp_bound)

        ax.plot(ai_range, roofline, "k-", linewidth=2.0, label="Roofline (HW bound)", zorder=5)
        ax.axvline(PEAK_FP64_GFLOPS / PEAK_BW_GBS, color="grey",
                   linestyle=":", linewidth=1.0, alpha=0.6)

        # Shade the SpMV AI region
        ai_vals = df["ai"].dropna()
        if len(ai_vals) > 0:
            ai_min, ai_max = float(ai_vals.min()), float(ai_vals.max())
            ax.axvspan(ai_min * 0.9, ai_max * 1.1, alpha=0.07, color="steelblue",
                       label=f"SpMV AI range [{ai_min:.3f}–{ai_max:.3f}]")

        # ── Plot each (abstraction, matrix_type, size) point ──────────────────
        legend_abs  = {}
        legend_mtyp = {}
        for _, row in df.iterrows():
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
                short = mtype.replace("_", " ")
                legend_mtyp[mtype] = plt.Line2D(
                    [0], [0], marker=marker, color="grey",
                    markersize=9, linestyle="None", label=short)

        # Size legend entries (by marker size)
        size_handles = [
            plt.Line2D([0], [0], marker="o", color="grey",
                       markersize=int(np.sqrt(s) * 0.6), linestyle="None",
                       label=sz)
            for sz, s in SIZE_SIZES.items()
        ]

        # ── Labels ────────────────────────────────────────────────────────────
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title("E4 SpMV — Roofline Analysis\n"
                     f"RTX 5060 (peak BW={PEAK_BW_GBS} GB/s, "
                     f"peak FP64={PEAK_FP64_GFLOPS} GFLOP/s)")

        handles = list(legend_abs.values()) + list(legend_mtyp.values()) + size_handles
        ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

        ax.annotate(f"Peak BW slope\n({PEAK_BW_GBS} GB/s)",
                    xy=(0.05, PEAK_BW_GBS * 0.05),
                    xytext=(0.08, PEAK_BW_GBS * 0.12),
                    fontsize=7, color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))
        ax.annotate(f"FP64 ceiling\n({PEAK_FP64_GFLOPS} GFLOP/s)",
                    xy=(10, PEAK_FP64_GFLOPS),
                    xytext=(5, PEAK_FP64_GFLOPS * 0.55),
                    fontsize=7, color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))

        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig18_e4_roofline.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_e4_roofline] Saved {out}")


if __name__ == "__main__":
    main()
