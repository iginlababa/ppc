#!/usr/bin/env python3
"""
E6 BFS — roofline plot.

fig26_e6_roofline.png — roofline model with all (abstraction, graph_type, size) overlaid.

Hardware: NVIDIA RTX 5060
  - Peak FP64: 261 GFLOP/s  (Blackwell)
  - Peak BW:   272 GB/s     (GDDR7 128-bit; measured E1 STREAM ~270 GB/s)
  - Roofline ridgeline: AI_ridge = 261 / 272 ≈ 0.96 FLOP/byte

BFS arithmetic intensity:
  AI ≈ n_edges * (4 + 4) bytes_read / n_edges ops ≈ 8 bytes / traversal
  Treating 1 traversal ≈ 1 FLOP: AI ≈ 1 / 8 = 0.125 FLOP/byte
  (read src distance + read col_idx entry ≈ 2 × int32 = 8 bytes per edge)

IMPORTANT NOTE ON INTERPRETATION:
  BFS is simultaneously latency-bound AND memory-bandwidth-bound, but
  NEITHER ceiling dominates in the simple roofline sense:
    (a) Latency: frontier serialisation (one phase per level = n_levels syncs)
        creates a serial floor — same mechanism as SpTRSV P008.
    (b) Bandwidth: irregular frontier scatter creates non-coalesced memory
        access patterns — adjacent frontier vertices are not adjacent in
        memory, so cache lines are wasted.
    (c) Atomic contention: atomicCAS on d_distances during scatter creates
        memory-access serialisation at hot vertices (high-degree for ER).
  The roofline ceiling is NOT the binding constraint for BFS.
  Points will fall well below the bandwidth-bound line.
  The figure is included for consistency with E2–E5 and to visually confirm
  BFS sits in the same low-AI regime as SpMV and SpTRSV.

Note on P006 (Tiling Policy Overhead):
  P006 is NOT expected in BFS.  BFS has no tiling — there is no
  multi-dimensional iteration space for Kokkos MDRangePolicy or RAJA
  multi-level loops.  Kokkos and RAJA both use flat RangePolicy/forall
  over the frontier.  P006 absence in E6 is informative: confirms that
  P006 is specific to tiling-heavy kernels (stencil, DGEMM).

Output: figures/e6/fig26_e6_roofline.png
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
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e6")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e6_bfs_summary.csv")

# ── Hardware parameters (RTX 5060) ─────────────────────────────────────
PEAK_GFLOPS = 261.0    # FP64 GFLOP/s
PEAK_GBS    = 272.0    # GB/s
AI_RIDGE    = PEAK_GFLOPS / PEAK_GBS  # ≈ 0.96

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "julia":  "#d62728",
}

GRAPH_MARKERS = {
    "erdos_renyi": "o",
    "2d_grid":     "^",
}

SIZE_SIZES = {"small": 50, "medium": 100, "large": 180}
ABSTRACTION_ORDER = ["native", "kokkos", "raja", "julia"]


def compute_bfs_ai(n_vertices: int, n_edges: int) -> float:
    """
    Approximate BFS arithmetic intensity.
    Per undirected edge traversal:
      Reads: col_idx[j] (int32, 4 bytes) + d_distances[v] (int32, 4 bytes)
      Write (on success): d_distances[v] (int32, 4 bytes) + d_flags[v] (int32, 4 bytes)
      Plus row_ptr reads: (n_vertices+1)*4 bytes (amortised across n_edges traversals)
    Conservative: ~12 bytes per edge traversal, 1 op per edge → AI ≈ 1/12
    Lower bound: 8 bytes (col_idx + dist read only) → AI ≈ 0.125
    We use AI = 2*n_edges / (n_edges*12 + n_vertices*4) matching SpMV pattern.
    """
    bytes_ = (n_edges * 4.0       # col_idx reads
              + n_edges * 4.0     # d_distances reads (atomicCAS)
              + n_edges * 4.0     # d_distances writes (conditional, overestimate all)
              + n_edges * 4.0     # d_flags writes
              + (n_vertices + 1) * 4.0)   # row_ptr
    return float(n_edges) / bytes_  # 1 op per edge traversal


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e6_roofline] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e6_roofline] Run process_e6.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e6_roofline] Loaded {len(df)} rows")

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Roofline ceiling ──────────────────────────────────────────────────────
    ai_range = np.logspace(-2.5, 1.5, 300)
    roofline  = np.minimum(PEAK_GFLOPS, ai_range * PEAK_GBS)
    ax.loglog(ai_range, roofline, "k-", linewidth=1.8,
              label="Roofline ceiling", zorder=2)

    ax.axvline(AI_RIDGE, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(AI_RIDGE * 1.05, PEAK_GFLOPS * 0.92,
            f"ridge\n({AI_RIDGE:.2f} FLOP/byte)",
            fontsize=7, color="gray", va="top")

    ax.axhline(PEAK_GFLOPS, color="dimgray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(ai_range[-1] * 0.7, PEAK_GFLOPS * 1.02,
            f"{PEAK_GFLOPS:.0f} GFLOP/s (FP64 peak)",
            fontsize=7, color="dimgray", ha="right")

    # ── Data points ───────────────────────────────────────────────────────────
    plotted_labels = set()
    for _, row in df.iterrows():
        a     = row["abstraction"]
        gtype = row["graph_type"]
        sz    = row["problem_size"]
        if a not in ABSTRACTION_ORDER:
            continue
        gf = float(row["median_gflops"])
        if np.isnan(gf) or gf <= 0:
            continue
        ai = compute_bfs_ai(int(row["n_vertices"]), int(row["n_edges"]))

        marker = GRAPH_MARKERS.get(gtype, "s")
        msize  = SIZE_SIZES.get(sz, 80)
        short_type = "ER" if gtype == "erdos_renyi" else "grid"
        label  = f"{a} ({short_type})"
        do_label = label not in plotted_labels

        ax.scatter(ai, gf, c=COLORS[a], marker=marker, s=msize,
                   alpha=0.85, edgecolors="white", linewidths=0.5,
                   label=label if do_label else None, zorder=3)
        plotted_labels.add(label)

        ax.annotate(sz[0].upper(),
                    (ai, gf), xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=COLORS[a])

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=10)
    ax.set_ylabel("Performance (GTEPS stored as GFLOP/s equivalent)", fontsize=10)

    subtitle = (
        "⚠ BFS is latency-bound (n_levels sync barriers) AND memory-bound\n"
        "(irregular scatter → non-coalesced access) AND atomic-contention-bound.\n"
        "Roofline ceiling is NOT the binding constraint.\n"
        "P006 (Tiling Policy Overhead) is ABSENT in BFS — no tiling, flat forall/RangePolicy.\n"
        "Use fig23 (efficiency) and fig25 (irregularity) for diagnostics."
    )
    ax.set_title(
        f"E6 BFS — Roofline (RTX 5060, FP64 equivalent)\n{subtitle}",
        fontsize=7.5, fontweight="bold", loc="left"
    )

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, PEAK_GFLOPS * 2)
    ax.grid(which="both", linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper left",
              framealpha=0.9, title="abstraction (graph)", title_fontsize=7)

    out = os.path.join(FIG_DIR, "fig26_e6_roofline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_e6_roofline] Saved: {out}")


if __name__ == "__main__":
    main()
