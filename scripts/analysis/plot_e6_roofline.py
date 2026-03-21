#!/usr/bin/env python3
"""
E6 BFS — roofline plot (two-panel: NVIDIA RTX 5060 | AMD MI300X).

fig26_e6_roofline.png — side-by-side roofline model for both platforms.

IMPORTANT NOTE ON INTERPRETATION:
  BFS is simultaneously latency-bound (n_levels sync barriers), memory-bound
  (irregular scatter → non-coalesced access), and atomic-contention-bound
  (atomicCAS hot vertices). The roofline ceiling is NOT the binding constraint.
  Points will fall well below the bandwidth-bound line.
  The figure is included for consistency with E2–E5 and to confirm BFS sits
  in the same low-AI regime as SpMV and SpTRSV.

Hardware:
  RTX 5060: Peak FP64 = 261 GFLOP/s, Peak BW = 272 GB/s (E1 STREAM)
  MI300X:   Peak FP64 = 1307 GFLOP/s, Peak BW = 4010 GB/s (HBM3)

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

# ── Hardware parameters ────────────────────────────────────────────────────────
PLATFORMS = {
    "nvidia_rtx5060": {
        "label":      "RTX 5060",
        "peak_gflops": 261.0,
        "peak_gbs":    272.0,
    },
    "amd_mi300x": {
        "label":      "AMD MI300X",
        "peak_gflops": 1307.0,
        "peak_gbs":    4010.0,
    },
}

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "sycl":   "#9467bd",
    "julia":  "#d62728",
}

GRAPH_MARKERS = {
    "erdos_renyi": "o",
    "2d_grid":     "^",
}

SIZE_SIZES = {"small": 50, "medium": 100, "large": 180}
ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia"]


def compute_bfs_ai(n_vertices: int, n_edges: int) -> float:
    """
    Approximate BFS arithmetic intensity.
    Per undirected edge traversal:
      Reads: col_idx[j] (int32, 4B) + d_distances[v] (int32, 4B)
      Writes: d_distances[v] (int32, 4B, conditional) + d_flags[v] (int32, 4B)
      Plus row_ptr: (n_vertices+1)*4 bytes amortised
    AI = n_edges / (n_edges*16 + (n_vertices+1)*4)
    """
    bytes_ = (n_edges * 4.0        # col_idx reads
              + n_edges * 4.0      # d_distances reads (atomicCAS)
              + n_edges * 4.0      # d_distances writes (overestimate all)
              + n_edges * 4.0      # d_flags writes
              + (n_vertices + 1) * 4.0)   # row_ptr
    return float(n_edges) / bytes_


def plot_panel(ax, df_plat, hw, platform_label):
    peak_gflops = hw["peak_gflops"]
    peak_gbs    = hw["peak_gbs"]
    ai_ridge    = peak_gflops / peak_gbs

    ai_range = np.logspace(-2.5, 1.5, 300)
    roofline  = np.minimum(peak_gflops, ai_range * peak_gbs)
    ax.loglog(ai_range, roofline, "k-", linewidth=1.8,
              label="Roofline ceiling", zorder=2)

    ax.axvline(ai_ridge, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(ai_ridge * 1.05, peak_gflops * 0.88,
            f"ridge\n({ai_ridge:.2f})",
            fontsize=6, color="gray", va="top")

    ax.axhline(peak_gflops, color="dimgray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(ai_range[-1] * 0.6, peak_gflops * 1.02,
            f"{peak_gflops:.0f} GFLOP/s",
            fontsize=6, color="dimgray", ha="right")

    plotted_labels = set()
    for _, row in df_plat.iterrows():
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

        ax.scatter(ai, gf, c=COLORS.get(a, "#888888"), marker=marker, s=msize,
                   alpha=0.85, edgecolors="white", linewidths=0.5,
                   label=label if do_label else None, zorder=3)
        plotted_labels.add(label)
        ax.annotate(sz[0].upper(), (ai, gf), xytext=(4, 2),
                    textcoords="offset points", fontsize=6,
                    color=COLORS.get(a, "#888888"))

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=9)
    ax.set_ylabel("Performance (GTEPS as GFLOP/s equiv.)", fontsize=9)
    ax.set_title(platform_label, fontsize=9, fontweight="bold")
    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, peak_gflops * 2)
    ax.grid(which="both", linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, ncol=2, loc="upper left",
              framealpha=0.9, title="abstraction (graph)", title_fontsize=6)


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e6_roofline] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e6_roofline] Run process_e6.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e6_roofline] Loaded {len(df)} rows")

    subtitle = (
        "⚠ BFS is latency-bound (n_levels barriers) + BW-bound (irregular scatter) + atomic-contention-bound.\n"
        "Roofline ceiling is NOT the binding constraint. P006 absent (no tiling). "
        "Use fig23/fig25 for diagnostics."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"E6 BFS — Roofline (FP64 equivalent)\n{subtitle}",
                 fontsize=8, fontweight="bold", x=0.5, ha="center")

    for ax, (platform, hw) in zip(axes, PLATFORMS.items()):
        df_plat = df[df["platform"] == platform] if "platform" in df.columns else df
        if df_plat.empty:
            ax.set_title(f"{hw['label']} (no data)", fontsize=9)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            continue
        plot_panel(ax, df_plat, hw, hw["label"])

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIG_DIR, "fig26_e6_roofline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_e6_roofline] Saved: {out}")


if __name__ == "__main__":
    main()
