#!/usr/bin/env python3
"""
E5 SpTRSV — efficiency and level structure figures.

fig19_e5_efficiency_by_matrix.png  — grouped bars: efficiency by abstraction × matrix_type × size
fig20_e5_level_structure.png       — level width distribution (horizontal bar chart per matrix_type × size)
fig21_e5_efficiency_vs_parallelism.png — efficiency vs parallelism_ratio scatter

Output: figures/e5/
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
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e5")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e5_sptrsv_summary.csv")
DATA_RAW    = os.path.join(REPO_ROOT, "data", "raw")

PROBLEM_SIZES = {"small": 256, "medium": 2048, "large": 8192}
SIZE_ORDER    = ["small", "medium", "large"]
SIZE_LABELS   = ["small\n(N=256)", "medium\n(N=2K)", "large\n(N=8K)"]

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "julia"]
MATRIX_TYPES      = ["lower_triangular_laplacian", "lower_triangular_random"]
MATRIX_LABELS     = {
    "lower_triangular_laplacian": "Laplacian lower-triangular\n(regular levels)",
    "lower_triangular_random":    "Random lower-triangular\n(irregular levels)",
}

COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "julia":  "#d62728",
}

PLATFORM = "nvidia_rtx5060"


# ── Figure 19: efficiency grouped bar chart ───────────────────────────────────
def fig19_efficiency_bars(df: pd.DataFrame):
    """
    Grouped bar chart: efficiency vs native baseline per abstraction.
    Layout: 2 rows (matrix types) × 3 columns (problem sizes).
    Each subplot: one bar per non-native abstraction (kokkos, raja, julia).
    Horizontal dashed line at efficiency=1.0 (native) and 0.85 (deep profiling threshold).
    """
    abs_plot = [a for a in ABSTRACTION_ORDER if a != "native"]
    n_abs    = len(abs_plot)
    x        = np.arange(n_abs)
    width    = 0.55

    fig, axes = plt.subplots(
        len(MATRIX_TYPES), len(SIZE_ORDER),
        figsize=(13, 7), sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.12}
    )

    for i, mtype in enumerate(MATRIX_TYPES):
        for j, sz in enumerate(SIZE_ORDER):
            ax = axes[i, j]
            sub = df[(df["matrix_type"] == mtype) & (df["problem_size"] == sz)]
            if sub.empty:
                ax.set_visible(False)
                continue

            bars_eff = []
            bars_err = []
            for a in abs_plot:
                row = sub[sub["abstraction"] == a]
                if row.empty:
                    bars_eff.append(0.0)
                    bars_err.append(0.0)
                else:
                    bars_eff.append(float(row["efficiency"].iloc[0]))
                    # IQR / native_median as error bar proxy
                    native_row = df[(df["abstraction"] == "native") &
                                    (df["matrix_type"] == mtype) &
                                    (df["problem_size"] == sz)]
                    nm = float(native_row["median_gflops"].iloc[0]) if not native_row.empty else 1.0
                    err = float(row["iqr_gflops"].iloc[0]) / nm if nm > 0 else 0.0
                    bars_err.append(err)

            bar_colors = [COLORS[a] for a in abs_plot]
            rects = ax.bar(x, bars_eff, width, color=bar_colors, alpha=0.85,
                           yerr=bars_err, capsize=3,
                           error_kw={"linewidth": 0.8, "color": "dimgray"})

            # Reference lines
            ax.axhline(1.0,  color="black",      linewidth=0.9, linestyle="--",
                       label="native (1.0)", zorder=1)
            ax.axhline(0.85, color="#888888",     linewidth=0.7, linestyle=":",
                       label="deep-profile threshold (0.85)", zorder=1)

            # Value labels on bars
            for rect, eff in zip(rects, bars_eff):
                if not np.isnan(eff) and eff > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2.0, eff + 0.02,
                            f"{eff:.2f}", ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(abs_plot, fontsize=8)
            ax.set_ylim(0, max(1.4, max(bars_eff) + 0.2))
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            if i == 0:
                ax.set_title(SIZE_LABELS[j], fontsize=9, fontweight="bold")
            if j == 0:
                label = MATRIX_LABELS[mtype].replace("\n", " ")
                ax.set_ylabel(f"{label}\nEfficiency", fontsize=7.5)
            if i == len(MATRIX_TYPES) - 1:
                ax.set_xlabel("Abstraction", fontsize=8)

    # Legend
    handles = [mpatches.Patch(color=COLORS[a], label=a) for a in abs_plot]
    handles += [
        plt.Line2D([0], [0], color="black",  linestyle="--", label="native (1.0)"),
        plt.Line2D([0], [0], color="#888888", linestyle=":",  label="deep-profile (0.85)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "E5 SpTRSV — Abstraction Efficiency vs Native CUDA\n"
        "(level-set forward substitution; metric: GFLOP/s = 2·nnz/t; latency-bound)",
        fontsize=10, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig19_e5_efficiency_by_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 20: level structure visualisation ──────────────────────────────────
def fig20_level_structure(df: pd.DataFrame):
    """
    Level width distribution per (matrix_type, problem_size).
    For each configuration: horizontal bar chart where x=level index, y=rows in level.
    Uses the raw CSV to extract level_ptr / level-width counts.
    Falls back to n_levels, max_level_width, min_level_width from summary if no raw data.

    This figure shows WHY efficiency differs: shallow levels with many rows (high parallelism)
    vs deep levels with few rows (serialisation bottleneck).
    """
    # Build level width distributions from summary metadata only
    # (full level-width arrays would require re-running the generator; use summary stats)
    fig, axes = plt.subplots(
        len(SIZE_ORDER), len(MATRIX_TYPES),
        figsize=(12, 9), sharey=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.35}
    )

    for j, mtype in enumerate(MATRIX_TYPES):
        for i, sz in enumerate(SIZE_ORDER):
            ax = axes[i, j]
            # Use native row for level-set metadata (same for all abstractions)
            native_sub = df[
                (df["abstraction"] == "native") &
                (df["matrix_type"] == mtype) &
                (df["problem_size"] == sz)
            ]
            if native_sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_visible(False)
                continue

            row = native_sub.iloc[0]
            n_rows   = int(row["n_rows"])
            n_levels = int(row["n_levels"])
            max_lw   = int(row["max_level_width"])
            min_lw   = int(row["min_level_width"])
            par_ratio = float(row["parallelism_ratio"])

            # We only have aggregate metadata, not per-level widths.
            # Plot a summary bar showing: [min, mean (n_rows/n_levels), max] level width.
            mean_lw = n_rows / n_levels if n_levels > 0 else 0
            stats_labels = ["min level width", "mean level width", "max level width"]
            stats_vals   = [min_lw, mean_lw, max_lw]
            bar_colors   = ["#e87722", "#2ca02c", "#2b4590"]
            ypos         = np.arange(len(stats_labels))

            ax.barh(ypos, stats_vals, color=bar_colors, alpha=0.80, height=0.5)
            ax.set_yticks(ypos)
            ax.set_yticklabels(stats_labels, fontsize=7.5)
            ax.axvline(n_rows / max(n_levels, 1), color="black", linewidth=0.8,
                       linestyle="--", label="mean=N/n_levels")
            ax.set_xlabel("Rows in level", fontsize=8)
            ax.set_xscale("symlog", linthresh=1)
            ax.grid(axis="x", linestyle=":", alpha=0.5)

            title = (f"{MATRIX_LABELS[mtype].split(chr(10))[0]}\n"
                     f"{sz} (N={n_rows}, n_levels={n_levels}, "
                     f"par_ratio={par_ratio:.2f})")
            ax.set_title(title, fontsize=7.5, fontweight="bold")

            for val, y in zip(stats_vals, ypos):
                ax.text(val + 0.5, y, f"{val:.0f}", va="center", fontsize=7)

    fig.suptitle(
        "E5 SpTRSV — Level Width Distribution\n"
        "(level width = number of parallelisable rows per synchronisation barrier;\n"
        " low max_level_width → serial bottleneck regardless of abstraction)",
        fontsize=9, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig20_e5_level_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 21: efficiency vs parallelism_ratio scatter ───────────────────────
def fig21_efficiency_vs_parallelism(df: pd.DataFrame):
    """
    Scatter plot: efficiency vs parallelism_ratio (max_level_width / n_rows).
    One point per (abstraction, matrix_type, problem_size) for non-native abstractions.
    Colour = abstraction. Marker shape = matrix type.
    Tests: does low parallelism_ratio predict low efficiency?
    """
    abs_plot = [a for a in ABSTRACTION_ORDER if a != "native"]
    sub = df[df["abstraction"].isin(abs_plot)].copy()

    if sub.empty:
        print("  fig21: no data, skipping", file=sys.stderr)
        return

    markers = {
        "lower_triangular_laplacian": "o",
        "lower_triangular_random":    "^",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mtype, marker in markers.items():
        for a in abs_plot:
            pts = sub[(sub["abstraction"] == a) & (sub["matrix_type"] == mtype)]
            if pts.empty:
                continue
            ax.scatter(
                pts["parallelism_ratio"],
                pts["efficiency"],
                c=COLORS[a], marker=marker, s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5,
                label=f"{a} / {MATRIX_LABELS[mtype].split(chr(10))[0]}"
            )
            # Annotate with size label
            for _, row in pts.iterrows():
                ax.annotate(
                    row["problem_size"][0].upper(),  # S/M/L
                    (row["parallelism_ratio"], row["efficiency"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=COLORS[a]
                )

    ax.axhline(1.0,  color="black",  linewidth=0.9, linestyle="--", label="native (1.0)")
    ax.axhline(0.85, color="#888888", linewidth=0.7, linestyle=":",
               label="deep-profile threshold (0.85)")

    ax.set_xlabel("Parallelism ratio  (max_level_width / n_rows)", fontsize=10)
    ax.set_ylabel("Efficiency vs native CUDA", fontsize=10)
    ax.set_title(
        "E5 SpTRSV — Efficiency vs Available Parallelism\n"
        "(S=small, M=medium, L=large; ○=Laplacian, △=Random)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=0)

    out = os.path.join(FIG_DIR, "fig21_e5_efficiency_vs_parallelism.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e5] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e5] Run process_e5.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e5] Loaded {len(df)} rows from {SUMMARY_CSV}")

    print("[plot_e5] Generating fig19 (efficiency bar charts) ...")
    fig19_efficiency_bars(df)

    print("[plot_e5] Generating fig20 (level structure visualisation) ...")
    fig20_level_structure(df)

    print("[plot_e5] Generating fig21 (efficiency vs parallelism_ratio scatter) ...")
    fig21_efficiency_vs_parallelism(df)

    print(f"[plot_e5] All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
