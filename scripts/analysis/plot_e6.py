#!/usr/bin/env python3
"""
E6 BFS — efficiency and frontier structure figures.

fig23_e6_efficiency_by_graph.png   — grouped bar: efficiency by abstraction × graph_type × size
fig24_e6_frontier_profile.png      — frontier width vs level index (line chart per config)
fig25_e6_efficiency_vs_irregularity.png — efficiency vs frontier_irregularity scatter

Output: figures/e6/
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
DATA_RAW  = os.path.join(REPO_ROOT, "data", "raw")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e6")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e6_bfs_summary.csv")
PLATFORM    = "nvidia_rtx5060"

SIZE_ORDER      = ["small", "medium", "large"]
SIZE_LABELS     = ["small\n(N=1K)", "medium\n(N=16K)", "large\n(N=64K)"]
GRAPH_TYPES     = ["erdos_renyi", "2d_grid"]
GRAPH_LABELS    = {
    "erdos_renyi": "Erdős–Rényi G(N, 10/N)\n(irregular frontier)",
    "2d_grid":     "2D Grid √N×√N\n(regular diamond frontier)",
}
ABSTRACTION_ORDER = ["native", "kokkos", "raja", "julia"]
COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "julia":  "#d62728",
}


# ── Figure 23: efficiency grouped bar chart ───────────────────────────────────
def fig23_efficiency_bars(df: pd.DataFrame):
    abs_plot = [a for a in ABSTRACTION_ORDER if a != "native"]
    n_abs    = len(abs_plot)
    x        = np.arange(n_abs)
    width    = 0.55

    fig, axes = plt.subplots(
        len(GRAPH_TYPES), len(SIZE_ORDER),
        figsize=(13, 7), sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.12}
    )

    for i, gtype in enumerate(GRAPH_TYPES):
        for j, sz in enumerate(SIZE_ORDER):
            ax   = axes[i, j]
            sub  = df[(df["graph_type"] == gtype) & (df["problem_size"] == sz)]
            if sub.empty:
                ax.set_visible(False)
                continue

            bars_eff = []
            bars_err = []
            for a in abs_plot:
                row = sub[sub["abstraction"] == a]
                if row.empty:
                    bars_eff.append(0.0); bars_err.append(0.0)
                else:
                    bars_eff.append(float(row["efficiency"].iloc[0]))
                    native_row = df[(df["abstraction"] == "native") &
                                    (df["graph_type"]  == gtype) &
                                    (df["problem_size"] == sz)]
                    nm  = float(native_row["median_gflops"].iloc[0]) \
                          if not native_row.empty else 1.0
                    err = float(row["iqr_gflops"].iloc[0]) / nm if nm > 0 else 0.0
                    bars_err.append(err)

            bar_colors = [COLORS[a] for a in abs_plot]
            rects = ax.bar(x, bars_eff, width, color=bar_colors, alpha=0.85,
                           yerr=bars_err, capsize=3,
                           error_kw={"linewidth": 0.8, "color": "dimgray"})

            ax.axhline(1.0,  color="black",   linewidth=0.9, linestyle="--", zorder=1)
            ax.axhline(0.85, color="#888888",  linewidth=0.7, linestyle=":",  zorder=1)

            for rect, eff in zip(rects, bars_eff):
                if not np.isnan(eff) and eff > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2.0, eff + 0.02,
                            f"{eff:.2f}", ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(abs_plot, fontsize=8)
            ax.set_ylim(0, max(1.4, max(bars_eff) + 0.2) if bars_eff else 1.4)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            if i == 0:
                ax.set_title(SIZE_LABELS[j], fontsize=9, fontweight="bold")
            if j == 0:
                label = GRAPH_LABELS[gtype].split("\n")[0]
                ax.set_ylabel(f"{label}\nEfficiency", fontsize=7.5)
            if i == len(GRAPH_TYPES) - 1:
                ax.set_xlabel("Abstraction", fontsize=8)

    handles = [mpatches.Patch(color=COLORS[a], label=a) for a in abs_plot]
    handles += [
        plt.Line2D([0], [0], color="black",  linestyle="--", label="native (1.0)"),
        plt.Line2D([0], [0], color="#888888", linestyle=":",  label="deep-profile (0.85)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "E6 BFS — Abstraction Efficiency vs Native CUDA\n"
        "(level-set BFS; metric: GTEPS = n_edges / time_s / 1e9; latency- and memory-bound)",
        fontsize=10, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig23_e6_efficiency_by_graph.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 24: frontier width profile ────────────────────────────────────────
def fig24_frontier_profile():
    """
    Line chart of frontier width vs level index.
    One subplot per (graph_type, problem_size).
    Profile data loaded from bfs_profile_*_{PLATFORM}_*.csv.
    """
    import glob
    profile_files = sorted(glob.glob(
        os.path.join(DATA_RAW, f"bfs_profile_{PLATFORM}_*.csv")))
    if not profile_files:
        print("  fig24: no profile CSVs found — skipping", file=sys.stderr)
        return

    # Profile CSV has unquoted frontier_widths (commas within field) — rebuild manually
    FIXED_COLS = ["timestamp", "platform", "graph_type", "problem_size",
                  "n_vertices", "n_levels"]
    n_fixed = len(FIXED_COLS)
    rows_raw = []
    for fpath in profile_files:
        with open(fpath) as fh:
            for lineno, line in enumerate(fh):
                if lineno == 0:
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) < n_fixed + 1:
                    continue
                rec = {FIXED_COLS[i]: parts[i] for i in range(n_fixed)}
                rec["frontier_widths"] = ",".join(parts[n_fixed:])
                rows_raw.append(rec)
    if not rows_raw:
        print("  fig24: empty profile data — skipping", file=sys.stderr)
        return
    pdata = pd.DataFrame(rows_raw)
    pdata = pdata.drop_duplicates(subset=["graph_type", "problem_size"])

    size_colors = {"small": "#2b4590", "medium": "#e87722", "large": "#2ca02c"}

    fig, axes = plt.subplots(
        1, len(GRAPH_TYPES),
        figsize=(13, 5),
        gridspec_kw={"wspace": 0.3}
    )

    for j, gtype in enumerate(GRAPH_TYPES):
        ax = axes[j]
        sub = pdata[pdata["graph_type"] == gtype]
        if sub.empty:
            ax.set_visible(False)
            continue

        for sz in SIZE_ORDER:
            row = sub[sub["problem_size"] == sz]
            if row.empty:
                continue
            widths_str = str(row["frontier_widths"].iloc[0])
            try:
                widths = [int(w) for w in widths_str.split(",") if w]
            except ValueError:
                continue
            n_v = int(row["n_vertices"].iloc[0]) if "n_vertices" in row.columns else 0
            lbl = f"{sz} (N={n_v})"
            ax.plot(range(len(widths)), widths,
                    color=size_colors.get(sz, "gray"), linewidth=1.4,
                    label=lbl, alpha=0.9)

        ax.set_xlabel("BFS level index", fontsize=10)
        ax.set_ylabel("Frontier width (vertices)", fontsize=10)
        ax.set_title(f"{GRAPH_LABELS[gtype]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        "E6 BFS — Frontier Width Profile per Level\n"
        "(shows parallelism available at each BFS level; peak width drives GPU utilisation)",
        fontsize=10, fontweight="bold"
    )

    out = os.path.join(FIG_DIR, "fig24_e6_frontier_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 25: efficiency vs frontier_irregularity scatter ───────────────────
def fig25_efficiency_vs_irregularity(df: pd.DataFrame):
    abs_plot = [a for a in ABSTRACTION_ORDER if a != "native"]
    sub = df[df["abstraction"].isin(abs_plot)].copy()

    if sub.empty or sub["frontier_irregularity"].isna().all():
        print("  fig25: no data or no irregularity values — skipping", file=sys.stderr)
        return

    graph_markers = {
        "erdos_renyi": "o",
        "2d_grid":     "^",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for gtype, marker in graph_markers.items():
        for a in abs_plot:
            pts = sub[(sub["abstraction"] == a) & (sub["graph_type"] == gtype)]
            if pts.empty:
                continue
            ax.scatter(
                pts["frontier_irregularity"],
                pts["efficiency"],
                c=COLORS[a], marker=marker, s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5,
                label=f"{a} / {gtype.replace('_', ' ')}"
            )
            for _, row in pts.iterrows():
                ax.annotate(
                    row["problem_size"][0].upper(),  # S/M/L
                    (row["frontier_irregularity"], row["efficiency"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=COLORS[a]
                )

    ax.axhline(1.0,  color="black",  linewidth=0.9, linestyle="--",
               label="native (1.0)")
    ax.axhline(0.85, color="#888888", linewidth=0.7, linestyle=":",
               label="deep-profile threshold (0.85)")

    ax.set_xlabel("Frontier Irregularity  (σ / μ of per-level frontier widths)", fontsize=10)
    ax.set_ylabel("Efficiency vs native CUDA", fontsize=10)
    ax.set_title(
        "E6 BFS — Efficiency vs Frontier Irregularity\n"
        "(S=small, M=medium, L=large; ○=Erdős–Rényi, △=2D Grid)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=0)

    out = os.path.join(FIG_DIR, "fig25_e6_efficiency_vs_irregularity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot_e6] ERROR: {SUMMARY_CSV} not found.", file=sys.stderr)
        print("[plot_e6] Run process_e6.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"[plot_e6] Loaded {len(df)} rows from {SUMMARY_CSV}")

    print("[plot_e6] Generating fig23 (efficiency bar charts) ...")
    fig23_efficiency_bars(df)

    print("[plot_e6] Generating fig24 (frontier width profiles) ...")
    fig24_frontier_profile()

    print("[plot_e6] Generating fig25 (efficiency vs irregularity scatter) ...")
    fig25_efficiency_vs_irregularity(df)

    print(f"[plot_e6] All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
