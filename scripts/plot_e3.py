#!/usr/bin/env python3
"""
E3 3D Stencil — figures.

fig7_e3_throughput_by_size.png   — grouped bars: GB/s by abstraction and size
fig8_e3_efficiency_bars.png      — grouped bars: efficiency vs native by size
fig9_e3_scaling.png              — line plot: GB/s vs N for each abstraction
fig10_e3_ppc_bars.png            — horizontal bars: efficiency at large size
fig11_e3_cv_stability.png        — coefficient of variation heatmap

Output: figures/e3/
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
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e3")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e3_stencil_summary.csv")

PROBLEM_SIZES = {"small": 32, "medium": 128, "large": 256}
SIZE_ORDER    = ["small", "medium", "large"]
SIZE_LABELS   = [f"small\n(N=32)", f"medium\n(N=128)", f"large\n(N=256)"]

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]

# Colour palette — consistent with E1/E2 figures
COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "sycl":   "#9467bd",
    "julia":  "#d62728",
    "numba":  "#8c564b",
}

STYLE = {
    "figure.dpi":     150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.35,
    "font.size":        10,
}


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found. Run process_e3.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(SUMMARY_CSV)


def present_abstractions(df: pd.DataFrame) -> list:
    return [a for a in ABSTRACTION_ORDER if a in df["abstraction"].values]


# ── Fig 7: Throughput grouped bars ───────────────────────────────────────────
def fig_throughput_by_size(df: pd.DataFrame):
    abstractions = present_abstractions(df)
    n_abs   = len(abstractions)
    n_sizes = len(SIZE_ORDER)
    x       = np.arange(n_sizes)
    width   = 0.8 / n_abs

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, abs_name in enumerate(abstractions):
            sub = df[df["abstraction"] == abs_name].set_index("problem_size")
            vals  = [sub.loc[sz, "median_gbs"] if sz in sub.index else 0.0 for sz in SIZE_ORDER]
            errs  = [sub.loc[sz, "iqr_gbs"] / 2 if sz in sub.index else 0.0 for sz in SIZE_ORDER]
            offsets = x + (i - n_abs / 2 + 0.5) * width
            ax.bar(offsets, vals, width * 0.9,
                   color=COLORS.get(abs_name, "grey"), alpha=0.85,
                   label=abs_name, yerr=errs, capsize=3, ecolor="k", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_LABELS)
        ax.set_xlabel("Problem size")
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title("E3 3D Stencil — Throughput by Abstraction and Size\n"
                     "(7-point Jacobi, FP64, error bars = IQR/2)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig7_e3_throughput_by_size.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 8: Efficiency grouped bars ───────────────────────────────────────────
def fig_efficiency_bars(df: pd.DataFrame):
    non_native = [a for a in present_abstractions(df) if a != "native"]
    n_abs   = len(non_native)
    n_sizes = len(SIZE_ORDER)
    x       = np.arange(n_sizes)
    width   = 0.8 / max(n_abs, 1)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, abs_name in enumerate(non_native):
            sub = df[df["abstraction"] == abs_name].set_index("problem_size")
            vals = [float(sub.loc[sz, "efficiency"]) if sz in sub.index
                    and not pd.isna(sub.loc[sz, "efficiency"]) else 0.0
                    for sz in SIZE_ORDER]
            offsets = x + (i - n_abs / 2 + 0.5) * width
            ax.bar(offsets, vals, width * 0.9,
                   color=COLORS.get(abs_name, "grey"), alpha=0.85, label=abs_name)

        # Reference lines
        ax.axhline(1.0,  color="black",   linestyle="-",  linewidth=1.0, label="native (1.0)")
        ax.axhline(0.80, color="#e87722", linestyle="--", linewidth=0.8, alpha=0.7, label="PPC excellent (0.80)")
        ax.axhline(0.60, color="#d62728", linestyle=":",  linewidth=0.8, alpha=0.7, label="PPC acceptable (0.60)")

        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_LABELS)
        ax.set_xlabel("Problem size")
        ax.set_ylabel("Efficiency (abstraction / native)")
        ax.set_ylim(0, 1.4)
        ax.set_title("E3 3D Stencil — Efficiency vs Native Baseline\n"
                     "(memory-bound: all abstractions expected near 1.0)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig8_e3_efficiency_bars.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 9: Scaling line plot ──────────────────────────────────────────────────
def fig_scaling(df: pd.DataFrame):
    abstractions = present_abstractions(df)
    ns = [PROBLEM_SIZES[sz] for sz in SIZE_ORDER]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for abs_name in abstractions:
            sub = df[df["abstraction"] == abs_name].set_index("problem_size")
            vals = []
            for sz in SIZE_ORDER:
                if sz in sub.index and not pd.isna(sub.loc[sz, "median_gbs"]):
                    vals.append(float(sub.loc[sz, "median_gbs"]))
                else:
                    vals.append(np.nan)
            ax.plot(ns, vals, "o-", color=COLORS.get(abs_name, "grey"),
                    linewidth=1.8, markersize=6, label=abs_name)

        ax.set_xscale("log")
        ax.set_xlabel("Grid side N (N³ total cells)")
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title("E3 3D Stencil — GB/s vs Grid Size")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig9_e3_scaling.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 10: PPC horizontal bars (large size) ─────────────────────────────────
def fig_ppc_bars(df: pd.DataFrame):
    sub = df[(df["problem_size"] == "large")].copy()
    sub = sub[sub["abstraction"].isin(present_abstractions(df))]
    sub["_ord"] = sub["abstraction"].map({a: i for i, a in enumerate(ABSTRACTION_ORDER)})
    sub = sub.sort_values("_ord")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [COLORS.get(a, "grey") for a in sub["abstraction"]]
        ys = np.arange(len(sub))
        ax.barh(ys, sub["efficiency"].fillna(0), color=colors, alpha=0.85,
                xerr=sub["iqr_gbs"] / sub["median_gbs"].clip(lower=1e-9) / 2,
                capsize=3, ecolor="k", linewidth=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["abstraction"])
        ax.axvline(1.0,  color="black",   linestyle="-",  linewidth=1.0)
        ax.axvline(0.80, color="#e87722", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0.60, color="#d62728", linestyle=":",  linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Efficiency (vs native)")
        ax.set_title("E3 3D Stencil — Efficiency at Large (N=256³)")
        ax.grid(axis="x")
        ax.grid(axis="y", alpha=0.0)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig10_e3_ppc_bars.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 11: CV stability heatmap ─────────────────────────────────────────────
def fig_cv_stability(df: pd.DataFrame):
    abstractions = present_abstractions(df)
    cv_mat = np.full((len(abstractions), len(SIZE_ORDER)), np.nan)
    for i, abs_name in enumerate(abstractions):
        sub = df[df["abstraction"] == abs_name].set_index("problem_size")
        for j, sz in enumerate(SIZE_ORDER):
            if sz in sub.index:
                cv_mat[i, j] = float(sub.loc[sz, "cv_pct"])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cv_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=5)
        ax.set_xticks(range(len(SIZE_ORDER)))
        ax.set_xticklabels(SIZE_LABELS)
        ax.set_yticks(range(len(abstractions)))
        ax.set_yticklabels(abstractions)
        for i in range(len(abstractions)):
            for j in range(len(SIZE_ORDER)):
                if not np.isnan(cv_mat[i, j]):
                    ax.text(j, i, f"{cv_mat[i,j]:.2f}%", ha="center", va="center",
                            fontsize=8, color="black" if cv_mat[i, j] < 2.5 else "white")
        plt.colorbar(im, ax=ax, label="CV (%)")
        ax.set_title("E3 3D Stencil — Coefficient of Variation (lower = more stable)")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig11_e3_cv_stability.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[plot_e3] Loading e3_stencil_summary.csv ...")
    df = load_summary()
    print(f"  {len(df)} rows, abstractions: {df['abstraction'].unique().tolist()}")

    print("[plot_e3] Generating figures → figures/e3/")
    fig_throughput_by_size(df)
    fig_efficiency_bars(df)
    fig_scaling(df)
    fig_ppc_bars(df)
    fig_cv_stability(df)

    print("[plot_e3] Done.")


if __name__ == "__main__":
    main()
