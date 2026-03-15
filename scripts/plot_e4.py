#!/usr/bin/env python3
"""
E4 SpMV — figures.

fig13_e4_throughput_by_matrix.png  — grouped bars: GFLOP/s by abstraction per matrix type
fig14_e4_efficiency_heatmap.png    — heatmap: efficiency[abstraction × matrix_type] at large size
fig15_e4_scaling.png               — line plots: GFLOP/s vs N for each abstraction × matrix type
fig16_e4_ppc_by_matrix.png         — horizontal bars: efficiency at large size per matrix type
fig17_e4_cv_stability.png          — coefficient of variation heatmap

Output: figures/e4/
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
FIG_DIR   = os.path.join(REPO_ROOT, "figures", "e4")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e4_spmv_summary.csv")

PROBLEM_SIZES = {"small": 1024, "medium": 8192, "large": 32768}
SIZE_ORDER    = ["small", "medium", "large"]
SIZE_LABELS   = [f"small\n(N≈1K)", f"medium\n(N≈8K)", f"large\n(N≈32K)"]

ABSTRACTION_ORDER = ["native", "kokkos", "raja", "sycl", "julia", "numba"]
MATRIX_TYPES      = ["laplacian_2d", "random_sparse", "power_law"]
MATRIX_LABELS     = {
    "laplacian_2d":  "Laplacian 2D\n(structured)",
    "random_sparse": "Random Sparse\n(uniform)",
    "power_law":     "Power Law\n(imbalanced)",
}

# Colour palette — consistent with E1/E2/E3 figures
COLORS = {
    "native": "#2b4590",
    "kokkos": "#e87722",
    "raja":   "#2ca02c",
    "sycl":   "#9467bd",
    "julia":  "#d62728",
    "numba":  "#8c564b",
}

# Matrix type line styles for scaling plots
MTYPE_LINESTYLE = {
    "laplacian_2d":  "-",
    "random_sparse": "--",
    "power_law":     ":",
}

STYLE = {
    "figure.dpi":      150,
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
        print(f"ERROR: {SUMMARY_CSV} not found. Run process_e4.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(SUMMARY_CSV)


def present_abstractions(df: pd.DataFrame) -> list:
    return [a for a in ABSTRACTION_ORDER if a in df["abstraction"].values]


def present_matrices(df: pd.DataFrame) -> list:
    return [m for m in MATRIX_TYPES if m in df["matrix_type"].values]


# ── Fig 13: Throughput grouped bars (one subplot per matrix type) ─────────────
def fig_throughput_by_matrix(df: pd.DataFrame):
    matrices     = present_matrices(df)
    abstractions = present_abstractions(df)
    n_abs = len(abstractions)
    x     = np.arange(len(SIZE_ORDER))
    width = 0.8 / max(n_abs, 1)
    n_mat = len(matrices)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 5), sharey=True)
        if n_mat == 1:
            axes = [axes]
        for ax, mtype in zip(axes, matrices):
            sub_m = df[df["matrix_type"] == mtype]
            for i, abs_name in enumerate(abstractions):
                sub = sub_m[sub_m["abstraction"] == abs_name].set_index("problem_size")
                vals = [sub.loc[sz, "median_gflops"] if sz in sub.index else 0.0
                        for sz in SIZE_ORDER]
                errs = [sub.loc[sz, "iqr_gflops"] / 2 if sz in sub.index else 0.0
                        for sz in SIZE_ORDER]
                offsets = x + (i - n_abs / 2 + 0.5) * width
                ax.bar(offsets, vals, width * 0.9,
                       color=COLORS.get(abs_name, "grey"), alpha=0.85,
                       label=abs_name, yerr=errs, capsize=2, ecolor="k", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(SIZE_LABELS, fontsize=9)
            ax.set_xlabel("Problem size")
            ax.set_title(MATRIX_LABELS.get(mtype, mtype), fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel("Throughput (GFLOP/s)")
        handles = [mpatches.Patch(color=COLORS.get(a, "grey"), label=a)
                   for a in abstractions]
        axes[-1].legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)
        fig.suptitle("E4 SpMV — Throughput by Matrix Type and Size\n"
                     "(CSR, one thread per row, FP64, error bars = IQR/2)", fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig13_e4_throughput_by_matrix.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 14: Efficiency heatmap (abstraction × matrix_type at large size) ───────
def fig_efficiency_heatmap(df: pd.DataFrame):
    non_native   = [a for a in present_abstractions(df) if a != "native"]
    matrices     = present_matrices(df)
    sub = df[(df["problem_size"] == "large") & (df["abstraction"].isin(non_native))].copy()

    eff_mat = np.full((len(non_native), len(matrices)), np.nan)
    for i, abs_name in enumerate(non_native):
        for j, mtype in enumerate(matrices):
            row = sub[(sub["abstraction"] == abs_name) & (sub["matrix_type"] == mtype)]
            if not row.empty and not pd.isna(row.iloc[0]["efficiency"]):
                eff_mat[i, j] = float(row.iloc[0]["efficiency"])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(eff_mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.2)
        ax.set_xticks(range(len(matrices)))
        ax.set_xticklabels([MATRIX_LABELS.get(m, m).replace("\n", " ") for m in matrices],
                           fontsize=9)
        ax.set_yticks(range(len(non_native)))
        ax.set_yticklabels(non_native)
        for i in range(len(non_native)):
            for j in range(len(matrices)):
                if not np.isnan(eff_mat[i, j]):
                    txt = f"{eff_mat[i,j]:.3f}"
                    color = "black" if 0.5 < eff_mat[i, j] < 1.1 else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Efficiency (abstraction / native)")
        ax.set_title("E4 SpMV — Efficiency Heatmap at Large Size (N≈32K)\n"
                     "(green ≥ 0.80 = excellent, red < 0.60 = poor)")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig14_e4_efficiency_heatmap.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Fig 15: Scaling line plots (one subplot per matrix type) ──────────────────
def fig_scaling(df: pd.DataFrame):
    matrices     = present_matrices(df)
    abstractions = present_abstractions(df)
    ns = [PROBLEM_SIZES[sz] for sz in SIZE_ORDER]
    n_mat = len(matrices)

    with plt.rc_context(STYLE):
        with plt.rc_context({"axes.grid.axis": "both"}):
            fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 5), sharey=True)
            if n_mat == 1:
                axes = [axes]
            for ax, mtype in zip(axes, matrices):
                sub_m = df[df["matrix_type"] == mtype]
                for abs_name in abstractions:
                    sub = sub_m[sub_m["abstraction"] == abs_name].set_index("problem_size")
                    vals = []
                    for sz in SIZE_ORDER:
                        if sz in sub.index and not pd.isna(sub.loc[sz, "median_gflops"]):
                            vals.append(float(sub.loc[sz, "median_gflops"]))
                        else:
                            vals.append(np.nan)
                    ax.plot(ns, vals, "o-", color=COLORS.get(abs_name, "grey"),
                            linewidth=1.8, markersize=6, label=abs_name)
                ax.set_xscale("log")
                ax.set_xlabel("Matrix rows N")
                ax.set_title(MATRIX_LABELS.get(mtype, mtype), fontsize=10)
                ax.set_xticks(ns)
                ax.set_xticklabels(["1K", "8K", "32K"], fontsize=9)
                if ax is axes[0]:
                    ax.set_ylabel("Throughput (GFLOP/s)")
            axes[-1].legend(loc="best", framealpha=0.9, fontsize=9)
            fig.suptitle("E4 SpMV — GFLOP/s Scaling vs Problem Size", fontsize=11)
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "fig15_e4_scaling.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out}")


# ── Fig 16: PPC horizontal bars at large size, per matrix type ────────────────
def fig_ppc_by_matrix(df: pd.DataFrame):
    non_native = [a for a in present_abstractions(df) if a != "native"]
    matrices   = present_matrices(df)
    sub = df[(df["problem_size"] == "large") & (df["abstraction"].isin(non_native))].copy()
    sub["_ord"] = sub["abstraction"].map({a: i for i, a in enumerate(ABSTRACTION_ORDER)})
    n_mat = len(matrices)

    with plt.rc_context(STYLE):
        with plt.rc_context({"axes.grid.axis": "x"}):
            fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 4), sharey=True)
            if n_mat == 1:
                axes = [axes]
            for ax, mtype in zip(axes, matrices):
                sub_m = sub[sub["matrix_type"] == mtype].sort_values("_ord")
                colors = [COLORS.get(a, "grey") for a in sub_m["abstraction"]]
                ys = np.arange(len(sub_m))
                ax.barh(ys, sub_m["efficiency"].fillna(0), color=colors, alpha=0.85)
                ax.set_yticks(ys)
                if ax is axes[0]:
                    ax.set_yticklabels(sub_m["abstraction"])
                else:
                    ax.set_yticklabels([])
                ax.axvline(1.0,  color="black",   linestyle="-",  linewidth=1.0)
                ax.axvline(0.80, color="#e87722", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvline(0.60, color="#d62728", linestyle=":",  linewidth=0.8, alpha=0.7)
                ax.set_xlim(0, 1.5)
                ax.set_xlabel("Efficiency (vs native)")
                ax.set_title(MATRIX_LABELS.get(mtype, mtype), fontsize=10)
            fig.suptitle("E4 SpMV — Efficiency at Large Size (N≈32K) per Matrix Type\n"
                         "(dashed = PPC 0.80 excellent threshold, dotted = 0.60 acceptable)",
                         fontsize=10)
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "fig16_e4_ppc_by_matrix.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out}")


# ── Fig 17: CV stability heatmap (abstraction × size, for each matrix type) ───
def fig_cv_stability(df: pd.DataFrame):
    abstractions = present_abstractions(df)
    matrices     = present_matrices(df)
    n_mat = len(matrices)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 4), sharey=True)
        if n_mat == 1:
            axes = [axes]
        for ax, mtype in zip(axes, matrices):
            sub_m = df[df["matrix_type"] == mtype]
            cv_mat = np.full((len(abstractions), len(SIZE_ORDER)), np.nan)
            for i, abs_name in enumerate(abstractions):
                sub = sub_m[sub_m["abstraction"] == abs_name].set_index("problem_size")
                for j, sz in enumerate(SIZE_ORDER):
                    if sz in sub.index and not pd.isna(sub.loc[sz, "cv_pct"]):
                        cv_mat[i, j] = float(sub.loc[sz, "cv_pct"])
            im = ax.imshow(cv_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=5)
            ax.set_xticks(range(len(SIZE_ORDER)))
            ax.set_xticklabels(SIZE_LABELS, fontsize=9)
            ax.set_yticks(range(len(abstractions)))
            if ax is axes[0]:
                ax.set_yticklabels(abstractions)
            else:
                ax.set_yticklabels([])
            for i in range(len(abstractions)):
                for j in range(len(SIZE_ORDER)):
                    if not np.isnan(cv_mat[i, j]):
                        ax.text(j, i, f"{cv_mat[i,j]:.2f}%",
                                ha="center", va="center", fontsize=8,
                                color="black" if cv_mat[i, j] < 2.5 else "white")
            ax.set_title(MATRIX_LABELS.get(mtype, mtype), fontsize=10)
        # Shared colorbar on last axis
        plt.colorbar(im, ax=axes[-1], label="CV (%)")
        fig.suptitle("E4 SpMV — Coefficient of Variation (lower = more stable)", fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig17_e4_cv_stability.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[plot_e4] Loading e4_spmv_summary.csv ...")
    df = load_summary()
    print(f"  {len(df)} rows, abstractions: {df['abstraction'].unique().tolist()}")
    print(f"  matrix types: {df['matrix_type'].unique().tolist()}")

    print("[plot_e4] Generating figures → figures/e4/")
    fig_throughput_by_matrix(df)
    fig_efficiency_heatmap(df)
    fig_scaling(df)
    fig_ppc_by_matrix(df)
    fig_cv_stability(df)

    print("[plot_e4] Done.")


if __name__ == "__main__":
    main()
