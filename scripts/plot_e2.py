#!/usr/bin/env python3
"""
E2 DGEMM — publication-quality figures (Nature/IEEE style).

Generates 5 figures and saves to figures/e2/:
  fig1_e2_throughput_by_size.png   — grouped bar: GFLOP/s per abstraction × size
  fig2_e2_efficiency_bars.png      — efficiency vs native, flagged bars highlighted
  fig3_e2_scaling_crossover.png    — scaling lines: julia_naive crosses native at large N
  fig4_e2_ppc_bars.png             — harmonic-mean PPC per abstraction
  fig5_e2_cv_stability.png         — coefficient of variation per (abstraction, size)
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import hmean

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.4,
    "patch.linewidth":   0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIGURES   = os.path.join(REPO_ROOT, "figures", "e2")
os.makedirs(FIGURES, exist_ok=True)

# ── Abstraction ordering, colors, labels ──────────────────────────────────────
# Non-ceiling abstractions that ran (kokkos/sycl/numba absent from this dataset)
ABSTRACTIONS = ["native", "raja_naive", "julia_naive"]

# IBM colorblind-safe palette (consistent with plot_e1.py)
COLORS = {
    "native":      "#648FFF",   # blue
    "kokkos":      "#785EF0",   # violet  (not run — reserved)
    "raja_naive":  "#FFB000",   # amber
    "sycl":        "#FE6100",   # orange  (not run — reserved)
    "julia_naive": "#DC267F",   # magenta
    "numba":       "#A0A0A0",   # grey (UNSUPPORTED_CC120)
}

LABELS = {
    "native":      "CUDA (native)",
    "kokkos":      "Kokkos",
    "raja_naive":  "RAJA (naive)",
    "sycl":        "SYCL",
    "julia_naive": "Julia/CUDA.jl\n(naive)",
    "numba":       "Numba\n(UNSUPPORTED_CC120)",
}

# Problem sizes in display order
SIZES       = ["small", "medium", "large"]
SIZE_N      = {"small": 1024, "medium": 4096, "large": 8192}
SIZE_COLORS = {"small": "#4E79A7", "medium": "#F28E2B", "large": "#59A14F"}
SIZE_LABELS = {"small": "N=1 024 (small)", "medium": "N=4 096 (medium)",
               "large": "N=8 192 (large)"}

FLAG_THRESH = 0.85   # deep-profiling trigger
PPC_GREEN   = 0.85
PPC_YELLOW  = 0.50

# ── Load data ─────────────────────────────────────────────────────────────────
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(DATA_PROC, "e2_dgemm_summary.csv")
    df = pd.read_csv(path)
    df["is_ceiling_ref"] = df["is_ceiling_ref"].astype(str).str.lower() == "true"
    df["flag_deep_profiling"] = df["flag_deep_profiling"].astype(str).str.lower() == "true"
    perf = df[~df["is_ceiling_ref"]].copy()
    ceil = df[df["is_ceiling_ref"]].copy()
    return perf, ceil


# ── Fig 1 — Throughput grouped bar by size ────────────────────────────────────
def fig1_throughput(perf: pd.DataFrame, ceil: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    n_abs  = len(ABSTRACTIONS)
    n_sz   = len(SIZES)
    width  = 0.22
    group_gap = 0.08
    x = np.arange(n_abs) * (n_sz * width + group_gap)

    for si, sz in enumerate(SIZES):
        offsets = x + si * width - (n_sz - 1) * width / 2
        vals, errs = [], []
        for ab in ABSTRACTIONS:
            row = perf[(perf["abstraction"] == ab) & (perf["problem_size"] == sz)]
            if len(row) == 0:
                vals.append(0); errs.append(0)
            else:
                vals.append(row["median_gflops"].values[0])
                errs.append(row["iqr_gflops"].values[0] / 2)

        bars = ax.bar(offsets, vals, width=width,
                      color=SIZE_COLORS[sz], label=SIZE_LABELS[sz],
                      yerr=errs, capsize=2,
                      error_kw={"linewidth": 0.7, "ecolor": "#333333"},
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Annotate julia_naive/large
        if sz == "large":
            jn_idx = ABSTRACTIONS.index("julia_naive")
            ax.annotate("η=1.25\n(Pattern 5)",
                        xy=(offsets[jn_idx], vals[jn_idx]),
                        xytext=(offsets[jn_idx] + 0.01, vals[jn_idx] + 14),
                        fontsize=6.5, ha="center", color="#8B0057",
                        arrowprops=dict(arrowstyle="-", color="#8B0057",
                                        lw=0.7, relpos=(0.5, 0)))

    # cuBLAS ceiling horizontal lines per size
    ceil_by_size = ceil[ceil["abstraction"] == "native_cublas"].set_index("problem_size")
    linestyles = {"small": (0, (5, 3)), "medium": (0, (3, 2)), "large": "dashed"}
    for sz in SIZES:
        if sz in ceil_by_size.index:
            c_val = ceil_by_size.loc[sz, "median_gflops"]
            ax.axhline(c_val, linestyle=linestyles[sz],
                       color=SIZE_COLORS[sz], linewidth=0.8, alpha=0.6,
                       label=f"cuBLAS ceiling ({sz}: {c_val:.0f} GFLOP/s)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ABSTRACTIONS], rotation=0, ha="center")
    ax.set_ylabel("Median GFLOP/s")
    ax.set_title("E2 DGEMM — Throughput by Abstraction and Problem Size")
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=7)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel("Abstraction")

    path = os.path.join(FIGURES, "fig1_e2_throughput_by_size.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Fig 2 — Efficiency bars ────────────────────────────────────────────────────
def fig2_efficiency(perf: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))

    n_abs = len(ABSTRACTIONS)
    n_sz  = len(SIZES)
    width = 0.22
    group_gap = 0.08
    x = np.arange(n_abs) * (n_sz * width + group_gap)

    for si, sz in enumerate(SIZES):
        offsets = x + si * width - (n_sz - 1) * width / 2
        vals = []
        for ab in ABSTRACTIONS:
            row = perf[(perf["abstraction"] == ab) & (perf["problem_size"] == sz)]
            vals.append(row["efficiency"].values[0] if len(row) else 0)

        for oi, (off, val, ab) in enumerate(zip(offsets, vals, ABSTRACTIONS)):
            # julia_naive/large gets gold for Pattern 5
            if ab == "julia_naive" and sz == "large" and val > 1.0:
                facecolor = "#FFD700"
                edgecolor = "#B8860B"
                lw = 1.2
            elif val < FLAG_THRESH and val > 0:
                facecolor = SIZE_COLORS[sz]
                edgecolor = "#CC0000"
                lw = 1.2
            else:
                facecolor = SIZE_COLORS[sz]
                edgecolor = "white"
                lw = 0.5

            ax.bar(off, val, width=width, color=facecolor,
                   edgecolor=edgecolor, linewidth=lw, zorder=3,
                   label=SIZE_LABELS[sz] if oi == 0 else None)

            # Annotate Pattern 5
            if ab == "julia_naive" and sz == "large" and val > 1.0:
                ax.annotate("Pattern 5", xy=(off, val),
                            xytext=(off, val + 0.05),
                            fontsize=6, ha="center", color="#8B0057",
                            fontweight="bold")

    # η=1.0 reference line
    ax.axhline(1.0, linestyle="--", color="#222222", linewidth=0.9,
               zorder=4, label="η=1.0 (native baseline)")
    # 0.85 threshold
    ax.axhline(FLAG_THRESH, linestyle=":", color="#CC0000", linewidth=0.8,
               zorder=4, label="η=0.85 (profiling threshold)")

    # UNSUPPORTED_CC120 note for numba
    ax.text(0.98, 0.05,
            "numba: UNSUPPORTED_CC120\n(Numba 0.64.0 predates Blackwell CC 12.0)",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#AAAAAA", linewidth=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ABSTRACTIONS], rotation=0, ha="center")
    ax.set_ylabel("Efficiency η (vs native baseline)")
    ax.set_title("E2 DGEMM — Efficiency by Abstraction and Problem Size")
    # Deduplicate legend
    handles, labels_l = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_l):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right",
              frameon=False, ncol=1, fontsize=7)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel("Abstraction")

    # Red border legend patch
    red_patch = mpatches.Patch(facecolor="#DDDDDD", edgecolor="#CC0000",
                                linewidth=1.2, label="Below 0.85 threshold (red border)")
    gold_patch = mpatches.Patch(facecolor="#FFD700", edgecolor="#B8860B",
                                 linewidth=1.2, label="Pattern 5 (η>1.0, gold)")
    ax.legend(handles=list(seen.values()) + [red_patch, gold_patch],
              labels=list(seen.keys()) + ["Below η=0.85 (red border)",
                                          "Pattern 5: η>1.0 (gold)"],
              loc="upper right", frameon=False, ncol=1, fontsize=6.5)

    path = os.path.join(FIGURES, "fig2_e2_efficiency_bars.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Fig 3 — Scaling crossover ──────────────────────────────────────────────────
def fig3_crossover(perf: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))

    size_x = [1024, 4096, 8192]
    size_labels = ["1 024\n(small)", "4 096\n(medium)", "8 192\n(large)"]

    for ab in ABSTRACTIONS:
        sub = perf[perf["abstraction"] == ab].set_index("n_matrix")
        ys = [sub.loc[n, "median_gflops"] if n in sub.index else np.nan
              for n in size_x]
        ax.plot(range(len(size_x)), ys,
                color=COLORS[ab], marker="o", markersize=5,
                label=LABELS[ab].replace("\n", " "), linewidth=1.5, zorder=3)
        # Label endpoints
        if not np.isnan(ys[-1]):
            ax.annotate(f"{ys[-1]:.0f}",
                        xy=(2, ys[-1]),
                        xytext=(2.05, ys[-1]),
                        fontsize=7, va="center", color=COLORS[ab])

    # Annotate crossover: julia_naive crosses native between medium and large
    ax.annotate("Crossover:\njulia_naive > native",
                xy=(1.6, 176),
                xytext=(1.1, 240),
                fontsize=7, color="#8B0057",
                arrowprops=dict(arrowstyle="->", color="#8B0057",
                                lw=0.9, connectionstyle="arc3,rad=0.2"))

    ax.set_xticks(range(len(size_x)))
    ax.set_xticklabels(size_labels)
    ax.set_ylabel("Median GFLOP/s")
    ax.set_xlabel("Problem Size (N×N matrix)")
    ax.set_title("E2 DGEMM — Scaling: julia_naive crosses native at large N\n"
                 "(Pattern 5: Layout-Induced Coalescing Advantage)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.2, 2.5)

    path = os.path.join(FIGURES, "fig3_e2_scaling_crossover.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Fig 4 — PPC bars (harmonic mean over sizes) ───────────────────────────────
def fig4_ppc(perf: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ppc_vals = {}
    for ab in ABSTRACTIONS:
        if ab == "native":
            continue
        effs = []
        for sz in SIZES:
            row = perf[(perf["abstraction"] == ab) & (perf["problem_size"] == sz)]
            if len(row) and row["efficiency"].values[0] > 0:
                effs.append(row["efficiency"].values[0])
        if effs:
            ppc_vals[ab] = float(hmean(effs))

    abs_order = [a for a in ABSTRACTIONS if a in ppc_vals]
    ppc_list  = [ppc_vals[a] for a in abs_order]

    bar_colors = []
    for v in ppc_list:
        if v >= PPC_GREEN:
            bar_colors.append("#2CA02C")   # green
        elif v >= PPC_YELLOW:
            bar_colors.append("#DBAB09")   # yellow
        else:
            bar_colors.append("#D62728")   # red

    y = np.arange(len(abs_order))
    bars = ax.barh(y, ppc_list, color=bar_colors, edgecolor="white",
                   linewidth=0.5, height=0.55, zorder=3)

    # Value labels
    for i, (v, bar) in enumerate(zip(ppc_list, bars)):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8,
                color="#222222")

    # Threshold line
    ax.axvline(PPC_GREEN, linestyle="--", color="#555555", linewidth=0.9,
               zorder=4, label=f"η={PPC_GREEN:.2f} threshold")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[a].replace("\n", " ") for a in abs_order])
    ax.set_xlabel("PPC (harmonic mean of η across small/medium/large)")
    ax.set_title("E2 DGEMM — Performance Portability Coefficient")
    ax.set_xlim(0, max(ppc_list) * 1.18)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)

    # Color legend
    patches = [
        mpatches.Patch(color="#2CA02C", label=f"PPC ≥ {PPC_GREEN:.2f} (excellent)"),
        mpatches.Patch(color="#DBAB09", label=f"PPC {PPC_YELLOW:.2f}–{PPC_GREEN:.2f} (acceptable)"),
        mpatches.Patch(color="#D62728", label=f"PPC < {PPC_YELLOW:.2f} (poor)"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=7, loc="lower right")

    path = os.path.join(FIGURES, "fig4_e2_ppc_bars.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Fig 5 — CV stability scatter ──────────────────────────────────────────────
def fig5_cv(perf: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.8))

    CV_THRESH = 2.0
    x_pos = {ab: i for i, ab in enumerate(ABSTRACTIONS)}
    jitter = {"small": -0.12, "medium": 0.0, "large": 0.12}

    for sz in SIZES:
        sub = perf[perf["problem_size"] == sz]
        xs, ys = [], []
        for ab in ABSTRACTIONS:
            row = sub[sub["abstraction"] == ab]
            if len(row):
                xs.append(x_pos[ab] + jitter[sz])
                ys.append(row["cv_pct"].values[0])
        ax.scatter(xs, ys, color=SIZE_COLORS[sz], s=48, zorder=3,
                   label=SIZE_LABELS[sz], edgecolors="white", linewidths=0.5)

        # Annotate points above threshold
        for x_val, y_val, ab in zip(xs, ys,
                [a for a in ABSTRACTIONS
                 if len(sub[sub["abstraction"]==a])]):
            if y_val > CV_THRESH:
                ax.annotate(f"{y_val:.1f}%",
                            xy=(x_val, y_val),
                            xytext=(x_val + 0.08, y_val + 0.05),
                            fontsize=6.5, color="#CC0000")

    ax.axhline(CV_THRESH, linestyle="--", color="#CC0000", linewidth=0.8,
               zorder=4, label=f"CV={CV_THRESH:.1f}% stability threshold")

    ax.set_xticks(range(len(ABSTRACTIONS)))
    ax.set_xticklabels([LABELS[a].replace("\n", " ") for a in ABSTRACTIONS])
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Run Stability — E2 DGEMM")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_xlim(-0.5, len(ABSTRACTIONS) - 0.5)
    ax.set_ylim(0, max(perf["cv_pct"].max() * 1.3, CV_THRESH * 1.4))

    path = os.path.join(FIGURES, "fig5_e2_cv_stability.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass   # fallback to rcParams already set

    perf, ceil = load()
    print(f"[plot_e2] Loaded {len(perf)} non-ceiling rows, "
          f"{len(ceil)} ceiling-ref rows")

    paths = []
    print("[plot_e2] Generating fig1 ...", end=" ", flush=True)
    paths.append(fig1_throughput(perf, ceil));  print("done")

    print("[plot_e2] Generating fig2 ...", end=" ", flush=True)
    paths.append(fig2_efficiency(perf));        print("done")

    print("[plot_e2] Generating fig3 ...", end=" ", flush=True)
    paths.append(fig3_crossover(perf));         print("done")

    print("[plot_e2] Generating fig4 ...", end=" ", flush=True)
    paths.append(fig4_ppc(perf));               print("done")

    print("[plot_e2] Generating fig5 ...", end=" ", flush=True)
    paths.append(fig5_cv(perf));                print("done")

    print("\n[plot_e2] Saved figures:")
    for p in paths:
        kb = os.path.getsize(p) / 1024
        print(f"  {os.path.relpath(p, REPO_ROOT):50s}  {kb:6.1f} KB")


if __name__ == "__main__":
    main()
