#!/usr/bin/env python3
"""
E3 3D Stencil — multi-platform efficiency + PPC figure (fig13).

Three-panel figure:
  Left   panel: NVIDIA RTX 5060 efficiency bars (η vs native, by abstraction × size)
  Centre panel: AMD MI300X efficiency bars
  Right  panel: Cross-platform PPC φ (Pennycook harmonic mean, by abstraction × size)

Abstractions shown: native, kokkos, raja, sycl, julia
  • SYCL is N/A on NVIDIA RTX 5060 (not built); shown as hatched N/A bar.
  • Native is included for reference (η = 1.0 by definition).

Output: figures/e3/fig13_e3_efficiency_multiplatform.png
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
    "patch.linewidth":   0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
FIGURES   = os.path.join(REPO_ROOT, "figures", "e3")
os.makedirs(FIGURES, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_PROC, "e3_stencil_summary.csv")
PPC_CSV     = os.path.join(DATA_PROC, "e3_ppc_scores.csv")

# ── Layout constants ───────────────────────────────────────────────────────────
ABSTRACTIONS = ["native", "kokkos", "raja", "sycl", "julia"]
ABS_LABELS   = {
    "native": "Native\n(HIP/CUDA)",
    "kokkos": "Kokkos",
    "raja":   "RAJA",
    "sycl":   "SYCL",
    "julia":  "Julia",
}
# SYCL not built on NVIDIA RTX 5060
NVIDIA_NA = {"sycl"}

SIZES       = ["small", "medium", "large"]
SIZE_LABELS = {"small": "N=32³", "medium": "N=128³", "large": "N=256³"}
SIZE_COLORS = {"small": "#4E79A7", "medium": "#F28E2B", "large": "#59A14F"}

PLATFORMS = ["nvidia_rtx5060", "amd_mi300x"]
PANEL_TITLES = {
    "nvidia_rtx5060": "NVIDIA RTX 5060\nEfficiency η vs native",
    "amd_mi300x":     "AMD MI300X\nEfficiency η vs native",
}


def load_data():
    for path in (SUMMARY_CSV, PPC_CSV):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run process_e3.py first.", file=sys.stderr)
            sys.exit(1)
    summary = pd.read_csv(SUMMARY_CSV)
    ppc     = pd.read_csv(PPC_CSV)
    return summary, ppc


def plot_efficiency_panel(ax, summary: pd.DataFrame, platform: str, title: str):
    """Grouped bar chart: x = abstraction, groups = problem size."""
    na_set   = NVIDIA_NA if platform == "nvidia_rtx5060" else set()
    df_plat  = summary[summary["platform"] == platform]

    n_abs    = len(ABSTRACTIONS)
    n_sizes  = len(SIZES)
    bar_w    = 0.22
    group_w  = n_sizes * bar_w + 0.08
    x_pos    = np.arange(n_abs) * group_w

    for s_idx, size in enumerate(SIZES):
        offsets = x_pos + s_idx * bar_w - (n_sizes - 1) * bar_w / 2
        for a_idx, abs_name in enumerate(ABSTRACTIONS):
            xval = offsets[a_idx]
            color = SIZE_COLORS[size]

            if abs_name in na_set:
                ax.bar(xval, 0.05, width=bar_w * 0.85, color="lightgrey",
                       hatch="///", edgecolor="grey", linewidth=0.5, zorder=3)
                if s_idx == 0:
                    ax.text(xval, 0.07, "N/A", ha="center", va="bottom",
                            fontsize=6, color="grey")
                continue

            row = df_plat[
                (df_plat["abstraction"] == abs_name) &
                (df_plat["problem_size"] == size)
            ]
            if row.empty:
                continue

            eff   = float(row["efficiency"].iloc[0])
            tier  = row["ppc_tier"].iloc[0]
            alpha = 0.9
            edge  = "none"

            # Red border for poor tier
            if tier == "poor":
                edge = "#c0392b"
            elif tier == "acceptable":
                edge = "#e67e22"

            ax.bar(xval, eff, width=bar_w * 0.85, color=color, alpha=alpha,
                   edgecolor=edge, linewidth=1.2 if edge != "none" else 0.5,
                   zorder=3)

    # Reference line η = 1.0 and η = 0.85
    ax.axhline(1.0, color="black",   linewidth=0.8, linestyle="-",  alpha=0.5)
    ax.axhline(0.85, color="#e67e22", linewidth=0.7, linestyle="--", alpha=0.6,
               label="η = 0.85 (acceptable threshold)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ABS_LABELS[a] for a in ABSTRACTIONS], fontsize=7.5)
    ax.set_ylabel("Efficiency η")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_ppc_panel(ax, ppc: pd.DataFrame):
    """Bar chart: x = abstraction, groups = problem size, y = φ."""
    n_abs   = len(ABSTRACTIONS)
    n_sizes = len(SIZES)
    bar_w   = 0.22
    group_w = n_sizes * bar_w + 0.08
    x_pos   = np.arange(n_abs) * group_w

    for s_idx, size in enumerate(SIZES):
        offsets = x_pos + s_idx * bar_w - (n_sizes - 1) * bar_w / 2
        for a_idx, abs_name in enumerate(ABSTRACTIONS):
            xval = offsets[a_idx]
            row  = ppc[
                (ppc["abstraction"] == abs_name) &
                (ppc["problem_size"] == size)
            ]
            if row.empty:
                continue
            phi      = float(row["phi"].iloc[0])
            n_plat   = int(row["n_platforms"].iloc[0])
            color    = SIZE_COLORS[size]
            # Single-platform scores shown with hatching (less portable)
            hatch    = "" if n_plat > 1 else "///"
            ax.bar(xval, phi, width=bar_w * 0.85, color=color, alpha=0.9,
                   hatch=hatch, edgecolor="grey" if hatch else "none",
                   linewidth=0.5, zorder=3)

    ax.axhline(1.0, color="black",    linewidth=0.8, linestyle="-",  alpha=0.5)
    ax.axhline(0.85, color="#e67e22", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ABS_LABELS[a] for a in ABSTRACTIONS], fontsize=7.5)
    ax.set_ylabel("PPC score φ")
    ax.set_ylim(0, 1.15)
    ax.set_title("Cross-Platform PPC φ\n(harmonic mean, RTX 5060 + MI300X)")
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Annotate single-platform configs
    ax.text(0.98, 0.97, "/// = single-platform only",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="grey", style="italic")


def make_legend(fig):
    size_patches = [
        mpatches.Patch(color=SIZE_COLORS[s], label=SIZE_LABELS[s])
        for s in SIZES
    ]
    tier_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#c0392b", linewidth=1.5,
                       label="Poor tier (η < 0.60)"),
        mpatches.Patch(facecolor="white", edgecolor="#e67e22", linewidth=1.5,
                       label="Acceptable tier (0.60 ≤ η < 0.80)"),
        plt.Line2D([0], [0], color="#e67e22", linestyle="--", linewidth=1.0,
                   label="η = 0.85 threshold"),
    ]
    fig.legend(handles=size_patches + tier_handles,
               loc="lower center", ncol=6, fontsize=7.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))


def main():
    summary, ppc = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for ax, platform in zip(axes[:2], PLATFORMS):
        plot_efficiency_panel(ax, summary, platform, PANEL_TITLES[platform])

    plot_ppc_panel(axes[2], ppc)
    make_legend(fig)

    fig.suptitle("E3 3D Stencil — Multi-Platform Efficiency & Portability",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES, "fig13_e3_efficiency_multiplatform.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_e3_multiplatform] Saved {out}")


if __name__ == "__main__":
    main()
