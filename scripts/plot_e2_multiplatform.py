#!/usr/bin/env python3
"""
E2 DGEMM — multi-platform efficiency figure (fig7).

Two-panel grouped-bar chart:
  Left  panel: NVIDIA RTX 5060 Laptop
  Right panel: AMD MI300X

Y-axis: efficiency η = median_gflops / native_gflops
X-axis: non-ceiling abstractions (kokkos / sycl marked N/A on NVIDIA)
Bars grouped by problem size (small / medium / large).

Key findings annotated:
  • Julia reversal: η=1.25 on NVIDIA (Pattern 5, gold) → η=0.31 on AMD (red border)
  • RAJA consistency: poor on both platforms (~0.19–0.48)
  • Kokkos/SYCL: N/A NVIDIA; near-native on AMD (η≈0.96–0.97)

Output: figures/e2/fig_e2_efficiency_multiplatform.pdf
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
FIGURES   = os.path.join(REPO_ROOT, "figures", "e2")
os.makedirs(FIGURES, exist_ok=True)

# ── Layout ────────────────────────────────────────────────────────────────────
# Abstractions shown in both panels (ceiling refs excluded)
ABSTRACTIONS = ["native", "kokkos", "raja_naive", "sycl", "julia_naive"]
ABS_LABELS   = {
    "native":     "Native\n(HIP/CUDA)",
    "kokkos":     "Kokkos",
    "raja_naive": "RAJA\n(naive)",
    "sycl":       "SYCL",
    "julia_naive":"Julia\n(naive)",
}
# Abstractions absent on NVIDIA → show N/A markers
NVIDIA_NA = {"kokkos", "sycl"}

PLATFORMS = ["nvidia_rtx5060_laptop", "amd_mi300x"]
PANEL_TITLES = {
    "nvidia_rtx5060_laptop": "NVIDIA RTX 5060 Laptop",
    "amd_mi300x":            "AMD MI300X",
}

SIZES        = ["small", "medium", "large"]
SIZE_COLORS  = {"small": "#4E79A7", "medium": "#F28E2B", "large": "#59A14F"}
SIZE_LABELS  = {"small": "N=1 024", "medium": "N=4 096", "large": "N=8 192"}

FLAG_THRESH = 0.85

# ── Load data ─────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    path = os.path.join(DATA_PROC, "e2_dgemm_summary.csv")
    df   = pd.read_csv(path)
    df["is_ceiling_ref"]   = df["is_ceiling_ref"].astype(str).str.lower() == "true"
    df["flag_deep_profiling"] = df["flag_deep_profiling"].astype(str).str.lower() == "true"
    return df[~df["is_ceiling_ref"]].copy()


# ── Draw one panel ────────────────────────────────────────────────────────────
def draw_panel(ax: plt.Axes, perf: pd.DataFrame, platform: str, show_legend: bool):
    n_abs = len(ABSTRACTIONS)
    n_sz  = len(SIZES)
    width = 0.20
    gap   = 0.12
    x     = np.arange(n_abs) * (n_sz * width + gap)

    for si, sz in enumerate(SIZES):
        offsets = x + si * width - (n_sz - 1) * width / 2
        for ai, ab in enumerate(ABSTRACTIONS):
            off = offsets[ai]

            # N/A on NVIDIA
            if platform in ("nvidia_rtx5060_laptop",) and ab in NVIDIA_NA:
                ax.text(off, 0.03, "N/A", ha="center", va="bottom",
                        fontsize=6, color="#AAAAAA", rotation=90)
                continue

            row = perf[(perf["platform"] == platform) &
                       (perf["abstraction"] == ab) &
                       (perf["problem_size"] == sz)]
            if row.empty:
                continue
            eff = float(row["efficiency"].iloc[0])

            # Bar color / edge
            is_pattern5 = (ab == "julia_naive" and sz == "large" and eff > 1.0)
            below_thresh = (eff < FLAG_THRESH) and (eff > 0)

            if is_pattern5:
                fc, ec, lw = "#FFD700", "#B8860B", 1.4
            elif below_thresh:
                fc, ec, lw = SIZE_COLORS[sz], "#CC0000", 1.2
            else:
                fc, ec, lw = SIZE_COLORS[sz], "white", 0.5

            ax.bar(off, eff, width=width, color=fc, edgecolor=ec,
                   linewidth=lw, zorder=3,
                   label=SIZE_LABELS[sz] if (ai == 0 and not show_legend) else None)

    # η=1 reference
    ax.axhline(1.0, linestyle="--", color="#333333", linewidth=0.9,
               zorder=4, label="η=1.0 (native)")
    ax.axhline(FLAG_THRESH, linestyle=":", color="#CC0000", linewidth=0.7,
               zorder=4, alpha=0.8)

    ax.set_title(PANEL_TITLES[platform], fontsize=10, pad=6)
    ax.set_xticks(x + (n_sz - 1) * width / 2 - (n_sz - 1) * width / 2)
    ax.set_xticks(x)
    ax.set_xticklabels([ABS_LABELS[a] for a in ABSTRACTIONS],
                       ha="center", fontsize=7.5)
    ax.set_ylabel("Efficiency η (vs native)" if show_legend else "")
    ax.set_ylim(0, max(1.45, ax.get_ylim()[1]))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlim(x[0] - width * 2, x[-1] + width * 2)

    # Annotate key findings
    if platform == "nvidia_rtx5060_laptop":
        # Julia/large: Pattern 5 gold annotation
        ai  = ABSTRACTIONS.index("julia_naive")
        si  = SIZES.index("large")
        off = x[ai] + si * width - (n_sz - 1) * width / 2
        row = perf[(perf["platform"] == platform) &
                   (perf["abstraction"] == "julia_naive") &
                   (perf["problem_size"] == "large")]
        if not row.empty:
            eff = float(row["efficiency"].iloc[0])
            ax.annotate(f"η={eff:.2f}\nPattern 5",
                        xy=(off, eff), xytext=(off + 0.38, eff + 0.07),
                        fontsize=6.5, color="#8B0057", ha="left",
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#B8860B",
                                        lw=0.8, relpos=(0.0, 0.5)))

    if platform == "amd_mi300x":
        # Julia/large: reversal annotation
        ai  = ABSTRACTIONS.index("julia_naive")
        si  = SIZES.index("large")
        off = x[ai] + si * width - (n_sz - 1) * width / 2
        row = perf[(perf["platform"] == platform) &
                   (perf["abstraction"] == "julia_naive") &
                   (perf["problem_size"] == "large")]
        if not row.empty:
            eff = float(row["efficiency"].iloc[0])
            ax.annotate(f"η={eff:.2f}\n↓ reversal",
                        xy=(off, eff), xytext=(off + 0.42, eff + 0.16),
                        fontsize=6.5, color="#CC0000", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#CC0000",
                                        lw=0.8, relpos=(0.0, 0.0)))

        # RAJA/large: consistency annotation
        ai  = ABSTRACTIONS.index("raja_naive")
        si  = SIZES.index("large")
        off = x[ai] + si * width - (n_sz - 1) * width / 2
        row = perf[(perf["platform"] == platform) &
                   (perf["abstraction"] == "raja_naive") &
                   (perf["problem_size"] == "large")]
        if not row.empty:
            eff = float(row["efficiency"].iloc[0])
            ax.annotate(f"η={eff:.2f}\n(consistent\npoor)",
                        xy=(off, eff), xytext=(off + 0.40, eff + 0.17),
                        fontsize=6, color="#666666", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#888888",
                                        lw=0.7, relpos=(0.0, 0.0)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    perf = load()
    print(f"[plot_e2_multiplatform] {len(perf)} non-ceiling rows loaded")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    draw_panel(axes[0], perf, "nvidia_rtx5060_laptop", show_legend=True)
    draw_panel(axes[1], perf, "amd_mi300x",            show_legend=False)

    # Shared legend at bottom
    size_patches = [mpatches.Patch(color=SIZE_COLORS[sz], label=SIZE_LABELS[sz])
                    for sz in SIZES]
    gold_patch   = mpatches.Patch(facecolor="#FFD700", edgecolor="#B8860B",
                                  linewidth=1.4,
                                  label="Pattern 5: η>1 (Layout coalescing, NVIDIA)")
    red_ec_patch = mpatches.Patch(facecolor="#59A14F", edgecolor="#CC0000",
                                  linewidth=1.2,
                                  label="η<0.85 (red border = deep-profiling flag)")
    na_patch     = mpatches.Patch(facecolor="white", edgecolor="#AAAAAA",
                                  linewidth=0.7, label="N/A: not run on this platform")
    dash_line    = mpl.lines.Line2D([], [], color="#333333", linewidth=0.9,
                                    linestyle="--", label="η=1.0 (native baseline)")

    all_handles = size_patches + [gold_patch, red_ec_patch, na_patch, dash_line]
    fig.legend(handles=all_handles, loc="lower center",
               ncol=len(all_handles), fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "E2 DGEMM — Efficiency η by Platform, Abstraction, and Problem Size\n"
        "Left: NVIDIA RTX 5060 Laptop  |  Right: AMD MI300X",
        fontsize=10, y=1.01)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_e2_efficiency_multiplatform.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    kb = os.path.getsize(out) / 1024
    print(f"[plot_e2_multiplatform] Saved → {os.path.relpath(out, REPO_ROOT)}  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
