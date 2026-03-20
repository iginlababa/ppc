#!/usr/bin/env python3
"""
E1 STREAM Triad — publication-quality figures (Nature/IEEE style).

Generates 6 figures and saves to figures/:
  fig1_e1_bw_stability.png     — BW vs run_id line plot (all abstractions)
  fig2_e1_median_bw.png        — Median BW bar chart with IQR error bars
  fig3_e1_ppc.png              — PPC bar chart with threshold lines
  fig4_e1_roofline.png         — % of measured native peak (4 bars)
  fig5_e1_cv.png               — CV% stability comparison
  fig6_e1_hw_state_timeline.png — hw_state scatter per abstraction
"""

import glob
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
    "lines.linewidth":   1.2,
    "patch.linewidth":   0.8,
    "pdf.fonttype":      42,   # embed fonts
    "ps.fonttype":       42,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(REPO_ROOT, "data", "raw")
DATA_PROC  = os.path.join(REPO_ROOT, "data", "processed")
FIGURES    = os.path.join(REPO_ROOT, "figures", "e1")
os.makedirs(FIGURES, exist_ok=True)

PLATFORM     = "nvidia_rtx5060_laptop_locked"
PEAK_BW_GBS  = 384.0   # GDDR7 12001 MHz × 128-bit × 2 theoretical peak

# ── Abstraction ordering and colors ───────────────────────────────────────────
ABSTRACTIONS = ["native", "kokkos", "raja", "julia", "numba"]

# IBM colorblind-safe palette
COLORS = {
    "native": "#648FFF",   # blue
    "kokkos": "#785EF0",   # violet
    "raja":   "#FFB000",   # amber
    "julia":  "#DC267F",   # magenta
    "numba":  "#FE6100",   # orange
}

LABELS = {
    "native": "CUDA (native)",
    "kokkos": "Kokkos",
    "raja":   "RAJA",
    "julia":  "Julia/CUDA.jl",
    "numba":  "Numba",
}

# ── AMD-specific constants ─────────────────────────────────────────────────────
AMD_ABSTRACTIONS = ["native", "kokkos", "raja", "julia"]
AMD_LABELS = {
    "native": "HIP (native)",
    "kokkos": "Kokkos",
    "raja":   "RAJA",
    "julia":  "Julia/AMDGPU.jl",
}
SIZE_ORDER  = ["small", "medium", "large"]
SIZE_LABELS = {"small": "Small\n(2²⁰)", "medium": "Medium\n(2²⁴)", "large": "Large\n(2²⁸)"}

# Platform colors for cross-platform figure (brand colors, colorblind-distinct)
PLATFORM_COLORS = {
    "nvidia_rtx5060_laptop_locked": "#76b900",  # NVIDIA green
    "amd_mi300x":                   "#ED1C24",  # AMD red
}
PLATFORM_LABELS = {
    "nvidia_rtx5060_laptop_locked": "NVIDIA RTX 5060 (GDDR7)",
    "amd_mi300x":                   "AMD MI300X (HBM3)",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_summary() -> pd.DataFrame:
    path = os.path.join(DATA_PROC, "e1_stream_summary.csv")
    df = pd.read_csv(path)
    # Enforce display order
    order = {a: i for i, a in enumerate(ABSTRACTIONS)}
    df["_ord"] = df["abstraction"].map(order)
    return df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def load_raw() -> pd.DataFrame:
    frames = []
    for abs_name in ABSTRACTIONS:
        pattern = os.path.join(DATA_RAW, f"stream_{abs_name}_{PLATFORM}_*.csv")
        files = sorted(glob.glob(pattern))
        for f in files:
            chunk = pd.read_csv(f)
            frames.append(chunk)
    raw = pd.concat(frames, ignore_index=True)
    return raw


# ── Helper: save figure ───────────────────────────────────────────────────────
def savefig(fig, name: str):
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Fig 1: BW stability line plot ─────────────────────────────────────────────
def fig1_stability(raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for abs_name in ABSTRACTIONS:
        sub = raw[raw["abstraction"] == abs_name].sort_values("run_id")
        color = COLORS[abs_name]
        label = LABELS[abs_name]

        clean   = sub[sub["hardware_state_verified"] == 1]
        flagged = sub[sub["hardware_state_verified"] == 0]

        ax.plot(sub["run_id"], sub["throughput"],
                color=color, linewidth=1.2, label=label, zorder=2)
        if not flagged.empty:
            ax.scatter(flagged["run_id"], flagged["throughput"],
                       color="crimson", marker="x", s=22, linewidths=1.0,
                       zorder=3, label=None)

    # Legend entry for flagged marker
    ax.scatter([], [], color="crimson", marker="x", s=22,
               linewidths=1.0, label="hw_state=0 (thermal outlier)")

    ax.set_xlabel("Run ID")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("E1 STREAM Triad — bandwidth stability per run\n"
                 f"Platform: {PLATFORM}, problem size: large (2²⁸ × 8 B × 3 = 6.4 GB)")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    ax.set_xlim(0, 31)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig1_e1_bw_stability.png")


# ── Fig 2: Median BW bar chart ────────────────────────────────────────────────
def fig2_median_bw(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    native_median = float(summary.loc[summary["abstraction"] == "native",
                                      "median_bw_gbs"].iloc[0])
    x = np.arange(len(ABSTRACTIONS))
    width = 0.60

    for i, abs_name in enumerate(ABSTRACTIONS):
        row = summary[summary["abstraction"] == abs_name].iloc[0]
        med  = row["median_bw_gbs"]
        q1   = row["q1_bw_gbs"]
        q3   = row["q3_bw_gbs"]
        bar = ax.bar(i, med, width,
                     color=COLORS[abs_name], edgecolor="white",
                     linewidth=0.6, zorder=2)
        # IQR error bar (asymmetric: med-q1 below, q3-med above)
        ax.errorbar(i, med, yerr=[[med - q1], [q3 - med]],
                    fmt="none", color="black", capsize=3,
                    capthick=0.8, elinewidth=0.8, zorder=3)

    # Native baseline dashed line
    ax.axhline(native_median, linestyle="--", color="black",
               linewidth=0.9, zorder=1, label=f"Native ({native_median:.1f} GB/s)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ABSTRACTIONS], rotation=15, ha="right")
    ax.set_ylabel("Median throughput (GB/s)")
    ax.set_title("E1 STREAM Triad — median bandwidth\n(error bars: IQR)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, summary["max_bw_gbs"].max() * 1.12)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig2_e1_median_bw.png")


# ── Fig 3: PPC bar chart ──────────────────────────────────────────────────────
def fig3_ppc(summary: pd.DataFrame):
    non_native = [a for a in ABSTRACTIONS if a != "native"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    x = np.arange(len(non_native))
    width = 0.55

    for i, abs_name in enumerate(non_native):
        row = summary[summary["abstraction"] == abs_name].iloc[0]
        ppc = row["ppc"]
        ax.bar(i, ppc, width,
               color=COLORS[abs_name], edgecolor="white",
               linewidth=0.6, zorder=2)
        ax.text(i, ppc + 0.004, f"{ppc:.4f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Threshold lines
    ax.axhline(0.90, linestyle="--", color="#555555", linewidth=0.8, zorder=1)
    ax.axhline(0.70, linestyle=":",  color="#888888", linewidth=0.8, zorder=1)
    ax.text(len(non_native) - 0.05, 0.904, "0.90",
            ha="right", va="bottom", fontsize=7, color="#555555")
    ax.text(len(non_native) - 0.05, 0.704, "0.70",
            ha="right", va="bottom", fontsize=7, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in non_native], rotation=15, ha="right")
    ax.set_ylabel("PPC (median BW / native median BW)")
    ax.set_title("E1 STREAM Triad — PPC\n(relative to CUDA native baseline)")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig3_e1_ppc.png")


# ── Fig 4: Roofline efficiency ────────────────────────────────────────────────
def fig4_roofline(summary: pd.DataFrame):
    # Use the native median as the effective measured peak bandwidth
    native_median = float(summary.loc[summary["abstraction"] == "native",
                                      "median_bw_gbs"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(ABSTRACTIONS))
    width = 0.60

    for i, abs_name in enumerate(ABSTRACTIONS):
        row  = summary[summary["abstraction"] == abs_name].iloc[0]
        pct  = 100.0 * row["median_bw_gbs"] / native_median
        bar  = ax.bar(i, pct, width,
                      color=COLORS[abs_name], edgecolor="white",
                      linewidth=0.6, zorder=2)
        ax.text(i, pct + 0.4, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7.5)

    ax.axhline(100.0, linestyle="--", color="black", linewidth=0.9, zorder=1,
               label="Native peak (100%)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ABSTRACTIONS], rotation=15, ha="right")
    ax.set_ylabel("Efficiency (% of native peak BW)")
    ax.set_title(f"E1 STREAM Triad — roofline efficiency\n"
                 f"Native peak: {native_median:.1f} GB/s "
                 f"(GDDR7 spec: {PEAK_BW_GBS:.0f} GB/s)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig4_e1_roofline.png")


# ── Fig 5: CV% stability ──────────────────────────────────────────────────────
def fig5_cv(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(ABSTRACTIONS))
    width = 0.60

    for i, abs_name in enumerate(ABSTRACTIONS):
        row = summary[summary["abstraction"] == abs_name].iloc[0]
        cv  = row["cv_pct"]
        bar = ax.bar(i, cv, width,
                     color=COLORS[abs_name], edgecolor="white",
                     linewidth=0.6, zorder=2)
        label_y = cv + 0.08
        ax.text(i, label_y, f"{cv:.2f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ABSTRACTIONS], rotation=15, ha="right")
    ax.set_ylabel("CV (%) — lower is better")
    ax.set_title("E1 STREAM Triad — run-to-run stability\n"
                 "(coefficient of variation on clean runs)")
    ax.set_ylim(0, summary["cv_pct"].max() * 1.35)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig5_e1_cv.png")


# ── Fig 6: hw_state timeline ──────────────────────────────────────────────────
def fig6_hw_state(raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    y_pos = {a: i for i, a in enumerate(ABSTRACTIONS)}
    y_labels = [LABELS[a] for a in ABSTRACTIONS]

    for abs_name in ABSTRACTIONS:
        sub = raw[raw["abstraction"] == abs_name].sort_values("run_id")
        y   = y_pos[abs_name]

        clean   = sub[sub["hardware_state_verified"] == 1]
        flagged = sub[sub["hardware_state_verified"] == 0]

        ax.scatter(clean["run_id"],   [y] * len(clean),
                   color=COLORS[abs_name], s=18, marker="o", zorder=2,
                   label="hw_state=1 (clean)" if abs_name == ABSTRACTIONS[0] else None)
        if not flagged.empty:
            ax.scatter(flagged["run_id"], [y] * len(flagged),
                       color="crimson", s=22, marker="x", linewidths=1.0, zorder=3,
                       label="hw_state=0 (outlier)" if abs_name == ABSTRACTIONS[0] else None)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Run ID")
    ax.set_title("E1 STREAM Triad — hw_state timeline\n"
                 "(red ✕ = thermal outlier flagged by run_stream.sh)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 31)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig6_e1_hw_state_timeline.png")


# ── Fig 7: AMD MI300X — bandwidth vs problem size ─────────────────────────────
def fig7_amd_size_sensitivity(summary: pd.DataFrame):
    amd = summary[summary["platform"] == "amd_mi300x"].copy()

    n_sizes = len(SIZE_ORDER)
    n_abs   = len(AMD_ABSTRACTIONS)
    width   = 0.18
    x       = np.arange(n_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2),
                             gridspec_kw={"wspace": 0.35})

    # ── left panel: absolute bandwidth ────────────────────────────────────────
    ax = axes[0]
    for i, abs_name in enumerate(AMD_ABSTRACTIONS):
        offsets = x + (i - (n_abs - 1) / 2) * width
        vals = [
            float(amd.loc[(amd["abstraction"] == abs_name) &
                           (amd["problem_size"] == sz), "median_bw_gbs"].iloc[0])
            if not amd.loc[(amd["abstraction"] == abs_name) &
                            (amd["problem_size"] == sz)].empty else 0.0
            for sz in SIZE_ORDER
        ]
        ax.bar(offsets, vals, width,
               color=COLORS[abs_name], edgecolor="white",
               linewidth=0.6, label=AMD_LABELS[abs_name], zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("Median throughput (GB/s)")
    ax.set_title("Absolute bandwidth")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # ── right panel: PPC relative to native at each size ──────────────────────
    ax = axes[1]
    for i, abs_name in enumerate(AMD_ABSTRACTIONS):
        offsets = x + (i - (n_abs - 1) / 2) * width
        vals = [
            float(amd.loc[(amd["abstraction"] == abs_name) &
                           (amd["problem_size"] == sz), "ppc"].iloc[0])
            if not amd.loc[(amd["abstraction"] == abs_name) &
                            (amd["problem_size"] == sz)].empty else 0.0
            for sz in SIZE_ORDER
        ]
        ax.bar(offsets, vals, width,
               color=COLORS[abs_name], edgecolor="white",
               linewidth=0.6, zorder=2)

    ax.axhline(0.90, linestyle="--", color="#555555", linewidth=0.8, zorder=1)
    ax.axhline(0.70, linestyle=":",  color="#888888", linewidth=0.8, zorder=1)
    ax.text(n_sizes - 0.05, 0.904, "0.90",
            ha="right", va="bottom", fontsize=7, color="#555555")
    ax.text(n_sizes - 0.05, 0.704, "0.70",
            ha="right", va="bottom", fontsize=7, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("PPC (relative to native at same size)")
    ax.set_title("Relative efficiency (PPC)")
    ax.set_ylim(0, 1.20)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle("E1 STREAM Triad — AMD MI300X: bandwidth vs problem size",
                 fontsize=10, y=1.01)

    savefig(fig, "fig7_e1_amd_size_sensitivity.png")


# ── Fig 8: Cross-platform PPC (large problem size) ────────────────────────────
def fig8_crossplatform_ppc(summary: pd.DataFrame):
    # Large problem size only; abstractions present on both platforms
    common_abs = AMD_ABSTRACTIONS  # native, kokkos, raja, julia
    platforms  = ["nvidia_rtx5060_laptop_locked", "amd_mi300x"]

    large = summary[summary["problem_size"] == "large"].copy()

    n_abs    = len(common_abs)
    n_plat   = len(platforms)
    width    = 0.30
    x        = np.arange(n_abs)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for j, platform in enumerate(platforms):
        offsets = x + (j - (n_plat - 1) / 2) * width
        vals = [
            float(large.loc[(large["platform"] == platform) &
                             (large["abstraction"] == abs_name), "ppc"].iloc[0])
            if not large.loc[(large["platform"] == platform) &
                              (large["abstraction"] == abs_name)].empty else 0.0
            for abs_name in common_abs
        ]
        hatch = None if j == 0 else "////"
        ax.bar(offsets, vals, width,
               color=PLATFORM_COLORS[platform],
               edgecolor="white", linewidth=0.6,
               hatch=hatch, label=PLATFORM_LABELS[platform], zorder=2)
        for k, v in enumerate(vals):
            ax.text(offsets[k], v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5)

    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8, zorder=1,
               label="Native baseline (PPC = 1.0)")
    ax.axhline(0.90, linestyle=":",  color="#555555", linewidth=0.7, zorder=1)
    ax.text(n_abs - 0.05, 0.904, "0.90",
            ha="right", va="bottom", fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([AMD_LABELS[a] for a in common_abs], rotation=12, ha="right")
    ax.set_ylabel("PPC (median BW / native median BW)")
    ax.set_title("E1 STREAM Triad — cross-platform PPC\n"
                 "(large problem size: 2²⁸ elements, hw_state=1 clean runs)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_ylim(0, 1.22)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    savefig(fig, "fig8_e1_crossplatform_ppc.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[plot_e1] Loading data ...")
    summary      = load_summary()
    nvidia_summ  = summary[summary["platform"] == PLATFORM].copy()
    raw          = load_raw()

    print("[plot_e1] Generating figures ...")
    fig1_stability(raw)
    fig2_median_bw(nvidia_summ)
    fig3_ppc(nvidia_summ)
    fig4_roofline(nvidia_summ)
    fig5_cv(nvidia_summ)
    fig6_hw_state(raw)
    fig7_amd_size_sensitivity(summary)
    fig8_crossplatform_ppc(summary)

    print(f"[plot_e1] All 8 figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
