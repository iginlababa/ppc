#!/usr/bin/env python3
"""fig6_e2_roofline_comparison.png — 4-panel roofline for E2 deep profiling."""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "pdf.fonttype": 42,
})

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(REPO, "figures", "e2")
os.makedirs(FIGURES, exist_ok=True)

# ── Hardware ceilings ──────────────────────────────────────────────────────────
BW_PEAK_GBS   = 281.0   # GB/s measured E1 peak (12001 MHz GDDR7)
FP64_PEAK     = 254.0   # GFLOP/s: julia_cublas/large (practical cuBLAS ceiling)
RIDGE         = FP64_PEAK / BW_PEAK_GBS   # ~0.904 FLOP/byte

# ── Arithmetic intensity: AI = 2N / (3*8) = N/12 FLOP/byte ───────────────────
AI = {1024: 1024/12.0, 4096: 4096/12.0, 8192: 8192/12.0}
# medium = 341 FLOP/byte, large = 683 FLOP/byte — both compute-bound

# ── Data points ───────────────────────────────────────────────────────────────
# (arithmetic_intensity, gflops, label, color, marker, linestyle)
PANELS = {
    ("raja_naive", "medium"): {
        "ai": AI[4096],
        "points": [
            (AI[4096], 175.1, "native/medium\n(benchmark)", "#648FFF", "s", "-"),
            (AI[4096],  42.0, "raja_naive/medium\n(benchmark, throttled)",  "#FFB000", "v", "--"),
            (AI[4096], 116.0, "raja_naive/medium\n(profiling, fresh)",      "#FF6600", "o", "-"),
        ],
        "title": "RAJA naive / medium (N=4096)",
        "annotation": "P004 API Limitation\n+ P004 Thermal Contamination\n(benchmark ≠ true perf)",
        "ann_color": "#CC4400",
    },
    ("raja_naive", "large"): {
        "ai": AI[8192],
        "points": [
            (AI[8192], 176.3, "native/large\n(benchmark)",   "#648FFF", "s", "-"),
            (AI[8192],  85.1, "raja_naive/large\n(benchmark)", "#FFB000", "o", "-"),
        ],
        "title": "RAJA naive / large (N=8192)",
        "annotation": "P004 API Limitation\n(eff=0.48 vs native;\nno tiling possible in RAJA forall)",
        "ann_color": "#CC4400",
    },
    ("julia_naive", "medium"): {
        "ai": AI[4096],
        "points": [
            (AI[4096], 175.1, "native/medium\n(benchmark)",   "#648FFF", "s", "-"),
            (AI[4096], 139.3, "julia_naive/medium\n(benchmark, throttled)", "#DC267F", "v", "--"),
            (AI[4096], 225.0, "julia_naive/medium\n(profiling, fresh)",     "#FF1493", "o", "-"),
        ],
        "title": "Julia naive / medium (N=4096)",
        "annotation": "P004 Thermal Contamination only\n(profiling: eff=1.29 vs native;\nbenchmark 0.80 is artifact)",
        "ann_color": "#8B0057",
    },
    ("julia_naive", "large"): {
        "ai": AI[8192],
        "points": [
            (AI[8192], 176.3, "native/large\n(benchmark)",    "#648FFF", "s", "-"),
            (AI[8192], 211.0, "julia_naive/large\n(profiling, fresh)",  "#FF1493", "o", "-"),
            (AI[8192], 220.4, "julia_naive/large\n(benchmark, ~consistent)", "#DC267F", "D", "--"),
        ],
        "title": "Julia naive / large (N=8192)",
        "annotation": "Pattern 5: Layout-Induced\nCoalescing Advantage\n(eff=1.20–1.25 vs native)",
        "ann_color": "#8B0057",
    },
}

order = [("raja_naive", "medium"), ("raja_naive", "large"),
         ("julia_naive", "medium"), ("julia_naive", "large")]

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

for ax, key in zip(axes_flat, order):
    info = PANELS[key]
    ai_val = info["ai"]

    # ── Roofline ceiling ────────────────────────────────────────────────────
    ai_range = np.logspace(-1, 4, 500)
    roof = np.minimum(ai_range * BW_PEAK_GBS, FP64_PEAK)
    ax.loglog(ai_range, roof, color="#333333", linewidth=1.0, zorder=1,
              label="Roofline ceiling")
    ax.axhline(FP64_PEAK, color="#333333", linewidth=0.6, linestyle=":",
               alpha=0.5, zorder=1)
    ax.axvline(RIDGE, color="#888888", linewidth=0.6, linestyle=":",
               alpha=0.4, zorder=1)

    # Ceiling labels
    ax.text(ai_range[-1]*0.6, FP64_PEAK*1.05, f"FP64 peak\n{FP64_PEAK:.0f} GFLOP/s",
            fontsize=6, color="#555555", va="bottom", ha="right")
    ax.text(RIDGE*1.05, 3.0, f"Ridge\n{RIDGE:.1f}", fontsize=5.5, color="#888888",
            va="bottom")
    # BW ceiling label (on slope)
    ax.text(0.2, 0.2*BW_PEAK_GBS*0.7, f"BW={BW_PEAK_GBS:.0f} GB/s",
            fontsize=5.5, color="#555555", rotation=45, va="bottom")

    # ── Data points ─────────────────────────────────────────────────────────
    for (ai, gf, lbl, col, mk, ls) in info["points"]:
        ax.scatter([ai], [gf], color=col, marker=mk, s=55, zorder=4,
                   edgecolors="white", linewidths=0.5)
        # Vertical line from point to ceiling
        ax.plot([ai, ai], [gf, min(ai * BW_PEAK_GBS, FP64_PEAK)],
                color=col, linewidth=0.5, linestyle=":", alpha=0.4, zorder=2)
        # Label
        offset_y = gf * 0.78 if gf > 100 else gf * 0.62
        ax.text(ai * 1.06, gf, lbl, fontsize=5.5, color=col, va="center")

    # ── Annotation ──────────────────────────────────────────────────────────
    ax.text(0.04, 0.95, info["annotation"],
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color=info["ann_color"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=info["ann_color"], linewidth=0.6, alpha=0.85))

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_title(info["title"], fontsize=8.5, pad=4)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_xlim(0.12, 2500)
    ax.set_ylim(2, 600)
    ax.grid(True, which="both", linestyle=":", linewidth=0.35, alpha=0.4)

# ── Shared legend ──────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color="#648FFF", label="CUDA native"),
    mpatches.Patch(color="#FFB000", label="RAJA naive"),
    mpatches.Patch(color="#FF6600", label="RAJA naive (profiling, fresh)"),
    mpatches.Patch(color="#DC267F", label="Julia naive"),
    mpatches.Patch(color="#FF1493", label="Julia naive (profiling, fresh)"),
    plt.Line2D([0],[0], color="#333333", linewidth=1.0, label="Roofline ceiling"),
    plt.Line2D([0],[0], marker="v", color="gray", markersize=5, linestyle="None",
               label="Benchmark (thermally contaminated)"),
    plt.Line2D([0],[0], marker="o", color="gray", markersize=5, linestyle="None",
               label="Profiling run (fresh thermal state)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.03))

fig.suptitle("E2 DGEMM — Roofline Analysis\n"
             "ncu hardware counters unavailable (RmProfilingAdminOnly=1); "
             "nsys CUDA trace used for timing; AI computed analytically",
             fontsize=8.5, y=1.01)
fig.tight_layout(rect=[0, 0.04, 1, 1])

out = os.path.join(FIGURES, "fig6_e2_roofline_comparison.png")
fig.savefig(out)
plt.close(fig)
print(f"Saved: {out}  ({os.path.getsize(out)//1024} KB)")
