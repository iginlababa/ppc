#!/usr/bin/env python3
"""
Stage 8: Generate all 21 paper figures.

Figure map (project_spec.md §16.1):
  Fig 1   — Three-dimensional experimental matrix
  Fig 2   — Abstraction spectrum (Native → Julia)
  Fig 3–9 — Per-kernel: throughput vs. abstraction (3 platforms, IQR bars, peak line)
  Fig 10–16 — Per-kernel: efficiency relative to native
  Fig 17  — PPC summary bar chart across all kernels and abstractions
  Fig 18  — Overhead attribution breakdown (flagged configurations)
  Fig 19  — Taxonomy map: failure modes × workload types × platforms
  Fig 20  — Decision framework flowchart
  Fig 21  — Tool validation: predicted vs. actual PPC

Usage:
    python analysis/generate_plots.py \
        --input data/processed/ \
        --output paper/figures/ \
        [--format pdf] [--dpi 300]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

ABSTRACTION_COLORS = {
    "native": "#1f77b4",
    "kokkos": "#ff7f0e",
    "raja":   "#2ca02c",
    "sycl":   "#d62728",
    "julia":  "#9467bd",
}

PLATFORM_MARKERS = {
    "nvidia_a100": "o",
    "amd_mi250x":  "s",
    "intel_pvc":   "^",
}

KERNELS = ["stream", "dgemm", "stencil", "spmv", "sptrsv", "bfs", "nbody"]
ABSTRACTIONS = ["native", "kokkos", "raja", "sycl", "julia"]

PPC_EXCELLENT = 0.80
PPC_ACCEPTABLE = 0.60


def load_data(input_dir: Path) -> dict:
    data = {}
    for name, filename in [
        ("perf",      "performance.csv"),
        ("ppc",       "ppc_results.csv"),
        ("overhead",  "overhead_breakdown.csv"),
        ("roofline",  "roofline.csv"),
    ]:
        path = input_dir / filename
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            print(f"WARNING: {path} not found — skipping dependent figures")
            data[name] = None
    return data


def save(fig, out_dir: Path, name: str, fmt: str = "pdf", dpi: int = 300):
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Figure implementations ────────────────────────────────────────────────────

def fig01_experimental_matrix(out_dir, fmt, dpi):
    """Schematic of the 5 × 3 × 3 × 7 experimental matrix."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "TODO: 3D experimental matrix diagram\n(5 abstractions × 3 platforms × 3 sizes × 7 kernels)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_axis_off()
    ax.set_title("Fig 1: Three-Dimensional Experimental Matrix")
    save(fig, out_dir, "fig01_experimental_matrix", fmt, dpi)


def fig02_abstraction_spectrum(out_dir, fmt, dpi):
    """Abstraction spectrum from Native to Julia."""
    fig, ax = plt.subplots(figsize=(10, 3))
    levels = ["Native\nCUDA/HIP", "SYCL\n(DPC++)", "RAJA\n(loop)", "Kokkos\n(data)", "Julia\n(JIT)"]
    positions = np.linspace(0.1, 0.9, len(levels))
    for pos, label, color in zip(positions, levels, ABSTRACTION_COLORS.values()):
        ax.annotate("", xy=(pos + 0.12, 0.5), xytext=(pos, 0.5),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    xycoords="axes fraction", textcoords="axes fraction")
        ax.text(pos, 0.5, label, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
    ax.text(0.05, 0.1, "← More control / less portable", transform=ax.transAxes, fontsize=8, color="gray")
    ax.text(0.95, 0.1, "More portable / less control →", transform=ax.transAxes, fontsize=8, color="gray", ha="right")
    ax.set_axis_off()
    ax.set_title("Fig 2: Abstraction Spectrum")
    save(fig, out_dir, "fig02_abstraction_spectrum", fmt, dpi)


def fig_throughput_vs_abstraction(df, kernel, fig_num, out_dir, fmt, dpi):
    """Figs 3–9: Throughput per abstraction per platform."""
    if df is None:
        return
    kdf = df[(df["kernel"] == kernel) & (df["problem_size"] == "large")]
    if kdf.empty:
        print(f"  No data for kernel={kernel}")
        return

    platforms = sorted(kdf["platform"].unique())
    fig, axes = plt.subplots(1, len(platforms), figsize=(5 * len(platforms), 4), sharey=True)
    if len(platforms) == 1:
        axes = [axes]

    for ax, platform in zip(axes, platforms):
        pdf = kdf[kdf["platform"] == platform]
        abstractions = [a for a in ABSTRACTIONS if a in pdf["abstraction"].values]
        medians = [pdf[pdf["abstraction"] == a]["throughput"].median() for a in abstractions]
        colors = [ABSTRACTION_COLORS.get(a, "gray") for a in abstractions]
        bars = ax.bar(abstractions, medians, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(platform.replace("_", " ").title())
        ax.set_xlabel("Abstraction")
        ax.set_ylabel("Throughput")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Fig {fig_num}: {kernel.upper()} — Throughput vs. Abstraction (Large Size)")
    plt.tight_layout()
    save(fig, out_dir, f"fig{fig_num:02d}_{kernel}_throughput", fmt, dpi)


def fig_efficiency_vs_native(df, kernel, fig_num, out_dir, fmt, dpi):
    """Figs 10–16: Efficiency relative to native per platform."""
    if df is None:
        return
    # TODO: compute efficiency column if not present
    kdf = df[(df["kernel"] == kernel) & (df["abstraction"] != "native")]
    if kdf.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f"TODO: Efficiency plot for {kernel}\n(requires efficiency column)",
            ha="center", va="center", transform=ax.transAxes)
    ax.axhline(PPC_EXCELLENT, color="green", linestyle="--", label=f"Excellent ({PPC_EXCELLENT})")
    ax.axhline(PPC_ACCEPTABLE, color="orange", linestyle="--", label=f"Acceptable ({PPC_ACCEPTABLE})")
    ax.set_title(f"Fig {fig_num}: {kernel.upper()} — Efficiency Relative to Native")
    ax.legend()
    save(fig, out_dir, f"fig{fig_num:02d}_{kernel}_efficiency", fmt, dpi)


def fig17_ppc_summary(ppc_df, out_dir, fmt, dpi):
    """Fig 17: PPC summary across all kernels and abstractions."""
    if ppc_df is None:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    if ppc_df.empty:
        ax.text(0.5, 0.5, "No PPC data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        pivot = ppc_df.pivot_table(values="ppc", index="abstraction", columns="kernel", aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5)
        ax.axhline(PPC_EXCELLENT, color="white", linewidth=1)
    ax.set_title("Fig 17: PPC Summary — All Kernels × Abstractions")
    save(fig, out_dir, "fig17_ppc_summary", fmt, dpi)


def fig18_overhead_breakdown(overhead_df, out_dir, fmt, dpi):
    """Fig 18: Overhead attribution stacked bar chart."""
    if overhead_df is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, "TODO: Stacked overhead attribution bar chart",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Fig 18: Overhead Attribution Breakdown")
    save(fig, out_dir, "fig18_overhead_breakdown", fmt, dpi)


def fig19_taxonomy_map(out_dir, fmt, dpi):
    """Fig 19: Taxonomy map — failure modes × workload types × platforms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "TODO: Taxonomy heatmap\n(requires complete taxonomy.json)",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Fig 19: Taxonomy Map — Failure Modes × Workloads × Platforms")
    ax.set_axis_off()
    save(fig, out_dir, "fig19_taxonomy_map", fmt, dpi)


def fig20_decision_framework(out_dir, fmt, dpi):
    """Fig 20: Decision framework flowchart."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.text(0.5, 0.5, "TODO: Decision framework flowchart\n(generate from decision_framework.py)",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Fig 20: Decision Framework")
    ax.set_axis_off()
    save(fig, out_dir, "fig20_decision_framework", fmt, dpi)


def fig21_validation(out_dir, fmt, dpi):
    """Fig 21: Tool validation — predicted vs. actual PPC."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect prediction", linewidth=0.8)
    ax.set_xlabel("Predicted PPC")
    ax.set_ylabel("Actual PPC")
    ax.set_title("Fig 21: Abstraction Advisor — Predicted vs. Actual PPC")
    ax.text(0.5, 0.4, "TODO: populate after tool validation (Month 9)",
            ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.legend()
    save(fig, out_dir, "fig21_validation", fmt, dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="data/processed/")
    parser.add_argument("--output", required=True, help="paper/figures/")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.input))
    perf = data["perf"]
    ppc  = data["ppc"]
    overhead = data["overhead"]

    print("Generating figures...")
    fig01_experimental_matrix(out_dir, args.format, args.dpi)
    fig02_abstraction_spectrum(out_dir, args.format, args.dpi)

    for i, kernel in enumerate(KERNELS, start=3):
        fig_throughput_vs_abstraction(perf, kernel, i, out_dir, args.format, args.dpi)

    for i, kernel in enumerate(KERNELS, start=10):
        fig_efficiency_vs_native(perf, kernel, i, out_dir, args.format, args.dpi)

    fig17_ppc_summary(ppc, out_dir, args.format, args.dpi)
    fig18_overhead_breakdown(overhead, out_dir, args.format, args.dpi)
    fig19_taxonomy_map(out_dir, args.format, args.dpi)
    fig20_decision_framework(out_dir, args.format, args.dpi)
    fig21_validation(out_dir, args.format, args.dpi)

    print(f"\nAll figures written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
