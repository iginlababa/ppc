"""
Unit tests for analysis/compute_ppc.py

Tests cover:
  - PPC formula correctness against worked example from project_spec.md Appendix A
  - Efficiency computation
  - PPC tier classification
  - Profiling flag trigger (< 0.85 * native)
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis.compute_ppc import compute_medians, compute_efficiency, compute_ppc, _ppc_tier, flag_for_profiling


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_perf_df():
    """30 runs per configuration for 2 abstractions x 2 platforms."""
    rows = []
    # Native: 1500 GB/s A100, 1700 GB/s MI250X, 1100 GB/s PVC
    for platform, tp in [("nvidia_a100", 1500.0), ("amd_mi250x", 1700.0), ("intel_pvc", 1100.0)]:
        for run_id in range(1, 31):
            rows.append({
                "kernel": "stream", "abstraction": "native",
                "platform": platform, "problem_size": "large",
                "run_id": run_id, "throughput": tp,
            })
    # Kokkos: 1450, 1650, 980 (from Appendix A)
    for platform, tp in [("nvidia_a100", 1450.0), ("amd_mi250x", 1650.0), ("intel_pvc", 980.0)]:
        for run_id in range(1, 31):
            rows.append({
                "kernel": "stream", "abstraction": "kokkos",
                "platform": platform, "problem_size": "large",
                "run_id": run_id, "throughput": tp,
            })
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ppc_appendix_a(sample_perf_df):
    """
    Reproduce the worked example from project_spec.md Appendix A.
    Expected PPC = 0.942 (±0.001 for floating point).
    """
    agg = compute_medians(sample_perf_df)
    eff = compute_efficiency(agg)
    ppc_df = compute_ppc(eff)

    kokkos_ppc = ppc_df[
        (ppc_df["kernel"] == "stream") &
        (ppc_df["abstraction"] == "kokkos") &
        (ppc_df["problem_size"] == "large")
    ]["ppc"].iloc[0]

    # project_spec.md Appendix A: PPC = 3 / (1/0.967 + 1/0.971 + 1/0.891) ≈ 0.942
    assert abs(kokkos_ppc - 0.942) < 0.002, f"Expected ~0.942, got {kokkos_ppc}"


def test_efficiency_relative_to_native(sample_perf_df):
    agg = compute_medians(sample_perf_df)
    eff = compute_efficiency(agg)

    kokkos_a100 = eff[
        (eff["abstraction"] == "kokkos") &
        (eff["platform"] == "nvidia_a100") &
        (eff["problem_size"] == "large")
    ]["efficiency"].iloc[0]

    assert abs(kokkos_a100 - (1450.0 / 1500.0)) < 1e-6


def test_ppc_tier():
    assert _ppc_tier(0.95) == "excellent"
    assert _ppc_tier(0.80) == "excellent"
    assert _ppc_tier(0.79) == "acceptable"
    assert _ppc_tier(0.60) == "acceptable"
    assert _ppc_tier(0.59) == "poor"
    assert _ppc_tier(0.0)  == "poor"


def test_flag_for_profiling(sample_perf_df):
    """Configurations with efficiency < 0.85 should be flagged."""
    agg = compute_medians(sample_perf_df)
    eff = compute_efficiency(agg)
    flagged = flag_for_profiling(eff)

    # PVC Kokkos: 980/1100 = 0.891 < 0.85? 0.891 > 0.85, should NOT be flagged
    # Adjust: use a clearly below-threshold value
    eff.loc[
        (eff["abstraction"] == "kokkos") & (eff["platform"] == "intel_pvc"),
        "efficiency"
    ] = 0.80

    flagged = flag_for_profiling(eff)
    assert len(flagged) > 0
    assert all(flagged["efficiency"] < 0.85)
    assert "native" not in flagged["abstraction"].values


def test_native_not_in_ppc_output(sample_perf_df):
    """Native should never appear in PPC results (it is the reference)."""
    agg = compute_medians(sample_perf_df)
    eff = compute_efficiency(agg)
    ppc_df = compute_ppc(eff)
    assert "native" not in ppc_df["abstraction"].values


def test_perfect_ppc_is_one():
    """If abstraction matches native on all platforms, PPC should be 1.0."""
    rows = []
    for platform, tp in [("nvidia_a100", 1000.0), ("amd_mi250x", 2000.0), ("intel_pvc", 1500.0)]:
        for abs_name in ["native", "kokkos"]:
            for run_id in range(1, 31):
                rows.append({
                    "kernel": "stream", "abstraction": abs_name,
                    "platform": platform, "problem_size": "large",
                    "run_id": run_id, "throughput": tp,
                })
    df = pd.DataFrame(rows)
    agg = compute_medians(df)
    eff = compute_efficiency(agg)
    ppc_df = compute_ppc(eff)
    ppc_val = ppc_df[ppc_df["abstraction"] == "kokkos"]["ppc"].iloc[0]
    assert abs(ppc_val - 1.0) < 1e-6
