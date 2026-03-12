"""
Unit tests for analysis/overhead_attribution.py
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis.overhead_attribution import attribute_overhead, _dominant


def make_merged_df(native_total, native_kernel, abs_total, abs_kernel, launch=0.0):
    rows = [
        {
            "kernel": "stream", "abstraction": "native",
            "platform": "nvidia_a100", "problem_size": "large",
            "execution_time_ms": native_total,
            "kernel_time_ms": native_kernel,
            "launch_overhead_ms": 0.0,
            "memory_transfer_mb": 0.0,
        },
        {
            "kernel": "stream", "abstraction": "kokkos",
            "platform": "nvidia_a100", "problem_size": "large",
            "execution_time_ms": abs_total,
            "kernel_time_ms": abs_kernel,
            "launch_overhead_ms": launch,
            "memory_transfer_mb": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_zero_overhead():
    df = make_merged_df(100.0, 100.0, 100.0, 100.0)
    result = attribute_overhead(df)
    assert len(result) == 1
    assert result.iloc[0]["total_overhead_ms"] == pytest.approx(0.0)


def test_host_framework_dominant():
    # Abstraction has extra 10ms entirely in host overhead, no kernel delta
    df = make_merged_df(100.0, 90.0, 110.0, 90.0, launch=0.0)
    result = attribute_overhead(df)
    assert result.iloc[0]["dominant_category"] == "host_framework"


def test_dominant_function():
    assert _dominant(5.0, 1.0, 1.0, 1.0) == "kernel_launch"
    assert _dominant(0.0, 5.0, 0.0, 0.0) == "host_framework"
    assert _dominant(0.0, 0.0, 5.0, 0.0) == "memory_transfer"
    assert _dominant(0.0, 0.0, 0.0, 5.0) == "compiler_quality"
