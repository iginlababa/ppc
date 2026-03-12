"""
Workload Profiler — Component 1 of abstraction-advisor.

Extracts workload characteristics from profiler output and produces
a workload_profile.json for consumption by the recommendation engine.

Supported profiler backends:
  - nsys + ncu (NVIDIA)
  - rocprof + omniperf (AMD)
  - vtune (Intel)

Output format (workload_profile.json):
  {
    "kernel_name": "stream_triad",
    "memory_regularity": 0.95,
    "arithmetic_intensity": 0.25,
    "kernel_duration_us": 450.0,
    "control_flow_divergence": 0.02,
    "data_structure_type": "dense_array",
    "multi_dimensional": false
  }
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WorkloadProfile:
    kernel_name: str
    memory_regularity: float        # 0–1
    arithmetic_intensity: float     # FLOP/byte
    kernel_duration_us: float       # µs
    control_flow_divergence: float  # 0–1
    data_structure_type: str
    multi_dimensional: bool = False
    source_profiler: str = "unknown"
    platform: str = "unknown"


def from_ncu_csv(csv_path: Path, kernel_name: str = "unknown") -> WorkloadProfile:
    """
    Extract workload characteristics from an ncu CSV export.
    Column names vary by CUDA version — adjust regex accordingly.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # TODO: extract actual metrics from ncu CSV columns
    # This is a placeholder — fill in after collecting first ncu reports

    def get_metric(df, pattern: str, default: float) -> float:
        cols = [c for c in df.columns if re.search(pattern, c, re.IGNORECASE)]
        if cols and not df[cols[0]].empty:
            return float(df[cols[0]].iloc[0])
        return default

    duration_us = get_metric(df, r"duration", 0.0) / 1000.0  # ns → µs
    divergence  = get_metric(df, r"branch.*uniform|divergen", 0.0) / 100.0
    # Memory regularity: approximate from L1/L2 hit rates
    l2_hit      = get_metric(df, r"l2.*hit", 50.0) / 100.0
    regularity  = min(1.0, l2_hit * 1.2)  # heuristic

    return WorkloadProfile(
        kernel_name=kernel_name,
        memory_regularity=round(regularity, 3),
        arithmetic_intensity=0.0,   # TODO: compute from FLOP counters
        kernel_duration_us=round(duration_us, 2),
        control_flow_divergence=round(divergence, 3),
        data_structure_type="unknown",
        source_profiler="ncu",
    )


def from_manual(
    kernel_name: str,
    memory_regularity: float,
    arithmetic_intensity: float,
    kernel_duration_us: float,
    control_flow_divergence: float,
    data_structure_type: str = "dense_array",
    multi_dimensional: bool = False,
) -> WorkloadProfile:
    """Create a profile from manually measured/estimated values."""
    return WorkloadProfile(
        kernel_name=kernel_name,
        memory_regularity=memory_regularity,
        arithmetic_intensity=arithmetic_intensity,
        kernel_duration_us=kernel_duration_us,
        control_flow_divergence=control_flow_divergence,
        data_structure_type=data_structure_type,
        multi_dimensional=multi_dimensional,
        source_profiler="manual",
    )


def save(profile: WorkloadProfile, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    print(f"Workload profile written to {output_path}")


def load(path: Path) -> WorkloadProfile:
    with open(path) as f:
        data = json.load(f)
    return WorkloadProfile(**data)
