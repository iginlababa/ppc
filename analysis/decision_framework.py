#!/usr/bin/env python3
"""
Stage 7: Decision framework — rule-based abstraction recommendation engine.

Implements the decision logic from project_spec.md §13.2.
Thresholds are hypothetical until determined from experimental data (§13.3).

Usage:
    python analysis/decision_framework.py \
        --profile workload_profile.json \
        --targets nvidia_a100,amd_mi250x \
        --taxonomy data/taxonomy.json \
        --output recommendation.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Thresholds (§13.3 — update from experimental data) ───────────────────────
REGULARITY_HIGH    = 0.8
REGULARITY_MEDIUM  = 0.5
DIVERGENCE_LOW     = 0.1
DIVERGENCE_MEDIUM  = 0.3
MIN_KERNEL_DURATION_US = 100.0  # µs — below this, launch overhead may dominate
PPC_SUCCESS_THRESHOLD  = 0.80
PPC_CAUTION_THRESHOLD  = 0.60


@dataclass
class WorkloadProfile:
    kernel_name: str
    memory_regularity: float      # 0–1; 1 = perfectly regular unit-stride
    arithmetic_intensity: float   # FLOP/byte
    kernel_duration_us: float     # microseconds
    control_flow_divergence: float  # 0–1; 1 = all warps diverge
    data_structure_type: str      # "dense_array" | "sparse" | "graph" | "dynamic"
    multi_dimensional: bool = False


@dataclass
class Recommendation:
    strategy: str                  # FULL_ABSTRACTION | ABSTRACTION_WITH_TUNING | HYBRID | NATIVE | TASK_RUNTIME
    confidence: str                # HIGH | MEDIUM | LOW
    rationale: str
    suggested_abstractions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    known_patterns: list[str] = field(default_factory=list)
    expected_ppc_range: Optional[tuple[float, float]] = None


def recommend(profile: WorkloadProfile, targets: list[str], taxonomy: dict) -> Recommendation:
    """
    Core decision logic. See project_spec.md §13.2.
    """
    r = profile.memory_regularity
    d = profile.control_flow_divergence
    dur = profile.kernel_duration_us
    warnings = []
    patterns = []

    # ── Regular workloads ────────────────────────────────────────────────────
    if r > REGULARITY_HIGH and d < DIVERGENCE_LOW:
        if dur > MIN_KERNEL_DURATION_US:
            return Recommendation(
                strategy="FULL_ABSTRACTION",
                confidence="HIGH",
                rationale=(
                    f"Regular memory access (regularity={r:.2f}) and low divergence "
                    f"({d:.2f}), kernel duration {dur:.0f} µs > {MIN_KERNEL_DURATION_US} µs. "
                    "Abstractions likely to achieve PPC > 0.80."
                ),
                suggested_abstractions=["kokkos", "raja", "sycl"],
                expected_ppc_range=(0.80, 1.0),
            )
        else:
            return Recommendation(
                strategy="CAUTION",
                confidence="MEDIUM",
                rationale=(
                    f"Regular workload but kernel duration {dur:.0f} µs < {MIN_KERNEL_DURATION_US} µs. "
                    "Launch overhead may dominate. Profile first."
                ),
                suggested_abstractions=["native"],
                warnings=["Kernel too short — launch overhead may dominate (Pattern P001)"],
                known_patterns=["P001"],
                expected_ppc_range=(0.40, 0.70),
            )

    # ── Semi-irregular workloads ──────────────────────────────────────────────
    if r > REGULARITY_MEDIUM and d < DIVERGENCE_MEDIUM:
        return Recommendation(
            strategy="HYBRID",
            confidence="MEDIUM",
            rationale=(
                f"Semi-irregular access (regularity={r:.2f}, divergence={d:.2f}). "
                "Use abstraction for data management but native for hot loops."
            ),
            suggested_abstractions=["kokkos"],
            warnings=["Hot loop kernels should use native CUDA/HIP for performance-critical paths"],
            expected_ppc_range=(0.60, 0.80),
        )

    # ── Irregular workloads ───────────────────────────────────────────────────
    if profile.data_structure_type in ("graph", "dynamic"):
        # Check if task runtime is applicable
        task_applicable = profile.data_structure_type == "graph"
        if task_applicable:
            return Recommendation(
                strategy="TASK_RUNTIME",
                confidence="MEDIUM",
                rationale=(
                    f"Highly irregular workload (regularity={r:.2f}, divergence={d:.2f}). "
                    "Consider PaRSEC/Legion for load balancing."
                ),
                suggested_abstractions=["native"],
                warnings=["Abstractions expected to fail — PPC likely < 0.60"],
                expected_ppc_range=(0.30, 0.60),
            )

    return Recommendation(
        strategy="NATIVE",
        confidence="HIGH",
        rationale=(
            f"Irregular workload (regularity={r:.2f}, divergence={d:.2f}). "
            "Platform-specific native implementation recommended."
        ),
        suggested_abstractions=["native"],
        warnings=["PPC will vary significantly across platforms (>0.30 variance expected)"],
        expected_ppc_range=(0.20, 0.50),
    )


def load_profile(path: Path) -> WorkloadProfile:
    with open(path) as f:
        data = json.load(f)
    return WorkloadProfile(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="workload_profile.json")
    parser.add_argument("--targets", default="nvidia_a100,amd_mi250x,intel_pvc",
                        help="Comma-separated target platforms")
    parser.add_argument("--taxonomy", default="data/taxonomy.json")
    parser.add_argument("--output", help="Write recommendation to JSON file")
    args = parser.parse_args()

    profile = load_profile(Path(args.profile))
    targets = [t.strip() for t in args.targets.split(",")]

    taxonomy = {}
    tax_path = Path(args.taxonomy)
    if tax_path.exists():
        with open(tax_path) as f:
            taxonomy = json.load(f)

    rec = recommend(profile, targets, taxonomy)

    output = {
        "kernel": profile.kernel_name,
        "targets": targets,
        "strategy": rec.strategy,
        "confidence": rec.confidence,
        "rationale": rec.rationale,
        "suggested_abstractions": rec.suggested_abstractions,
        "warnings": rec.warnings,
        "known_patterns": rec.known_patterns,
        "expected_ppc_range": rec.expected_ppc_range,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Recommendation written to {out_path}")
    else:
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
