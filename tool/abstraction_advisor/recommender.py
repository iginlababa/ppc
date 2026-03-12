"""
Recommendation Engine — Component 3 of abstraction-advisor.

Wraps analysis/decision_framework.py logic and enriches the output
with taxonomy database lookups.
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from decision_framework import recommend, WorkloadProfile, Recommendation
from .database import TaxonomyDatabase
from .profiler import WorkloadProfile as ProfilerWorkloadProfile


def get_recommendation(
    profile: ProfilerWorkloadProfile,
    targets: list[str],
    taxonomy_path: Path = Path("data/taxonomy.json"),
) -> dict:
    """
    Full recommendation pipeline:
    1. Run decision logic
    2. Enrich with known taxonomy patterns
    3. Return structured output
    """
    # Convert profiler profile to analysis profile
    analysis_profile = WorkloadProfile(
        kernel_name=profile.kernel_name,
        memory_regularity=profile.memory_regularity,
        arithmetic_intensity=profile.arithmetic_intensity,
        kernel_duration_us=profile.kernel_duration_us,
        control_flow_divergence=profile.control_flow_divergence,
        data_structure_type=profile.data_structure_type,
        multi_dimensional=profile.multi_dimensional,
    )

    taxonomy = {}
    if taxonomy_path.exists():
        import json
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

    rec = recommend(analysis_profile, targets, taxonomy)

    # Enrich with taxonomy database lookups
    known_patterns = []
    if taxonomy_path.exists():
        try:
            db = TaxonomyDatabase(taxonomy_path)
            for target in targets:
                failures = db.query_failures(platform=target)
                known_patterns.extend([p["id"] for p in failures])
        except Exception:
            pass

    return {
        "kernel": profile.kernel_name,
        "targets": targets,
        "strategy": rec.strategy,
        "confidence": rec.confidence,
        "rationale": rec.rationale,
        "suggested_abstractions": rec.suggested_abstractions,
        "warnings": rec.warnings,
        "known_patterns": list(set(rec.known_patterns + known_patterns)),
        "expected_ppc_range": rec.expected_ppc_range,
        "profile_summary": {
            "memory_regularity": profile.memory_regularity,
            "arithmetic_intensity": profile.arithmetic_intensity,
            "kernel_duration_us": profile.kernel_duration_us,
            "control_flow_divergence": profile.control_flow_divergence,
        },
    }
