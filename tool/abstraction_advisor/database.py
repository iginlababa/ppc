"""
Taxonomy Database — Component 2 of abstraction-advisor.

Provides query interface over data/taxonomy.json.
Returns: expected PPC, known issues, mitigation strategies.
"""

import json
from pathlib import Path
from typing import Optional


DEFAULT_TAXONOMY = Path("data/taxonomy.json")


class TaxonomyDatabase:
    def __init__(self, taxonomy_path: Path = DEFAULT_TAXONOMY):
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")
        with open(taxonomy_path) as f:
            self._data = json.load(f)
        self._patterns = self._data.get("patterns", [])
        self._success = self._data.get("success_patterns", [])

    def query_failures(
        self,
        kernel: Optional[str] = None,
        abstraction: Optional[str] = None,
        platform: Optional[str] = None,
        data_structure_type: Optional[str] = None,
    ) -> list[dict]:
        results = []
        for p in self._patterns:
            if p.get("type") != "failure":
                continue
            if p.get("status") == "hypothesis":
                continue  # only return validated patterns
            if kernel and kernel not in p.get("affected_workloads", []):
                continue
            if platform:
                affected = p.get("affected_platforms", ["all"])
                if "all" not in affected and platform not in affected:
                    continue
            results.append(p)
        return results

    def query_successes(
        self,
        kernel: Optional[str] = None,
        abstraction: Optional[str] = None,
    ) -> list[dict]:
        return [
            p for p in self._success
            if p.get("status") != "hypothesis"
        ]

    def get_pattern(self, pattern_id: str) -> Optional[dict]:
        for p in self._patterns + self._success:
            if p["id"] == pattern_id:
                return p
        return None

    def all_patterns(self, include_hypotheses: bool = False) -> list[dict]:
        all_p = self._patterns + self._success
        if include_hypotheses:
            return all_p
        return [p for p in all_p if p.get("status") != "hypothesis"]

    def summary(self) -> dict:
        return {
            "total_patterns": len(self._patterns) + len(self._success),
            "failure_patterns": len(self._patterns),
            "success_patterns": len(self._success),
            "validated": sum(
                1 for p in self._patterns + self._success
                if p.get("status") != "hypothesis"
            ),
            "hypotheses": sum(
                1 for p in self._patterns + self._success
                if p.get("status") == "hypothesis"
            ),
        }
