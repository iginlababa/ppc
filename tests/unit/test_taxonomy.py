"""
Unit tests for the taxonomy database (tool/abstraction_advisor/database.py)
and taxonomy.json schema.
"""

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TAXONOMY_PATH = Path("data/taxonomy.json")


@pytest.fixture
def taxonomy_data():
    if not TAXONOMY_PATH.exists():
        pytest.skip("data/taxonomy.json not found")
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def test_taxonomy_schema_version(taxonomy_data):
    assert "_schema_version" in taxonomy_data


def test_patterns_key_present(taxonomy_data):
    assert "patterns" in taxonomy_data
    assert isinstance(taxonomy_data["patterns"], list)


def test_each_pattern_has_required_fields(taxonomy_data):
    required = {"id", "type", "name", "symptom", "root_cause_category",
                "affected_workloads", "affected_platforms", "mitigation", "evidence"}
    for p in taxonomy_data["patterns"]:
        missing = required - set(p.keys())
        assert not missing, f"Pattern {p.get('id', '?')} missing: {missing}"


def test_pattern_type_valid(taxonomy_data):
    for p in taxonomy_data["patterns"]:
        assert p["type"] in ("failure", "success"), f"Pattern {p['id']} has invalid type"


def test_pattern_ids_unique(taxonomy_data):
    ids = [p["id"] for p in taxonomy_data["patterns"]]
    assert len(ids) == len(set(ids)), "Duplicate pattern IDs found"


def test_root_cause_categories_valid(taxonomy_data):
    valid_categories = {
        "Compiler Backend Failure",
        "Runtime Coordination Overhead",
        "Memory Model Mismatch",
        "API Limitation",
    }
    for p in taxonomy_data["patterns"]:
        cat = p.get("root_cause_category", "")
        assert cat in valid_categories, (
            f"Pattern {p['id']} has invalid root_cause_category: '{cat}'"
        )


def test_database_loads():
    if not TAXONOMY_PATH.exists():
        pytest.skip("data/taxonomy.json not found")
    from tool.abstraction_advisor.database import TaxonomyDatabase
    db = TaxonomyDatabase(TAXONOMY_PATH)
    summary = db.summary()
    assert summary["total_patterns"] >= 3  # seed patterns
    assert "failure_patterns" in summary


def test_get_pattern_by_id():
    if not TAXONOMY_PATH.exists():
        pytest.skip()
    from tool.abstraction_advisor.database import TaxonomyDatabase
    db = TaxonomyDatabase(TAXONOMY_PATH)
    p = db.get_pattern("P001")
    assert p is not None
    assert p["id"] == "P001"
