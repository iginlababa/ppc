#!/usr/bin/env python3
"""
Stage 7: Decision framework — rule-based abstraction recommendation engine.

Implements the five calibrated decision rules (R1-R5) from taxonomy.json
§decision_framework. Rules are derived from E1-E7 experimental data, not
hypothetical thresholds. See taxonomy.json for evidence citations.

Rules
-----
R1  Launch Overhead Dominance (P001): T_k < overhead_us[abstraction]
R2  Level-Set Dispatch Budget (P008): L × overhead_us > 0.10 × T_compute_us
R3  API Expressivity Gate    (P004): kernel requires tiling AND abstraction is RAJA/Julia
R4  Load Imbalance Compound  (P007): R1 active AND sigma > 1.0
R5  Safe Abstraction          (success): T_k > 10 × overhead AND R3 not active

Usage:
    python analysis/decision_framework.py \
        --profile workload_profile.json \
        --targets nvidia_rtx5060,amd_mi300x \
        --taxonomy data/taxonomy.json \
        --output recommendation.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── PPC thresholds (project_spec.md §9.4) ────────────────────────────────────
PPC_EXCELLENT        = 0.80
PPC_ACCEPTABLE       = 0.60
DEEP_PROFILE_TRIGGER = 0.85

# ── Calibrated launch overhead thresholds (µs) ───────────────────────────────
# Recalibrated from efficiency back-calculation: overhead = T_k × (1-eff)/eff
# These are effective GPU-side overheads AFTER JIT warmup, not nsys host traces.
#
# Sources:
#   julia_cuda:   E3 stencil small RTX (T_k=14µs, eff=0.704 → 5.9µs)
#                 E7 nbody small RTX   (T_k= 9µs, eff=0.540 → 7.7µs)  → 7µs
#   julia_rocm:   E1 stream small AMD  (T_k=3.8µs, eff=0.503 → 3.8µs) → 4µs
#   sycl_hsa:     E1 stream small AMD  (T_k=3.8µs, eff=0.167 → 19µs)
#   kokkos_cuda:  E3 stencil small RTX (T_k=14µs, eff=0.760 → 4.4µs)
#                 E7 nbody small RTX   (T_k= 9µs, eff=0.592 → 6.2µs)  → 5µs
#   kokkos_hip:   E1 stream small AMD  (T_k=3.8µs, eff=0.498 → 3.8µs) → 4µs
#   raja_cuda:    E3 stencil small RTX (T_k=14µs, eff=0.942 → 0.9µs)
#                 E7 nbody small RTX   (T_k= 9µs, eff=0.733 → 3.3µs)  → 2µs
#   raja_hip:     E1 stream small AMD  (T_k=3.8µs, eff=0.513 → 3.6µs) → 4µs
_LAUNCH_OVERHEAD_US: dict[str, float] = {
    "julia_cuda":    7.0,
    "julia_rocm":    4.0,
    "julia":         7.0,   # alias: CUDA path as default
    "sycl_hsa":     19.0,
    "sycl":         19.0,   # alias
    "kokkos_cuda":   5.0,
    "kokkos_hip":    4.0,
    "kokkos":        5.0,   # alias: CUDA path as default (more conservative)
    "raja_cuda":     2.0,
    "raja_hip":      4.0,
    "raja":          2.0,   # alias: CUDA path as default
    "native":        0.0,
}

# ── Platform → abstraction backend suffix ─────────────────────────────────────
_PLATFORM_SUFFIX: dict[str, str] = {
    "nvidia_rtx5060": "cuda",
    "amd_mi300x":     "hip",
}

# ── Abstractions where API prevents shared-memory tiling (P004) ──────────────
_P004_AFFECTED: frozenset[str] = frozenset({"raja_forall", "raja", "julia_cuda",
                                             "julia_rocm", "julia"})

# ── R2 budget fraction: if L × overhead > this fraction of T_compute → poor ──
_R2_BUDGET_FRACTION = 0.10

# ── R1 safe ratio: T_k / overhead > this → abstraction is safe (R5) ─────────
_R5_SAFE_RATIO = 10.0


@dataclass
class WorkloadProfile:
    kernel_name: str
    kernel_duration_us: float           # single-kernel (per-level) GPU execution time (µs)
    arithmetic_intensity: float         # FLOP/byte (roofline x-axis)
    n_levels: int = 1                   # level-set depth (SpTRSV, BFS); 1 for non-level-set
    load_imbalance_sigma: float = 0.0   # CV of row lengths; 0 for regular workloads
    requires_shared_memory_tiling: bool = False  # True for DGEMM, convolution
    problem_size: str = "large"         # "small" | "medium" | "large"
    platform: str = ""                  # "nvidia_rtx5060" | "amd_mi300x" | ""
    # Legacy fields retained for backward compatibility
    memory_regularity: float = 1.0
    control_flow_divergence: float = 0.0
    data_structure_type: str = "dense_array"
    multi_dimensional: bool = False


@dataclass
class RuleResult:
    rule_id: str
    triggered: bool
    pattern: str          # pattern ID (e.g. "P001") or "" for success
    predicted_ppc: float  # point estimate; -1 if not applicable
    rationale: str


@dataclass
class Recommendation:
    strategy: str                   # FULL_ABSTRACTION | CAUTION | HYBRID | NATIVE
    confidence: str                 # HIGH | MEDIUM | LOW
    rationale: str
    suggested_abstractions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    known_patterns: list[str] = field(default_factory=list)
    expected_ppc_range: Optional[tuple[float, float]] = None
    rule_results: list[dict] = field(default_factory=list)
    predicted_ppc_by_abstraction: dict[str, float] = field(default_factory=dict)


# ── Rule implementations ──────────────────────────────────────────────────────

def _overhead_us(abstraction: str, platform: str, n_levels: int = 1) -> float:
    """Return platform-specific overhead_us for abstraction.

    Julia level-set overhead (n_levels > 1) is higher than regular kernel overhead
    because each level requires variable-size dispatch + potential multi-step operations
    (e.g. BFS compact phase uses cumsum which spawns multiple sub-kernels).
    Calibrated from E5 SpTRSV laplacian large RTX (observed eff=0.464 → ~25µs/level).
    BFS julia overhead (~267µs/level from E6 calibration) is flagged as unmodeled.
    """
    abs_lower = abstraction.lower()
    suffix = _PLATFORM_SUFFIX.get(platform.lower(), "")

    # Julia level-set path: higher overhead due to multi-operation dispatch per level.
    # BFS julia (data_structure_type='graph', high n_levels) has even higher overhead
    # (~267µs/level from E6 back-calculation) because each level runs scatter + cumsum
    # (10+ sub-kernel CUDA launches) + gather. Treated as unmodeled; warning added below.
    if abs_lower in ("julia", "julia_cuda", "julia_rocm") and n_levels > 1:
        base = 25.0 if (not suffix or suffix == "cuda") else 5.0
        return base


    if suffix:
        key = f"{abs_lower}_{suffix}"
        if key in _LAUNCH_OVERHEAD_US:
            return _LAUNCH_OVERHEAD_US[key]
    return _LAUNCH_OVERHEAD_US.get(abs_lower, 5.0)


def _r1_launch_overhead(profile: WorkloadProfile,
                        abstraction: str) -> RuleResult:
    """P001: Launch Overhead Dominance — T_k < overhead_us."""
    overhead = _overhead_us(abstraction, profile.platform, profile.n_levels)
    t_k = profile.kernel_duration_us
    triggered = t_k < overhead
    if triggered:
        predicted = t_k / (t_k + overhead) if (t_k + overhead) > 0 else 0.0
        return RuleResult(
            rule_id="R1", triggered=True, pattern="P001",
            predicted_ppc=round(predicted, 3),
            rationale=(
                f"T_k={t_k:.0f} µs < overhead={overhead:.0f} µs for {abstraction}. "
                f"Launch overhead dominates; predicted PPC ≈ {predicted:.2f} "
                f"(conservative lower bound — see taxonomy R1 notes)."
            ),
        )
    return RuleResult(rule_id="R1", triggered=False, pattern="",
                      predicted_ppc=-1, rationale="")


def _r2_level_set_budget(profile: WorkloadProfile,
                         abstraction: str) -> RuleResult:
    """P008: Level-Set Dispatch Amplification — L × overhead > 0.10 × T_compute."""
    if profile.n_levels <= 1:
        return RuleResult(rule_id="R2", triggered=False, pattern="",
                          predicted_ppc=-1, rationale="n_levels=1: not a level-set workload")
    overhead = _overhead_us(abstraction, profile.platform, profile.n_levels)
    t_compute = profile.kernel_duration_us * profile.n_levels  # total compute budget
    accumulated_overhead = profile.n_levels * overhead
    fraction = accumulated_overhead / t_compute if t_compute > 0 else float("inf")
    triggered = fraction > _R2_BUDGET_FRACTION
    if triggered:
        predicted = t_compute / (t_compute + accumulated_overhead)
        return RuleResult(
            rule_id="R2", triggered=True, pattern="P008",
            predicted_ppc=round(predicted, 3),
            rationale=(
                f"L={profile.n_levels} × overhead={overhead:.0f} µs = "
                f"{accumulated_overhead/1000:.1f} ms accumulated overhead vs "
                f"T_compute={t_compute/1000:.1f} ms. "
                f"Overhead fraction={fraction:.1%} > {_R2_BUDGET_FRACTION:.0%} threshold. "
                f"Predicted PPC ≈ {predicted:.2f}. "
                f"Prefer RAJA (overhead={_LAUNCH_OVERHEAD_US['raja']:.0f} µs) or "
                f"Kokkos (overhead={_LAUNCH_OVERHEAD_US['kokkos']:.0f} µs) instead."
            ),
        )
    return RuleResult(rule_id="R2", triggered=False, pattern="",
                      predicted_ppc=-1, rationale="")


def _r3_api_expressivity(profile: WorkloadProfile,
                         abstraction: str) -> RuleResult:
    """P004: API Expressivity Gap — kernel requires tiling AND abstraction cannot express it."""
    if not profile.requires_shared_memory_tiling:
        return RuleResult(rule_id="R3", triggered=False, pattern="",
                          predicted_ppc=-1, rationale="Kernel does not require shared-memory tiling")
    affected = abstraction.lower() in _P004_AFFECTED
    if affected:
        return RuleResult(
            rule_id="R3", triggered=True, pattern="P004",
            predicted_ppc=0.25,   # midpoint of [0.19, 0.35] observed range
            rationale=(
                f"{abstraction} forall/macro cannot express shared-memory tiling. "
                f"Expected PPC ≈ 0.19-0.35 (E2 AMD evidence: raja_naive 0.19-0.32, "
                f"julia_naive 0.20-0.31). "
                f"Use Kokkos TeamPolicy, SYCL local_accessor, or cuBLAS/rocBLAS instead."
            ),
        )
    return RuleResult(rule_id="R3", triggered=False, pattern="",
                      predicted_ppc=-1,
                      rationale=f"{abstraction} can express shared-memory tiling (no P004)")


def _r4_load_imbalance_compound(profile: WorkloadProfile,
                                abstraction: str,
                                r1: RuleResult) -> RuleResult:
    """P007: Load Imbalance Amplification.

    Applies when sigma > 1.0 AND the kernel is in the 'small regime'
    (T_k / overhead < 3 — close enough that imbalance amplifies the base overhead).
    Does not require R1 to have fired: a kernel slightly above the R1 threshold
    still experiences compound P007 penalty under power-law imbalance.
    """
    if profile.load_imbalance_sigma <= 1.0:
        return RuleResult(rule_id="R4", triggered=False, pattern="",
                          predicted_ppc=-1, rationale="")
    overhead = _overhead_us(abstraction, profile.platform, profile.n_levels)
    ratio = profile.kernel_duration_us / overhead if overhead > 0 else float("inf")
    in_small_regime = ratio < 3.0   # within 3× overhead: imbalance amplifies overhead penalty
    if not (r1.triggered or in_small_regime):
        return RuleResult(rule_id="R4", triggered=False, pattern="",
                          predicted_ppc=-1, rationale="")
    sigma = profile.load_imbalance_sigma
    compound_factor = 1.0 - 0.10 * min(sigma, 3.0)
    # Baseline PPC: from R1 if fired, else from T_k/(T_k+overhead)
    baseline = r1.predicted_ppc if r1.triggered else (
        profile.kernel_duration_us / (profile.kernel_duration_us + overhead))
    predicted = baseline * compound_factor
    return RuleResult(
        rule_id="R4", triggered=True, pattern="P007",
        predicted_ppc=round(predicted, 3),
        rationale=(
            f"Load imbalance sigma={sigma:.2f} > 1.0 AND T_k/overhead ratio={ratio:.1f} < 3 "
            f"(small-kernel regime). Baseline PPC={baseline:.3f} × "
            f"(1 - 0.10 × min({sigma:.1f}, 3.0)) = {predicted:.2f}. "
            f"E4 validation: julia/power_law/small predicted≈0.47, observed=0.449."
        ),
    )


def _r5_safe_abstraction(profile: WorkloadProfile,
                         abstraction: str,
                         r1: RuleResult, r3: RuleResult) -> RuleResult:
    """Success prediction — T_k >> overhead AND no API gate AND large problem."""
    if r1.triggered or r3.triggered:
        return RuleResult(rule_id="R5", triggered=False, pattern="",
                          predicted_ppc=-1, rationale="R1 or R3 active: not in safe regime")
    overhead = _overhead_us(abstraction, profile.platform)
    ratio = profile.kernel_duration_us / overhead if overhead > 0 else float("inf")
    if ratio >= _R5_SAFE_RATIO and profile.problem_size == "large":
        # RAJA/Kokkos may exceed 1.0 on wide-level workloads (P008 inverse)
        upper = 1.05 if abstraction.lower() in {"raja", "kokkos"} else 1.00
        return RuleResult(
            rule_id="R5", triggered=True, pattern="",
            predicted_ppc=0.95,
            rationale=(
                f"T_k/overhead ratio={ratio:.0f} >= {_R5_SAFE_RATIO} AND "
                f"problem_size=large AND no API expressivity constraint. "
                f"Expected PPC ≈ 0.90-{upper:.2f}. "
                f"E1/E5/E7 validation: raja large RTX PPC=0.981, kokkos large AMD PPC=1.016."
            ),
        )
    return RuleResult(rule_id="R5", triggered=False, pattern="",
                      predicted_ppc=-1,
                      rationale=f"T_k/overhead ratio={ratio:.1f} < {_R5_SAFE_RATIO} threshold")


# ── Per-abstraction PPC prediction ────────────────────────────────────────────

def predict_ppc(profile: WorkloadProfile, abstraction: str) -> tuple[float, list[RuleResult]]:
    """
    Apply R1-R5 in priority order; return (predicted_ppc, rule_results).
    R3 (API gate) takes priority over R1/R2 for compute-bound kernels.
    """
    r3 = _r3_api_expressivity(profile, abstraction)
    r1 = _r1_launch_overhead(profile, abstraction)
    r2 = _r2_level_set_budget(profile, abstraction)
    r4 = _r4_load_imbalance_compound(profile, abstraction, r1)
    r5 = _r5_safe_abstraction(profile, abstraction, r1, r3)

    rules = [r3, r1, r2, r4, r5]

    # Priority: R3 > R4 > R2 > R1 > R5
    if r3.triggered:
        return r3.predicted_ppc, rules
    if r4.triggered:
        return r4.predicted_ppc, rules
    if r2.triggered:
        return r2.predicted_ppc, rules
    if r1.triggered:
        return r1.predicted_ppc, rules
    if r5.triggered:
        return r5.predicted_ppc, rules

    # No rule triggered: moderate confidence, workload-dependent
    return 0.80, rules


# ── Multi-abstraction recommendation ─────────────────────────────────────────

_ABSTRACTIONS = ["kokkos", "raja", "sycl", "julia"]


def recommend(profile: WorkloadProfile,
              taxonomy: dict,
              abstractions: list[str] | None = None) -> Recommendation:
    """
    Apply R1-R5 to each abstraction and produce a ranked recommendation.
    """
    if abstractions is None:
        abstractions = _ABSTRACTIONS

    # Taxonomy is loaded for documentation and pattern lookup only.
    # Overhead thresholds are authoritative in _LAUNCH_OVERHEAD_US above
    # (recalibrated from efficiency back-calculation; taxonomy values kept in sync).

    results: dict[str, tuple[float, list[RuleResult]]] = {}
    for abs_name in abstractions:
        ppc, rules = predict_ppc(profile, abs_name)
        results[abs_name] = (ppc, rules)

    # Rank by predicted PPC
    ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)

    # ── Unmodeled case warnings ────────────────────────────────────────────
    _unmodeled_warnings: list[str] = []
    if (profile.data_structure_type == "graph"
            and profile.n_levels > 50
            and any(a.lower() in ("julia", "julia_cuda", "julia_rocm")
                    for a in (abstractions or _ABSTRACTIONS))):
        _unmodeled_warnings.append(
            "UNMODELED: Julia BFS-type workloads (graph, n_levels>50) have ~267µs/level "
            "overhead from multi-kernel-per-level (scatter+cumsum+gather, ~10 sub-launches). "
            "Current model uses 25µs/level — actual julia efficiency will be significantly lower. "
            "Evidence: E6 BFS 2d_grid RTX (eff=0.358, back-calc overhead=267µs/level)."
        )
    best_abs, (best_ppc, best_rules) = ranked[0]

    # Collect warnings and patterns
    all_warnings: list[str] = []
    all_patterns: list[str] = []
    rule_output: list[dict] = []
    for abs_name, (ppc, rules) in ranked:
        for r in rules:
            if r.triggered:
                if r.pattern and r.pattern not in all_patterns:
                    all_patterns.append(r.pattern)
                msg = f"[{abs_name}] {r.rule_id}/{r.pattern or 'success'}: {r.rationale}"
                if msg not in all_warnings and r.triggered and r.pattern:
                    all_warnings.append(msg)
        rule_output.append({
            "abstraction": abs_name,
            "predicted_ppc": round(results[abs_name][0], 3),
            "triggered_rules": [
                {"rule_id": r.rule_id, "pattern": r.pattern,
                 "predicted_ppc": r.predicted_ppc}
                for r in results[abs_name][1] if r.triggered
            ],
        })

    predicted_ppc_map = {k: round(v[0], 3) for k, v in results.items()}

    # Strategy selection
    if best_ppc >= PPC_EXCELLENT:
        strategy = "FULL_ABSTRACTION"
        confidence = "HIGH"
    elif best_ppc >= PPC_ACCEPTABLE:
        strategy = "ABSTRACTION_WITH_TUNING"
        confidence = "MEDIUM"
    else:
        strategy = "NATIVE"
        confidence = "HIGH"

    safe_abstractions = [a for a, (p, _) in ranked if p >= PPC_ACCEPTABLE]

    all_warnings = _unmodeled_warnings + all_warnings
    return Recommendation(
        strategy=strategy,
        confidence=confidence,
        rationale=(
            f"Best predicted abstraction: {best_abs} (PPC ≈ {best_ppc:.2f}). "
            f"Abstractions meeting PPC ≥ {PPC_ACCEPTABLE}: "
            f"{safe_abstractions if safe_abstractions else 'none — use native'}."
        ),
        suggested_abstractions=safe_abstractions or ["native"],
        warnings=all_warnings,
        known_patterns=all_patterns,
        expected_ppc_range=(best_ppc - 0.05, min(best_ppc + 0.10, 1.30)),
        rule_results=rule_output,
        predicted_ppc_by_abstraction=predicted_ppc_map,
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_profile(path: Path) -> WorkloadProfile:
    with open(path) as f:
        data = json.load(f)
    # Accept both new and legacy field names
    if "kernel_name" not in data and "kernel" in data:
        data["kernel_name"] = data.pop("kernel")
    return WorkloadProfile(**{k: v for k, v in data.items()
                               if k in WorkloadProfile.__dataclass_fields__})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile",  required=True,
                        help="workload_profile.json with WorkloadProfile fields")
    parser.add_argument("--targets",  default="nvidia_rtx5060,amd_mi300x",
                        help="Comma-separated target platforms (informational)")
    parser.add_argument("--taxonomy", default="data/taxonomy.json",
                        help="Path to taxonomy.json (loads calibrated thresholds)")
    parser.add_argument("--abstractions",
                        default="kokkos,raja,sycl,julia",
                        help="Comma-separated abstractions to evaluate")
    parser.add_argument("--output",   help="Write recommendation to JSON file")
    args = parser.parse_args()

    profile = load_profile(Path(args.profile))

    taxonomy: dict = {}
    tax_path = Path(args.taxonomy)
    if tax_path.exists():
        with open(tax_path) as f:
            taxonomy = json.load(f)
    else:
        print(f"WARNING: taxonomy not found at {args.taxonomy}; "
              f"using built-in calibrated thresholds.", file=sys.stderr)

    targets = [t.strip() for t in args.targets.split(",")]
    abstractions = [a.strip() for a in args.abstractions.split(",")]

    rec = recommend(profile, taxonomy, abstractions)

    output = {
        "kernel":                        profile.kernel_name,
        "targets":                       targets,
        "strategy":                      rec.strategy,
        "confidence":                    rec.confidence,
        "rationale":                     rec.rationale,
        "suggested_abstractions":        rec.suggested_abstractions,
        "predicted_ppc_by_abstraction":  rec.predicted_ppc_by_abstraction,
        "expected_ppc_range":            rec.expected_ppc_range,
        "warnings":                      rec.warnings,
        "known_patterns":                rec.known_patterns,
        "rule_results":                  rec.rule_results,
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
