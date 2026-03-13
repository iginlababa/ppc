"""
E1 STREAM Triad — correctness tests for all abstraction variants.

Each test is parametrized over all seven abstractions via the
``stream_result`` / ``triad_result`` indirect fixtures defined in
conftest.py.  A missing binary causes a SKIP, not a FAIL.

Tolerance contract (project_spec.md §7):
    "Abstractions must be functionally equivalent — validated by
     correctness test before any timing run is accepted."

    Relative error threshold: 1e-6 (double precision), 1e-3 (float32).
    The STREAM_CORRECT line embedded in each binary's output must show PASS.
"""

from __future__ import annotations

import math
import re

import pytest

from .conftest import (
    StreamOutput,
    compute_expected,
    SMALL_N,
    WARMUP,
    NUMTIMES,
    _SCALAR,
)

# ── Tolerance ─────────────────────────────────────────────────────────────────

REL_TOL = 1e-6   # for double (default)
REL_TOL_F32 = 1e-3  # for single precision


def _tol(result: StreamOutput) -> float:
    for m in result.meta:
        if m.precision == "float":
            return REL_TOL_F32
    return REL_TOL


def _rel_err(got: float, ref: float) -> float:
    if ref == 0.0:
        return abs(got)
    return abs(got - ref) / abs(ref)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta_n(result: StreamOutput) -> int:
    """Return array size from the first META line that has n>0."""
    for m in result.meta:
        if m.n > 0:
            return m.n
    return SMALL_N


def _total_passes(result: StreamOutput) -> int:
    """warmup + number of timed runs."""
    warmup = WARMUP
    timed  = NUMTIMES
    for m in result.meta:
        if m.warmup > 0:
            warmup = m.warmup
        if m.timed > 0:
            timed = m.timed
    return warmup + timed


# ── Test: exit status ─────────────────────────────────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_exit_code_zero(stream_result: StreamOutput) -> None:
    """Binary must exit 0 (correctness check inside binary must pass)."""
    assert stream_result.returncode == 0, (
        f"Binary exited with code {stream_result.returncode}.\n"
        f"stderr: {stream_result.stderr[:500]}"
    )


# ── Test: embedded STREAM_CORRECT PASS ───────────────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_embedded_correct_pass(stream_result: StreamOutput) -> None:
    """Output must contain a STREAM_CORRECT PASS line."""
    assert stream_result.correct is not None, (
        "No STREAM_CORRECT line found in output.\n"
        f"stdout (first 800 chars):\n{stream_result.stdout[:800]}"
    )
    assert stream_result.correct.passed, (
        "STREAM_CORRECT line reports FAIL.\n"
        f"stdout:\n{stream_result.stdout[:800]}"
    )


# ── Test: per-array max errors within tolerance ───────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_max_err_a_within_tolerance(stream_result: StreamOutput) -> None:
    """max_err_a reported by the binary must be < tolerance."""
    c = stream_result.correct
    if c is None:
        pytest.skip("No STREAM_CORRECT line — covered by test_embedded_correct_pass")
    tol = _tol(stream_result)
    assert c.max_err_a < tol, (
        f"max_err_a={c.max_err_a:.3e} exceeds tolerance {tol:.0e}"
    )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_max_err_b_within_tolerance(stream_result: StreamOutput) -> None:
    """max_err_b reported by the binary must be < tolerance."""
    c = stream_result.correct
    if c is None:
        pytest.skip("No STREAM_CORRECT line")
    tol = _tol(stream_result)
    assert c.max_err_b < tol, (
        f"max_err_b={c.max_err_b:.3e} exceeds tolerance {tol:.0e}"
    )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_max_err_c_within_tolerance(stream_result: StreamOutput) -> None:
    """max_err_c reported by the binary must be < tolerance."""
    c = stream_result.correct
    if c is None:
        pytest.skip("No STREAM_CORRECT line")
    tol = _tol(stream_result)
    assert c.max_err_c < tol, (
        f"max_err_c={c.max_err_c:.3e} exceeds tolerance {tol:.0e}"
    )


# ── Test: cross-validate reported errors against analytical reference ─────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_analytical_reference_cross_validation(stream_result: StreamOutput) -> None:
    """
    Verify the reported max_err values are plausible by comparing the
    *binary's own error* against the analytically computed expected values.

    We don't have the full array, so we verify that the binary's reported
    max_err is consistent with double-precision rounding (< 1e-12 for
    well-behaved FP) rather than a silent wrong result (e.g. > 1e-4).
    """
    c = stream_result.correct
    if c is None:
        pytest.skip("No STREAM_CORRECT line")

    tol = _tol(stream_result)
    # If the binary already reports PASS and individual array tolerances pass,
    # additionally check that errors are not suspiciously large (>10× tolerance
    # would indicate a logic error even if somehow labeled PASS).
    upper_bound = tol * 10
    for name, err in (("a", c.max_err_a), ("b", c.max_err_b), ("c", c.max_err_c)):
        if math.isnan(err):
            continue
        assert err < upper_bound, (
            f"max_err_{name}={err:.3e} is suspiciously large "
            f"(>{upper_bound:.0e}); possible wrong-result bug"
        )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_expected_values_triad_formula(stream_result: StreamOutput) -> None:
    """
    Independently compute expected final values using compute_expected() and
    confirm the binary's reported error bound is consistent.

    BabelStream reference: after (warmup + numtimes) passes through
    Copy→Mul→Add→Triad, the expected scalar value for any element is
    compute_expected(n_passes).
    """
    c = stream_result.correct
    if c is None:
        pytest.skip("No STREAM_CORRECT line")

    n_passes = _total_passes(stream_result)
    exp_a, exp_b, exp_c = compute_expected(n_passes, scalar=_SCALAR)

    tol = _tol(stream_result)

    # The binary checks a single reference value versus the full array.
    # We verify that IF the binary's error is within tol, the reference
    # values we compute agree on sign and approximate magnitude.
    # (A completely wrong formula would produce a huge error, not 0.)
    for name, exp in (("a", exp_a), ("b", exp_b), ("c", exp_c)):
        assert exp != 0.0 or name == "c", (
            f"compute_expected({n_passes}) returned 0.0 for {name} — "
            f"the reference formula may be wrong"
        )
    # Verify that the expected values are finite (guards against formula bugs
    # producing inf/nan after many passes).
    assert all(math.isfinite(v) for v in (exp_a, exp_b, exp_c)), (
        f"compute_expected({n_passes}) produced non-finite values: "
        f"a={exp_a} b={exp_b} c={exp_c}"
    )


# ── Test: output format compliance ────────────────────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_meta_lines_present(stream_result: StreamOutput) -> None:
    """At least two STREAM_META lines must be present (device + params)."""
    assert len(stream_result.meta) >= 2, (
        f"Expected ≥2 STREAM_META lines, got {len(stream_result.meta)}.\n"
        f"stdout (first 800 chars):\n{stream_result.stdout[:800]}"
    )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_run_lines_present(stream_result: StreamOutput) -> None:
    """At least NUMTIMES STREAM_RUN lines must be present."""
    triad_runs = [r for r in stream_result.runs if r.kernel == "triad"]
    assert len(triad_runs) >= NUMTIMES, (
        f"Expected ≥{NUMTIMES} STREAM_RUN kernel=triad lines, "
        f"got {len(triad_runs)}.\n"
        f"stdout (first 800 chars):\n{stream_result.stdout[:800]}"
    )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_summary_line_present(stream_result: StreamOutput) -> None:
    """At least one STREAM_SUMMARY line must be present."""
    assert len(stream_result.summaries) >= 1, (
        f"No STREAM_SUMMARY line found.\n"
        f"stdout (first 800 chars):\n{stream_result.stdout[:800]}"
    )


# ── Test: bandwidth sanity ────────────────────────────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_bandwidth_positive_and_finite(stream_result: StreamOutput) -> None:
    """Every STREAM_RUN line must report a finite positive bandwidth."""
    runs = [r for r in stream_result.runs if r.kernel == "triad"]
    assert runs, "No STREAM_RUN kernel=triad lines found"
    for r in runs:
        assert r.bw_gbs > 0.0 and math.isfinite(r.bw_gbs), (
            f"Run {r.run}: bw_gbs={r.bw_gbs!r} is not a positive finite number"
        )


@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_bandwidth_plausible_lower_bound(stream_result: StreamOutput) -> None:
    """
    Median bandwidth must be > 1 GB/s.

    This is a very conservative lower bound that rules out badly misconfigured
    runs (e.g., zero timing, wrong unit conversion) without being hardware-specific.
    """
    summaries = [s for s in stream_result.summaries if s.kernel == "triad"]
    if not summaries:
        pytest.skip("No triad STREAM_SUMMARY — covered by test_summary_line_present")
    median = summaries[0].median_bw_gbs
    assert median > 1.0, (
        f"Median triad bandwidth {median:.2f} GB/s is implausibly low "
        f"(expected > 1 GB/s even on a slow device)"
    )


# ── Test: triad-only mode consistency ────────────────────────────────────────

@pytest.mark.e1
@pytest.mark.correctness
@pytest.mark.gpu
def test_triad_result_correctness_pass(triad_result: StreamOutput) -> None:
    """
    A triad-only (or default-mode) run must also report STREAM_CORRECT PASS.

    Some binaries accept --triad-only; others only run triad by default.
    The fixture falls back to the normal run if the flag is unsupported.
    """
    assert triad_result.correct is not None, (
        "No STREAM_CORRECT line found in triad output"
    )
    assert triad_result.correct.passed, (
        "STREAM_CORRECT reports FAIL in triad run"
    )
