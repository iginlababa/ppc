"""
Fixtures and helpers for E1 STREAM Triad correctness tests.

Binary discovery:
    build/stream/<abstraction>_<platform>/<binary>
    e.g. build/stream/cuda_nvidia_a100/stream-cuda

Run caching:
    Each (binary, n, numtimes) combo is run once per session and the
    raw stdout is stored.  Tests share results without re-running.
"""

from __future__ import annotations

import re
import subprocess
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

# ── Constants ────────────────────────────────────────────────────────────────

# (label, binary-name, build-dir-prefix)
# build-dir-prefix is the part before '_<platform>' in the build directory name.
E1_ABSTRACTIONS: list[tuple[str, str, str]] = [
    ("native-cuda",  "stream-cuda",   "cuda"),
    ("native-hip",   "stream-hip",    "hip"),
    ("kokkos",       "stream-kokkos", "kokkos"),
    ("raja",         "stream-raja",   "raja"),
    ("sycl",         "stream-sycl",   "sycl"),
    ("julia",        "stream-julia",  "julia"),
    ("numba",        "stream-numba",  "numba"),
]

# Small array: fast for correctness, still exercises the kernel non-trivially.
SMALL_N = 1 << 20   # 2^20 = 1 048 576 elements  (~8 MB per array, double)
WARMUP   = 3
NUMTIMES = 5

# Analytical BabelStream initial conditions
_INIT_A  = 0.1
_INIT_B  = 0.2
_INIT_C  = 0.0
_SCALAR  = 0.4


# ── Output data model ────────────────────────────────────────────────────────

@dataclass
class StreamMeta:
    device: str      = ""
    abstraction: str = ""
    n: int           = 0
    warmup: int      = 0
    timed: int       = 0
    precision: str   = "double"
    raw: dict        = field(default_factory=dict)


@dataclass
class StreamCorrect:
    passed: bool     = False
    max_err_a: float = float("nan")
    max_err_b: float = float("nan")
    max_err_c: float = float("nan")


@dataclass
class StreamRun:
    kernel: str   = ""
    run: int      = 0
    n: int        = 0
    time_ms: float = 0.0
    bw_gbs: float  = 0.0


@dataclass
class StreamSummary:
    kernel: str         = ""
    median_bw_gbs: float = 0.0
    min_bw_gbs: float    = 0.0
    max_bw_gbs: float    = 0.0
    mean_bw_gbs: float   = 0.0
    iqr_bw_gbs: float    = 0.0
    outliers: int        = 0


@dataclass
class StreamOutput:
    """Parsed result of a single binary invocation."""
    returncode: int           = -1
    stdout: str               = ""
    stderr: str               = ""
    meta: list[StreamMeta]    = field(default_factory=list)
    correct: Optional[StreamCorrect] = None
    runs: list[StreamRun]     = field(default_factory=list)
    summaries: list[StreamSummary] = field(default_factory=list)


# ── Output parser ────────────────────────────────────────────────────────────

def _kv(line: str) -> dict[str, str]:
    """Parse key=value (and key="quoted value") tokens from a line."""
    out: dict[str, str] = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', line):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(r'(\w+)=([^\s"]+)', line):
        if m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return out


def parse_stream_output(proc: "subprocess.CompletedProcess[str]") -> StreamOutput:
    out = StreamOutput(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    for raw_line in out.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("STREAM_META"):
            kv = _kv(line)
            m = StreamMeta(raw=kv)
            m.device      = kv.get("device", "")
            m.abstraction = kv.get("abstraction", "")
            m.precision   = kv.get("precision", "double")
            if "n" in kv:
                m.n = int(kv["n"])
            if "warmup" in kv:
                m.warmup = int(kv["warmup"])
            if "timed" in kv:
                m.timed = int(kv["timed"])
            out.meta.append(m)

        elif line.startswith("STREAM_CORRECT"):
            kv = _kv(line)
            c = StreamCorrect()
            c.passed    = "PASS" in line
            c.max_err_a = float(kv.get("max_err_a", "nan"))
            c.max_err_b = float(kv.get("max_err_b", "nan"))
            c.max_err_c = float(kv.get("max_err_c", "nan"))
            out.correct = c

        elif line.startswith("STREAM_RUN"):
            kv = _kv(line)
            r = StreamRun()
            r.kernel  = kv.get("kernel", "")
            r.run     = int(kv.get("run", 0))
            r.n       = int(kv.get("n", 0))
            r.time_ms = float(kv.get("time_ms", 0))
            r.bw_gbs  = float(kv.get("bw_gbs", 0))
            out.runs.append(r)

        elif line.startswith("STREAM_SUMMARY"):
            kv = _kv(line)
            s = StreamSummary()
            s.kernel        = kv.get("kernel", "")
            s.median_bw_gbs = float(kv.get("median_bw_gbs", 0))
            s.min_bw_gbs    = float(kv.get("min_bw_gbs", 0))
            s.max_bw_gbs    = float(kv.get("max_bw_gbs", 0))
            s.mean_bw_gbs   = float(kv.get("mean_bw_gbs", 0))
            s.iqr_bw_gbs    = float(kv.get("iqr_bw_gbs", 0))
            s.outliers      = int(kv.get("outliers", 0))
            out.summaries.append(s)

    return out


# ── Analytical reference ─────────────────────────────────────────────────────

def compute_expected(n_passes: int, scalar: float = _SCALAR) -> tuple[float, float, float]:
    """
    Trace BabelStream Copy→Mul→Add→Triad loop for *n_passes* full passes.

    One "pass" = one Copy + one Mul + one Add + one Triad.
    Returns (expected_a, expected_b, expected_c) after n_passes.

    When --numtimes N is given, the kernel loop runs:
        warmup passes  (not counted toward reported timing, but still mutate arrays)
        N timed passes
    Total passes = warmup + N timed.
    """
    a, b, c = _INIT_A, _INIT_B, _INIT_C
    for _ in range(n_passes):
        # Copy:  c = a
        c = a
        # Mul:   b = scalar * c
        b = scalar * c
        # Add:   c = a + b
        c = a + b
        # Triad: a = b + scalar * c
        a = b + scalar * c
    return a, b, c


# ── Binary discovery ─────────────────────────────────────────────────────────

def _find_binary(build_base: Path, platform: str, prefix: str, binary: str) -> Optional[Path]:
    """
    Search for *binary* inside build_base.

    Tried paths (in order):
      1. build_base / f"{prefix}_{platform}" / binary
      2. build_base / binary   (flat layout)
      3. Any direct child directory of build_base that contains *binary*.
    """
    # Canonical path
    p = build_base / f"{prefix}_{platform}" / binary
    if p.exists():
        return p

    # Flat fallback
    p2 = build_base / binary
    if p2.exists():
        return p2

    # Walk one level
    for d in build_base.iterdir():
        if d.is_dir():
            p3 = d / binary
            if p3.exists():
                return p3

    return None


# ── Runner ───────────────────────────────────────────────────────────────────

def _run_binary(
    binary_path: Path,
    n: int = SMALL_N,
    numtimes: int = NUMTIMES,
    warmup: int = WARMUP,
    extra_args: list[str] | None = None,
    timeout: int = 300,
) -> StreamOutput:
    cmd = [
        str(binary_path),
        "--arraysize", str(n),
        "--numtimes",  str(numtimes),
        "--warmup",    str(warmup),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Return a synthetic failed result so tests can report the issue.
        fake = subprocess.CompletedProcess(
            args=cmd, returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Timeout after {timeout}s",
        )
        return parse_stream_output(fake)

    return parse_stream_output(proc)


# ── Session-level run cache ───────────────────────────────────────────────────

class _RunCache:
    """Thread-unsafe but pytest is single-threaded by default."""

    def __init__(self) -> None:
        self._cache: dict[tuple, StreamOutput] = {}

    def get(
        self,
        binary_path: Path,
        n: int,
        numtimes: int,
        warmup: int,
        extra_args: tuple[str, ...],
    ) -> StreamOutput:
        key = (str(binary_path), n, numtimes, warmup, extra_args)
        if key not in self._cache:
            self._cache[key] = _run_binary(
                binary_path, n=n, numtimes=numtimes, warmup=warmup,
                extra_args=list(extra_args) if extra_args else None,
            )
        return self._cache[key]


@pytest.fixture(scope="session")
def run_cache() -> _RunCache:
    return _RunCache()


# ── Per-abstraction result fixture ────────────────────────────────────────────

def _abstraction_binary(
    abstraction_label: str,
    build_base: Path,
    platform: str,
) -> tuple[Optional[Path], str]:
    """Return (binary_path_or_None, binary_name)."""
    for label, binary, prefix in E1_ABSTRACTIONS:
        if label == abstraction_label:
            path = _find_binary(build_base, platform, prefix, binary)
            return path, binary
    return None, abstraction_label


@pytest.fixture
def stream_result(request, build_base, platform, run_cache):
    """
    Indirect fixture.  param = abstraction label (e.g. "native-cuda").

    Returns the cached StreamOutput for a SMALL_N run, or skips if the
    binary was not found.
    """
    label: str = request.param
    binary_path, binary_name = _abstraction_binary(label, build_base, platform)
    if binary_path is None:
        pytest.skip(f"Binary '{binary_name}' not found under {build_base} "
                    f"(platform={platform})")
    return run_cache.get(binary_path, SMALL_N, NUMTIMES, WARMUP, ())


@pytest.fixture
def triad_result(request, build_base, platform, run_cache):
    """
    Like stream_result but passes --triad-only (or the equivalent flag).
    Falls back to the default run if the flag is not supported.
    """
    label: str = request.param
    binary_path, binary_name = _abstraction_binary(label, build_base, platform)
    if binary_path is None:
        pytest.skip(f"Binary '{binary_name}' not found under {build_base} "
                    f"(platform={platform})")
    # Try with --triad-only; fall back to normal run if flag not recognised.
    result = run_cache.get(binary_path, SMALL_N, NUMTIMES, WARMUP, ("--triad-only",))
    if result.returncode not in (0, 1):
        result = run_cache.get(binary_path, SMALL_N, NUMTIMES, WARMUP, ())
    return result


# ── --abstraction filter ──────────────────────────────────────────────────────

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Parametrize any fixture named stream_result or triad_result with
    all E1_ABSTRACTIONS, then apply the --abstraction substring filter.
    """
    for fixture_name in ("stream_result", "triad_result"):
        if fixture_name not in metafunc.fixturenames:
            continue

        labels = [label for label, _, _ in E1_ABSTRACTIONS]

        # --abstraction filter (substring match)
        abstraction_filter = metafunc.config.getoption("--abstraction", default=None)
        if abstraction_filter:
            labels = [l for l in labels if abstraction_filter.lower() in l.lower()]

        # --experiment filter — only include e1 tests when E1 (or no filter)
        experiment_filter = metafunc.config.getoption("--experiment", default=None)
        if experiment_filter and experiment_filter.upper() != "E1":
            labels = []

        metafunc.parametrize(
            fixture_name,
            labels,
            indirect=True,
            ids=labels,
        )
