"""
Project-wide pytest configuration.

CLI options added here are available to all test suites.
Platform and build-directory fixtures are session-scoped so that each
binary is located only once per pytest invocation.
"""

from pathlib import Path
import pytest

# ── Canonical repo root (two levels up from this file) ───────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--platform",
        default="nvidia_rtx5060_laptop",
        help="Target hardware platform (default: nvidia_rtx5060_laptop). "
             "Determines the build sub-directory and binary suffix. "
             "Examples: nvidia_rtx5060_laptop, nvidia_a100, amd_mi250x, intel_pvc",
    )
    parser.addoption(
        "--build-dir",
        default=None,
        metavar="PATH",
        help="Override the build base directory "
             "(default: <repo>/build/stream).",
    )
    parser.addoption(
        "--abstraction",
        default=None,
        metavar="NAME",
        help="Run tests for a single abstraction only "
             "(e.g. kokkos, raja, sycl, julia, numba). "
             "Substring match against the test ID.",
    )
    parser.addoption(
        "--experiment",
        default=None,
        metavar="ID",
        help="Filter to tests for one experiment (e.g. E1, E2).",
    )


@pytest.fixture(scope="session")
def platform(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--platform")


@pytest.fixture(scope="session")
def build_base(request: pytest.FixtureRequest) -> Path:
    override = request.config.getoption("--build-dir")
    if override:
        return Path(override).resolve()
    return REPO_ROOT / "build" / "stream"
