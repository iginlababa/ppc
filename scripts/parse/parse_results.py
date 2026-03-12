#!/usr/bin/env python3
"""
Stage 1: Parse raw benchmark output files → data/performance.csv.

Reads raw stdout/stderr files from results/<platform>/<kernel>/ and extracts
timing and throughput metrics into the canonical CSV schema.

Raw files are NEVER modified. All output is appended to data/performance.csv.

Usage:
    python scripts/parse/parse_results.py \
        --results-dir results/nvidia_a100/ \
        --output data/performance.csv \
        [--kernel stream] [--dry-run]
"""

import argparse
import csv
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── CSV schema (project_spec.md §10.1) ───────────────────────────────────────
COLUMNS = [
    "timestamp", "experiment_id", "kernel", "abstraction", "platform",
    "problem_size", "problem_size_n", "run_id", "execution_time_ms",
    "throughput", "efficiency", "hardware_state_verified",
    "compiler_version", "framework_version",
]


# ── Per-kernel parsers ────────────────────────────────────────────────────────

def parse_babelstream(raw_text: str) -> list[dict]:
    """
    BabelStream output format:
        BW = 1234.56 GB/s
    Returns list of {execution_time_ms, throughput} per run line.
    """
    rows = []
    for m in re.finditer(r"BW\s*=\s*([\d.]+)\s*GB/s", raw_text):
        bw = float(m.group(1))
        rows.append({"throughput": bw, "execution_time_ms": float("nan")})
    return rows


def parse_rajaperf(raw_text: str) -> list[dict]:
    """
    RAJAPerf Suite output format — extract GFLOP/s lines.
    TODO: fill in actual regex for RAJAPerf output.
    """
    rows = []
    # TODO: implement based on actual RAJAPerf output format
    return rows


KERNEL_PARSERS = {
    "stream": parse_babelstream,
    "dgemm":  parse_rajaperf,
    "stencil": parse_rajaperf,
    "spmv":   parse_rajaperf,
    "sptrsv": parse_rajaperf,
    "bfs":    lambda txt: [],  # TODO
    "nbody":  lambda txt: [],  # TODO
}


def make_experiment_id(kernel: str, abstraction: str, platform: str,
                       size: str, run_id: int) -> str:
    return f"{kernel}_{abstraction}_{platform}_{size}_{run_id:03d}"


def parse_file(raw_path: Path, platform: str) -> list[dict]:
    """
    Infer kernel/abstraction/size from directory structure:
      results/<platform>/<kernel>/<abstraction>_<size>.out
    """
    parts = raw_path.stem.split("_")
    if len(parts) < 2:
        print(f"  WARNING: Cannot infer abstraction/size from {raw_path.name} — skipping")
        return []

    kernel = raw_path.parent.name
    abstraction = parts[0]
    size = parts[1] if len(parts) > 1 else "unknown"

    parser = KERNEL_PARSERS.get(kernel)
    if parser is None:
        print(f"  WARNING: No parser for kernel '{kernel}' — skipping {raw_path}")
        return []

    raw_text = raw_path.read_text(errors="replace")
    parsed_rows = parser(raw_text)

    rows = []
    for run_id, prow in enumerate(parsed_rows, start=1):
        rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": make_experiment_id(kernel, abstraction, platform, size, run_id),
            "kernel": kernel,
            "abstraction": abstraction,
            "platform": platform,
            "problem_size": size,
            "problem_size_n": "",   # TODO: extract from raw file header
            "run_id": run_id,
            "execution_time_ms": prow.get("execution_time_ms", ""),
            "throughput": prow.get("throughput", ""),
            "efficiency": "",       # computed post-hoc by compute_ppc.py
            "hardware_state_verified": True,
            "compiler_version": "",  # TODO: extract from env_log.txt
            "framework_version": "", # TODO: extract from env_log.txt
        })
    return rows


def append_to_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="e.g. results/nvidia_a100/")
    parser.add_argument("--output", required=True,
                        help="data/performance.csv")
    parser.add_argument("--kernel", help="Filter to a single kernel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    platform = results_dir.name  # derive platform from directory name

    raw_files = sorted(results_dir.rglob("*.out"))
    if args.kernel:
        raw_files = [f for f in raw_files if f.parent.name == args.kernel]

    if not raw_files:
        print(f"No .out files found under {results_dir}")
        return 0

    total_rows = 0
    for raw_path in raw_files:
        print(f"Parsing: {raw_path.relative_to(results_dir)}")
        rows = parse_file(raw_path, platform)
        if rows and not args.dry_run:
            append_to_csv(rows, Path(args.output))
        total_rows += len(rows)
        print(f"  → {len(rows)} rows")

    print(f"\nTotal: {total_rows} rows appended to {args.output}")
    print("Next: python scripts/parse/validate_schema.py --input data/performance.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
