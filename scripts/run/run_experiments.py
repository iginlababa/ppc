#!/usr/bin/env python3
"""
Master experiment driver — runs the full or partial experiment matrix.

Delegates to per-kernel shell scripts in scripts/run/.
Logs all runs and checks for the native utilization gate before proceeding
to abstraction experiments.

Usage:
    python scripts/run/run_experiments.py \
        --platform nvidia_a100 \
        --reps 30 \
        [--experiments E1,E2] \
        [--abstractions kokkos,raja] \
        [--sizes small,medium,large] \
        [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path


KERNEL_SCRIPTS = {
    "E1": ("stream",  "scripts/run/run_stream.sh"),
    "E2": ("dgemm",   "scripts/run/run_dgemm.sh"),
    "E3": ("stencil", "scripts/run/run_stencil.sh"),
    "E4": ("spmv",    "scripts/run/run_spmv.sh"),
    "E5": ("sptrsv",  "scripts/run/run_sptrsv.sh"),
    "E6": ("bfs",     "scripts/run/run_bfs.sh"),
    "E7": ("nbody",   "scripts/run/run_nbody.sh"),
}

ALL_ABSTRACTIONS = ["native", "kokkos", "raja", "sycl", "julia"]
ALL_SIZES = ["small", "medium", "large"]


def run_kernel(experiment_id: str, kernel: str, script: str,
               platform: str, abstractions: list, sizes: list,
               reps: int, dry_run: bool) -> bool:
    """Run all abstraction/size combinations for one kernel. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"[{experiment_id}] {kernel.upper()} on {platform}")
    print(f"{'='*60}")

    # Always run native first
    ordered = ["native"] + [a for a in abstractions if a != "native"]

    for abstraction in ordered:
        for size in sizes:
            cmd = [
                "bash", script,
                "--abstraction", abstraction,
                "--platform", platform,
                "--size", size,
                "--reps", str(reps),
            ]
            print(f"  Running: {' '.join(cmd)}")
            if dry_run:
                print("  [DRY RUN] skipped")
                continue
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  ERROR: {experiment_id}/{abstraction}/{size} failed (rc={result.returncode})")
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True,
                        choices=["nvidia_rtx5060", "nvidia_a100", "amd_mi250x", "intel_pvc"])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--experiments", default=",".join(KERNEL_SCRIPTS.keys()),
                        help="Comma-separated experiment IDs (default: all)")
    parser.add_argument("--abstractions", default=",".join(ALL_ABSTRACTIONS),
                        help="Comma-separated abstractions (default: all)")
    parser.add_argument("--sizes", default=",".join(ALL_SIZES))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    abstractions = [a.strip() for a in args.abstractions.split(",")]
    sizes = [s.strip() for s in args.sizes.split(",")]

    failed = []
    for exp_id in experiments:
        if exp_id not in KERNEL_SCRIPTS:
            print(f"WARNING: Unknown experiment ID '{exp_id}' — skipping")
            continue
        kernel, script = KERNEL_SCRIPTS[exp_id]
        ok = run_kernel(exp_id, kernel, script, args.platform,
                        abstractions, sizes, args.reps, args.dry_run)
        if not ok:
            failed.append(exp_id)

    if failed:
        print(f"\nFailed experiments: {failed}")
        return 1

    print("\nAll experiments completed successfully.")
    print("Next: python scripts/parse/parse_results.py ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
