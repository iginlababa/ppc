#!/usr/bin/env bash
# Run BFS (E6).
# Usage: ./scripts/run/run_bfs.sh --platform nvidia_a100 --reps 30
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_BASE="${REPO_ROOT}/results"

PLATFORM="nvidia_rtx5060_laptop"; ABSTRACTION="all"; REPS=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/bfs"
mkdir -p "${OUT_DIR}"
echo "[run_bfs] Placeholder — implement GAP BFS invocation with RMAT/road/random graphs."
