#!/usr/bin/env bash
# Run N-Body (E7).
# Usage: ./scripts/run/run_nbody.sh --platform nvidia_a100 --reps 30
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_BASE="${REPO_ROOT}/results"

PLATFORM="nvidia_a100"; ABSTRACTION="all"; SIZE="all"; REPS=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --abstraction) ABSTRACTION="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

declare -A SIZES; SIZES[small]=32000; SIZES[medium]=256000; SIZES[large]=2000000

OUT_DIR="${RESULTS_BASE}/${PLATFORM}/nbody"
mkdir -p "${OUT_DIR}"
echo "[run_nbody] Placeholder — implement CoMD/miniMD invocation."
