#!/usr/bin/env bash
# No 7B downloads and no GPU execution. Run from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python -m cold_ghost.cli verify
python -m pytest -q --junitxml="${JUNIT_OUT:-/tmp/cold_ghost_cpu_results.xml}"
python -m cold_ghost.examples.gqa_fixed
