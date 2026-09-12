#!/usr/bin/env bash
# Use the currently activated conda environment; propagate every failure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -m cold_ghost.cli sweep --kind "profile" "$@"
