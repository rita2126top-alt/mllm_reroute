#!/usr/bin/env bash
# Run diagnostics on every trained Ghost checkpoint/config using the full validation manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-data/ghost_val.jsonl}"
OUT_ROOT="${OUT_ROOT:-experiments/cold_ghost_full/diagnostics}"

mapfile -t CONFIGS < <(python -m cold_ghost.cli list --group ghost | grep '^experiment/ghost/' | grep -v '_stagewise$')
mkdir -p "$OUT_ROOT"

for cfg in "${CONFIGS[@]}"; do
  tag="${cfg//\//_}"
  out="$OUT_ROOT/${tag}.json"
  echo "[DIAGNOSE] config=$cfg manifest=$MANIFEST out=$out"
  python -m cold_ghost.cli diagnose \
    --config "$cfg" \
    --manifest "$MANIFEST" \
    --out "$out"
done

echo "Full diagnostics completed: $OUT_ROOT"
