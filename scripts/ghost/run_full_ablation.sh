#!/usr/bin/env bash
# Full Cold/Ghost ablation evaluation. No --limit / smoke mode is used.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-experiments/cold_ghost_full/ablation}"
TASKS=(gqa mmbench refcoco_testA refcoco_testB)
ABLATIONS=(full self_only context_only no_fresh uniform_ghost)

mapfile -t CONFIGS < <(python -m cold_ghost.cli list --group ghost | grep '^experiment/ghost/' | grep -v '_stagewise$')

mkdir -p "$OUT_ROOT"
printf '%s\n' "# Full ablation sweep" "# configs=${#CONFIGS[@]} tasks=${#TASKS[@]} ablations=${#ABLATIONS[@]}" > "$OUT_ROOT/README.txt"

for cfg in "${CONFIGS[@]}"; do
  tag="${cfg//\//_}"
  for abl in "${ABLATIONS[@]}"; do
    for task in "${TASKS[@]}"; do
      out="$OUT_ROOT/$tag/$abl/$task"
      echo "[ABLATION] config=$cfg ablation=$abl task=$task out=$out"
      python -m cold_ghost.cli eval \
        --config "$cfg" \
        --ablation "$abl" \
        --task "$task" \
        --out "$out"
    done
  done
done

python -m cold_ghost.cli collect \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/summary.csv"

echo "Full ablation sweep completed: $OUT_ROOT"
