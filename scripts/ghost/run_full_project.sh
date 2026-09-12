#!/usr/bin/env bash
# Complete research pipeline for mllm-reroute + Cold/Ghost.
# This intentionally runs FULL datasets/configurations: there is no --limit and no debug-updates.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-all}"
SOURCE_JSONL="${SOURCE_JSONL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-data/ghost_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/ghost_val.jsonl}"
EVAL_INDEX="${EVAL_INDEX:-data/eval_images.jsonl}"
OUT_ROOT="${OUT_ROOT:-experiments/cold_ghost_full}"
RUNTIME_PASSES="${RUNTIME_PASSES:-5}"
RUNTIME_WARMUP="${RUNTIME_WARMUP:-2}"
PROFILE_PASSES="${PROFILE_PASSES:-1}"
PROFILE_WARMUP="${PROFILE_WARMUP:-0}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"

mkdir -p "$OUT_ROOT/commands"

run_prepare() {
  if [[ -z "$SOURCE_JSONL" ]]; then
    echo "ERROR: SOURCE_JSONL must point to the independent image-question JSONL used for Ghost training." >&2
    exit 2
  fi
  echo "[1/8] Building the complete original-evaluation exclusion index"
  python -m cold_ghost.cli index-eval --out "$EVAL_INDEX"
  echo "[1/8] Splitting independent Ghost data by image groups"
  python -m cold_ghost.cli split-data \
    --source "$SOURCE_JSONL" \
    --train "$TRAIN_JSONL" \
    --val "$VAL_JSONL" \
    --val-fraction 0.1
  echo "[1/8] Certifying train/val/evaluation image isolation"
  python -m cold_ghost.cli audit-data \
    --train "$TRAIN_JSONL" \
    --val "$VAL_JSONL" \
    --eval-index "$EVAL_INDEX"
}

run_train() {
  echo "[2/8] Training all 12 research Ghost checkpoints (1000 warm-up + 2000 rollout updates each)"
  python -m cold_ghost.cli sweep --kind train --group ghost --model all --tier all \
    --train "$TRAIN_JSONL" --val "$VAL_JSONL" --eval-index "$EVAL_INDEX"
}

run_eval_baseline() {
  echo "[3/8] Evaluating all 38 original configurations on all 12 task configs"
  python -m cold_ghost.cli sweep --kind eval --group original --model all --tier all \
    --tasks all --out "$OUT_ROOT/main"
}

run_eval_ghost() {
  echo "[4/8] Evaluating all 24 Ghost configurations on all 12 task configs"
  python -m cold_ghost.cli sweep --kind eval --group ghost --model all --tier all \
    --tasks all --out "$OUT_ROOT/main"
}

run_efficiency() {
  echo "[5/8] Full prefill FLOPs/KV profiling for all 62 configurations"
  python -m cold_ghost.cli sweep --kind profile --group all --model all --tier all \
    --out "$OUT_ROOT/main"
  echo "[5/8] Full CUDA-event runtime benchmark for all 62 configurations"
  # The sweep CLI uses the research defaults: 5 measured passes, 2 warm-ups, 64 decode tokens.
  # If non-default repetitions are required, invoke the per-config runtime command documented in FULL_RUN_GUIDE_ZH.md.
  python -m cold_ghost.cli sweep --kind runtime --group all --model all --tier all \
    --out "$OUT_ROOT/main"
}

run_diagnostics() {
  echo "[6/8] Running full validation-manifest Ghost diagnostics for all 12 trained checkpoint families"
  MANIFEST="$VAL_JSONL" OUT_ROOT="$OUT_ROOT/diagnostics" bash scripts/ghost/run_full_diagnostics.sh
}

run_ablation() {
  echo "[7/8] Running full ablation matrix: 12 checkpoint families x 5 ablations x 4 tasks"
  OUT_ROOT="$OUT_ROOT/ablation" bash scripts/ghost/run_full_ablation.sh
}

run_collect() {
  echo "[8/8] Collecting main benchmark/profile/runtime artifacts"
  python -m cold_ghost.cli collect \
    --root "$OUT_ROOT/main" \
    --out "$OUT_ROOT/main_summary.csv"
  echo "Results: $OUT_ROOT/main_summary.csv"
}

write_command_manifests() {
  python -m cold_ghost.cli sweep --kind train --group ghost --model all --tier all \
    --train "$TRAIN_JSONL" --val "$VAL_JSONL" --eval-index "$EVAL_INDEX" --dry-run \
    > "$OUT_ROOT/commands/train_12.txt"
  python -m cold_ghost.cli sweep --kind eval --group original --model all --tier all --tasks all \
    --out "$OUT_ROOT/main" --dry-run > "$OUT_ROOT/commands/eval_original_456.txt"
  python -m cold_ghost.cli sweep --kind eval --group ghost --model all --tier all --tasks all \
    --out "$OUT_ROOT/main" --dry-run > "$OUT_ROOT/commands/eval_ghost_288.txt"
  python -m cold_ghost.cli sweep --kind profile --group all --model all --tier all \
    --out "$OUT_ROOT/main" --dry-run > "$OUT_ROOT/commands/profile_62.txt"
  python -m cold_ghost.cli sweep --kind runtime --group all --model all --tier all \
    --out "$OUT_ROOT/main" --dry-run > "$OUT_ROOT/commands/runtime_62.txt"
}

write_command_manifests

case "$PHASE" in
  all)
    run_prepare
    run_train
    run_eval_baseline
    run_eval_ghost
    run_efficiency
    run_diagnostics
    run_ablation
    run_collect
    ;;
  prepare) run_prepare ;;
  train) run_train ;;
  eval-baseline) run_eval_baseline ;;
  eval-ghost) run_eval_ghost ;;
  efficiency) run_efficiency ;;
  diagnostics) run_diagnostics ;;
  ablation) run_ablation ;;
  collect) run_collect ;;
  manifests) echo "Command manifests written under $OUT_ROOT/commands" ;;
  *)
    echo "Unknown PHASE=$PHASE" >&2
    echo "Allowed: all prepare train eval-baseline eval-ghost efficiency diagnostics ablation collect manifests" >&2
    exit 2
    ;;
esac
