#!/usr/bin/env bash
# Single-variable scan for MCS (depth, width) and PPR (damping).
# Runs on 3 datasets (musique, hotpotqa, 2wikimultihopqa).
# Each dataset: depth×5 + width×6 + damping×9 = 20 runs, 200 samples each.
set -euo pipefail

cd "$(dirname "$0")/.."

EMBED_URL="http://localhost:8000/v1"
EMBED_MODEL="local_embedding"
LLM_MODEL="deepseek-v4-flash"
LLM_URL="https://api.deepseek.com/v1"
NUM_SAMPLES=200

BASE_ARGS=(
  --embed-backend http
  --embed-base-url "${EMBED_URL}"
  --embed-model "${EMBED_MODEL}"
  --llm-model "${LLM_MODEL}"
  --llm-base-url "${LLM_URL}"
  --num-samples "${NUM_SAMPLES}"
)

# Depth scan: width=8, damping=0.4
run_depth_scan() {
  local dataset=$1
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Depth scan: ${dataset}"
  echo "══════════════════════════════════════════════════════════════"
  echo ""
  for d in 1 2 3 5 10; do
    echo "── depth=${d} ──"
    python3 scripts/pipeline.py "${dataset}" retrieval \
      "${BASE_ARGS[@]}" \
      --inner-depth "${d}" --inner-width 8 --ppr-damping 0.4
  done
}

# Width scan: depth=3, damping=0.4
run_width_scan() {
  local dataset=$1
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Width scan: ${dataset}"
  echo "══════════════════════════════════════════════════════════════"
  echo ""
  for w in 1 2 4 8 16 32; do
    echo "── width=${w} ──"
    python3 scripts/pipeline.py "${dataset}" retrieval \
      "${BASE_ARGS[@]}" \
      --inner-depth 3 --inner-width "${w}" --ppr-damping 0.4
  done
}

# Damping scan: depth=3, width=8
run_damping_scan() {
  local dataset=$1
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Damping scan: ${dataset}"
  echo "══════════════════════════════════════════════════════════════"
  echo ""
  for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "── damping=${p} ──"
    python3 scripts/pipeline.py "${dataset}" retrieval \
      "${BASE_ARGS[@]}" \
      --inner-depth 3 --inner-width 8 --ppr-damping "${p}"
  done
}

# Run all scans for a dataset
run_dataset() {
  local dataset=$1
  run_depth_scan "${dataset}"
  run_width_scan "${dataset}"
  run_damping_scan "${dataset}"
}

for ds in musique hotpotqa 2wikimultihopqa; do
  run_dataset "${ds}"
done

echo ""
echo "========================================"
echo "  All scans complete!"
echo "========================================"
