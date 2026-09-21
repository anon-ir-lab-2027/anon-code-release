#!/usr/bin/env bash
# Grid search for MCS (inner_depth, inner_width) and PPR (damping) parameters.
# Runs retrieval on musique with 200 samples for each combination.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="musique"
NUM_SAMPLES=200

EMBED_URL="http://localhost:8000/v1"
EMBED_MODEL="local_embedding"
LLM_MODEL="deepseek-v4-flash"
LLM_URL="https://api.deepseek.com/v1"

SEARCH_DIR="outputs/grid_search"
SUMMARY_FILE="${SEARCH_DIR}/grid_summary_${DATASET}_200.csv"
mkdir -p "${SEARCH_DIR}"

# CSV header
echo "depth,width,damping,R@1,R@5,R@10,R@20" > "${SUMMARY_FILE}"

# Total combos
TOTAL=64
COUNT=0

for depth in 2 3 4 5; do
  for width in 4 8 12 16; do
    for damping in 0.2 0.4 0.6 0.85; do
      COUNT=$((COUNT + 1))
      echo ""
      echo "========================================================================"
      echo "  [${COUNT}/${TOTAL}]  depth=${depth}  width=${width}  damping=${damping}"
      echo "========================================================================"
      echo ""

      python3 scripts/pipeline.py "${DATASET}" retrieval \
        --embed-backend http \
        --embed-base-url "${EMBED_URL}" \
        --embed-model "${EMBED_MODEL}" \
        --llm-model "${LLM_MODEL}" \
        --llm-base-url "${LLM_URL}" \
        --num-samples "${NUM_SAMPLES}" \
        --inner-depth "${depth}" \
        --inner-width "${width}" \
        --ppr-damping "${damping}"

      # Parse R@ from the log output (last printed line has all recalls)
      # Find the result JSON with our param suffix
      param_tag="d${depth}w${width}p$(echo ${damping} | sed 's/\./_/')"
      result_path=$(ls outputs/flat/local_embedding/musique/retrieval_MCS_RerankPath_${param_tag}_*.json 2>/dev/null | head -1)

      if [ -z "${result_path}" ]; then
        echo "WARNING: result JSON not found for ${param_tag}, skipping metrics"
        echo "${depth},${width},${damping},ERROR,ERROR,ERROR,ERROR" >> "${SUMMARY_FILE}"
        continue
      fi

      metrics_path="${result_path%.json}_metrics.json"
      if [ -f "${metrics_path}" ]; then
        r1=$(python3 -c "import json; d=json.load(open('${metrics_path}')); print(d.get('R@1', d.get('recall@1', '?')))")
        r5=$(python3 -c "import json; d=json.load(open('${metrics_path}')); print(d.get('R@5', d.get('recall@5', '?')))")
        r10=$(python3 -c "import json; d=json.load(open('${metrics_path}')); print(d.get('R@10', d.get('recall@10', '?')))")
        r20=$(python3 -c "import json; d=json.load(open('${metrics_path}')); print(d.get('R@20', d.get('recall@20', '?')))")
      else
        # Fallback: extract from the log output piped to a temp file
        # We'll just note it as MISSING
        r1="MISSING"
        r5="MISSING"
        r10="MISSING"
        r20="MISSING"
      fi

      echo "${depth},${width},${damping},${r1},${r5},${r10},${r20}" >> "${SUMMARY_FILE}"
      echo "  → R@1=${r1} R@5=${r5} R@10=${r10} R@20=${r20}"
      echo ""

      # Move result JSON to search dir for safekeeping
      if [ -f "${result_path}" ]; then
        cp "${result_path}" "${SEARCH_DIR}/"
        if [ -f "${metrics_path}" ]; then
          cp "${metrics_path}" "${SEARCH_DIR}/"
        fi
      fi
    done
  done
done

echo ""
echo "========================================"
echo "  Grid search complete! ${COUNT} configurations tested."
echo "  Summary: ${SUMMARY_FILE}"
echo "========================================"

# Pretty print the top-10 results
python3 -c "
import csv, sys
rows = []
with open('${SUMMARY_FILE}') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r10 = float(r['R@10'])
            rows.append(r)
        except (ValueError, KeyError):
            pass
rows.sort(key=lambda x: -float(x.get('R@10', 0)))
print()
print('Top-10 by R@10:')
print(f\"{'rank':<5} {'depth':<6} {'width':<6} {'damping':<8} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'R@20':<8}\")
print('-'*55)
for i, r in enumerate(rows[:10]):
    print(f\"{i+1:<5} {r['depth']:<6} {r['width']:<6} {r['damping']:<8} {float(r['R@1']):<8.4f} {float(r['R@5']):<8.4f} {float(r['R@10']):<8.4f} {float(r['R@20']):<8.4f}\")
"
