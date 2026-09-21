#!/bin/bash
# ============================================================
# PAGR (MCS-RerankPath) — 多轮 QA 重复跑脚本
# 用法: bash scripts/run_qa_multi.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# ===== 配置 =====
DATASETS=("hotpotqa" "2wikimultihopqa" "musique")
RUNS=3
START_RUN=1

LLM_MODEL="deepseek-v4-flash"
LLM_BASE_URL="https://api.deepseek.com/v1"

EMBED_BACKEND="http"
EMBED_MODEL="local_embedding"
EMBED_BASE_URL="http://127.0.0.1:8000/v1"

QA_EVAL="hipporag-em-f1"

PIPELINE=".venv/bin/python -u scripts/pipeline.py"

# ===== 前置检查 =====
echo ""
echo "===== 前置检查 ====="

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "❌ DEEPSEEK_API_KEY 未设置"
    exit 1
else
    echo "✅ DEEPSEEK_API_KEY 已设置 (${#DEEPSEEK_API_KEY} chars)"
fi

for ds in "${DATASETS[@]}"; do
    CACHE="outputs/flat/local_embedding/${ds}/retrieval_MCS_RerankPath_1000.json"
    if [ -f "$CACHE" ]; then
        echo "✅ $ds retrieval 缓存存在"
    else
        echo "⚠️  $ds retrieval 缓存不存在，将实时检索"
    fi
done

echo ""
echo "------------------------------"
echo "将跑 3 datasets × ${RUNS} 轮 QA"
echo ""
read -p "按 Enter 开始，或 Ctrl+C 取消... "
echo ""

# ===== 执行 =====
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Dataset: $ds"
    echo "══════════════════════════════════════════════════════"

    for (( run=START_RUN; run<=RUNS; run++ )); do
        run_label="run_${run}"
        OUT_FILE="exp/qa_results_${LLM_MODEL}_mcsrp_${QA_EVAL}.json"
        ARCHIVE="exp/qa_results_${LLM_MODEL}_mcsrp_${QA_EVAL}_${ds}_${run_label}.json"

        if [ -f "$ARCHIVE" ]; then
            echo "  [SKIP] ${run_label}: $ARCHIVE 已存在"
            continue
        fi

        echo ""
        echo "  ── ${ds} / ${run_label} ──"

        # 如果上次结果残留，先删掉
        rm -f "$OUT_FILE"

        $PIPELINE "$ds" qa \
            --embed-backend "$EMBED_BACKEND" \
            --embed-model "$EMBED_MODEL" \
            --embed-base-url "$EMBED_BASE_URL" \
            --llm-model "$LLM_MODEL" \
            --llm-base-url "$LLM_BASE_URL" \
            --qa-eval "$QA_EVAL"

        # 重命名结果文件，带上 dataset 和 run_label
        if [ -f "$OUT_FILE" ]; then
            mv "$OUT_FILE" "$ARCHIVE"
            echo "  ✅ ${run_label} 完成: $ARCHIVE"
        else
            echo "  ❌ ${run_label}: 结果文件未生成！"
        fi
    done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  全部完成！"
echo "══════════════════════════════════════════════════════"

# ===== 汇总 =====
echo ""
echo "===== 结果汇总 ====="

python3 -c "
import json, os, math

datasets = ['hotpotqa', '2wikimultihopqa', 'musique']
llm_model = '${LLM_MODEL}'
qa_eval = '${QA_EVAL}'

for ds in datasets:
    run_files = []
    for run in range(${START_RUN}, ${RUNS} + 1):
        f = f'exp/qa_results_{llm_model}_mcsrp_{qa_eval}_{ds}_run_{run}.json'
        if os.path.exists(f):
            run_files.append(json.load(open(f)))
    
    if not run_files:
        print(f'\n=== {ds} === (无数据)')
        continue
    
    print(f'\n=== {ds} ({len(run_files)} runs) ===')
    ems = [r['qa']['ExactMatch'] for r in run_files]
    f1s = [r['qa']['F1'] for r in run_files]
    
    mem = sum(ems) / len(ems)
    mf1 = sum(f1s) / len(f1s)
    
    if len(ems) >= 2:
        sem = math.sqrt(sum((v-mem)**2 for v in ems) / len(ems))
        sf1 = math.sqrt(sum((v-mf1)**2 for v in f1s) / len(f1s))
        print(f'  EM = {mem:.4f} ± {sem:.4f}')
        print(f'  F1 = {mf1:.4f} ± {sf1:.4f}')
    else:
        print(f'  EM = {mem:.4f}')
        print(f'  F1 = {mf1:.4f}')
    
    for i, d in enumerate(run_files):
        print(f'  run_{START_RUN+i}: EM={d[\"qa\"][\"ExactMatch\"]:.4f}  F1={d[\"qa\"][\"F1\"]:.4f}')
" 2>/dev/null
