#!/bin/bash
# ============================================================
# PAGR (MCS-RerankPath) — 多轮检索 + QA 重复跑脚本
# 用法: bash scripts/run_multi_runs.sh
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
    GRAPH="outputs/flat/local_embedding/${ds}/graph_flat_deepseek-v4-flash.pkl"
    if [ -f "$GRAPH" ]; then
        echo "✅ $ds graph 就绪"
    else
        echo "❌ $ds graph 不存在，请先 build_graph"
        exit 1
    fi
done

echo ""
echo "------------------------------"
echo "将跑 3 datasets × ${RUNS} 轮 (retrieval + QA)"
echo ""
read -p "按 Enter 开始，或 Ctrl+C 取消... "
echo ""

NUM_SAMPLES=1000

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Dataset: $ds"
    echo "══════════════════════════════════════════════════════"

    for (( run=START_RUN; run<=RUNS; run++ )); do
        run_label="run_${run}"
        OUT_METRICS="outputs/flat/local_embedding/${ds}/retrieval_MCS_RerankPath_${NUM_SAMPLES}_metrics_${run_label}.json"

        echo ""
        echo "  ── ${ds} / ${run_label} ──"

        # ── Retrieval ──
        if [ -f "$OUT_METRICS" ]; then
            echo "  [SKIP] Retrieval ${run_label}: 已存在"
        else
            echo "  [RUN]  Retrieval ${run_label}..."

            $PIPELINE "$ds" retrieval \
                --embed-backend "$EMBED_BACKEND" \
                --embed-model "$EMBED_MODEL" \
                --embed-base-url "$EMBED_BASE_URL" \
                --llm-model "$LLM_MODEL" \
                --llm-base-url "$LLM_BASE_URL" \
                --num-samples "$NUM_SAMPLES"

            # 重命名 metrics 文件（retrieval 结果不重命名，QA 阶段需要固定文件名读取）
            BASE="outputs/flat/local_embedding/${ds}/retrieval_MCS_RerankPath_${NUM_SAMPLES}"
            if [ -f "${BASE}_metrics.json" ]; then
                mv "${BASE}_metrics.json" "${BASE}_metrics_${run_label}.json"
            fi
            echo "  ✅ Retrieval ${run_label} 完成"
        fi

        # ── QA ──
        QA_OUT="exp/qa_results_${LLM_MODEL}_mcsrp_${QA_EVAL}.json"
        QA_ARCHIVE="exp/qa_results_${LLM_MODEL}_mcsrp_${QA_EVAL}_${ds}_${run_label}.json"

        if [ -f "$QA_ARCHIVE" ]; then
            echo "  [SKIP] QA ${run_label}: 已存在"
            continue
        fi

        echo "  [RUN]  QA ${run_label}..."
        rm -f "$QA_OUT"

        $PIPELINE "$ds" qa \
            --embed-backend "$EMBED_BACKEND" \
            --embed-model "$EMBED_MODEL" \
            --embed-base-url "$EMBED_BASE_URL" \
            --llm-model "$LLM_MODEL" \
            --llm-base-url "$LLM_BASE_URL" \
            --qa-eval "$QA_EVAL"

        if [ -f "$QA_OUT" ]; then
            mv "$QA_OUT" "$QA_ARCHIVE"
            echo "  ✅ QA ${run_label} 完成: $QA_ARCHIVE"
        else
            echo "  ❌ QA ${run_label}: 结果文件未生成！"
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

python3 << PYEOF
import json, os, math

datasets = ['hotpotqa', '2wikimultihopqa', 'musique']
llm_model = '${LLM_MODEL}'
qa_eval = '${QA_EVAL}'

for ds in datasets:
    # Retrieval 汇总
    ret_files = []
    for run in range(${START_RUN}, ${RUNS} + 1):
        f = f'outputs/flat/local_embedding/{ds}/retrieval_MCS_RerankPath_${NUM_SAMPLES}_metrics_{run}.json'
        if os.path.exists(f):
            ret_files.append(json.load(open(f)))

    if ret_files:
        print(f'\n=== {ds} Retrieval ({len(ret_files)} runs) ===')
        for k in ['1', '5', '10', '20']:
            vals = [r['recall_at_k'][k] for r in ret_files if k in r.get('recall_at_k', {})]
            if vals:
                m = sum(vals) / len(vals)
                s = math.sqrt(sum((v-m)**2 for v in vals) / len(vals)) if len(vals) >= 2 else 0
                print(f'  R@{k:<2} = {m:.4f} ± {s:.4f}')
        for i, d in enumerate(ret_files):
            r = d['recall_at_k']
            print(f'  run_{${START_RUN}+i}: R@1={r.get(\"1\",0):.4f} R@5={r.get(\"5\",0):.4f} R@10={r.get(\"10\",0):.4f}')

    # QA 汇总
    qa_files = []
    for run in range(${START_RUN}, ${RUNS} + 1):
        f = f'exp/qa_results_{llm_model}_mcsrp_{qa_eval}_{ds}_run_{run}.json'
        if os.path.exists(f):
            qa_files.append(json.load(open(f)))

    if qa_files:
        print(f'\n=== {ds} QA ({len(qa_files)} runs) ===')
        ems = [r['qa']['ExactMatch'] for r in qa_files]
        f1s = [r['qa']['F1'] for r in qa_files]
        mem = sum(ems) / len(ems)
        mf1 = sum(f1s) / len(f1s)
        sem = math.sqrt(sum((v-mem)**2 for v in ems) / len(ems)) if len(ems) >= 2 else 0
        sf1 = math.sqrt(sum((v-mf1)**2 for v in f1s) / len(f1s)) if len(f1s) >= 2 else 0
        print(f'  EM = {mem:.4f} ± {sem:.4f}')
        print(f'  F1 = {mf1:.4f} ± {sf1:.4f}')
        for i, d in enumerate(qa_files):
            print(f'  run_{${START_RUN}+i}: EM={d["qa"]["ExactMatch"]:.4f}  F1={d["qa"]["F1"]:.4f}')
PYEOF
