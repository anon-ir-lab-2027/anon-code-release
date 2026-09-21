#!/bin/bash
# PAGR 消融实验 — 按数据集串行执行
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.bashrc 2>/dev/null || true

DATASETS=("hotpotqa" "2wikimultihopqa" "musique")
ABLATION_MODES=("none" "no_mcs" "no_reranker" "no_ppr" "no_dpr_blend" "dpr_only")

LLM_MODEL="deepseek-v4-flash"
LLM_BASE_URL="https://api.deepseek.com/v1"
EMBED_FLAGS="--embed-backend http --embed-model local_embedding --embed-base-url http://127.0.0.1:8000/v1"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')] 数据集: ${ds}"
    echo "═══════════════════════════════════════════════════"

    for mode in "${ABLATION_MODES[@]}"; do
        LOGFILE="outputs/abl_${ds}_${mode}.log"

        # 跳过已有结果
        if grep -q "Recall (passage-level" "$LOGFILE" 2>/dev/null; then
            echo "  [$(date '+%H:%M:%S')] ⏭  ${mode} 已有结果"
            grep -E "R@" "$LOGFILE"
            continue
        fi

        echo "  [$(date '+%H:%M:%S')] ▶  ${mode} 开始..."
        CMD="python3 -u scripts/pipeline.py ${ds} retrieval \
            ${EMBED_FLAGS} \
            --llm-model ${LLM_MODEL} \
            --llm-base-url ${LLM_BASE_URL} \
            --ablation-mode ${mode}"

        # 串行执行
        eval "$CMD" > "$LOGFILE" 2>&1
        echo "  [$(date '+%H:%M:%S')] ✓  ${mode} 完成"
        grep -E "R@" "$LOGFILE" || true
    done
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════"
echo ""
echo "结果汇总:"
echo ""
printf "%-22s %-16s %-16s %-16s %-16s\n" "Dataset.Mode" "R@1" "R@5" "R@10" "R@20"
printf "%-22s %-16s %-16s %-16s %-16s\n" "----------------------" "----------------" "----------------" "----------------" "----------------"
for ds in "${DATASETS[@]}"; do
    for mode in "${ABLATION_MODES[@]}"; do
        LOGFILE="outputs/abl_${ds}_${mode}.log"
        if [ -f "$LOGFILE" ]; then
            r1=$(grep "R@1" "$LOGFILE" | grep -oP '\d+\.\d+' | head -1)
            r5=$(grep "R@5" "$LOGFILE" | grep -oP '\d+\.\d+' | head -1)
            r10=$(grep "R@10" "$LOGFILE" | grep -oP '\d+\.\d+' | head -1)
            r20=$(grep "R@20" "$LOGFILE" | grep -oP '\d+\.\d+' | head -1)
            printf "%-22s %-16s %-16s %-16s %-16s\n" "${ds}.${mode}" "${r1:-N/A}" "${r5:-N/A}" "${r10:-N/A}" "${r20:-N/A}"
        fi
    done
done
