#!/bin/bash
# ============================================================
# PAGR OpenIE — 三个数据集批量跑
# 用法: bash scripts/run_openie.sh
#
# 前置条件:
#   1. DEEPSEEK_API_KEY 环境变量已设置
#   2. .venv 虚拟环境已就绪
#   3. data/{dataset}_corpus.json 文件存在
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# ===== 配置 =====
DATASETS=("hotpotqa" "2wikimultihopqa" "musique")

LLM_MODEL="deepseek-v4-flash"
LLM_BASE_URL="https://api.deepseek.com/v1"

PIPELINE=".venv/bin/python -u scripts/pipeline.py"
# -u 强制 stdout/stderr 无缓冲，确保进度信息实时打印到 tee

# ===== 前置检查 =====
echo ""
echo "===== 前置检查 ====="

# 1. API Key
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "❌ DEEPSEEK_API_KEY 未设置"
    echo "   请先: export DEEPSEEK_API_KEY='sk-...'"
    echo "   或添加到 ~/.bashrc"
    exit 1
else
    echo "✅ DEEPSEEK_API_KEY 已设置 (${#DEEPSEEK_API_KEY} chars)"
fi

# 2. .venv
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ .venv 虚拟环境不存在"
    echo "   请先: uv sync"
    exit 1
else
    echo "✅ .venv 就绪"
fi

# 3. 数据集文件
for ds in "${DATASETS[@]}"; do
    if [ -f "data/${ds}_corpus.json" ]; then
        echo "✅ data/${ds}_corpus.json 存在"
    else
        echo "❌ data/${ds}_corpus.json 不存在"
        exit 1
    fi
done

# 4. 检查已有缓存（不报错，只提示）
for ds in "${DATASETS[@]}"; do
    EXISTING="outputs/flat/minilm-embedding/${ds}/openie_results_deepseek-v4-flash.json"
    if [ -f "$EXISTING" ]; then
        SIZE_MB=$(du -h "$EXISTING" | cut -f1)
        echo "⚠️  ${ds} 已有 openie 缓存 ($SIZE_MB)，将用 --force-openie 覆盖"
    fi
done

echo ""
echo "------------------------------"
echo "即将依次跑 3 个数据集的 OpenIE:"
for ds in "${DATASETS[@]}"; do
    echo "  • $ds"
done
echo ""
echo "LLM: $LLM_MODEL"
echo "Base URL: $LLM_BASE_URL"
echo ""
read -p "按 Enter 开始，或 Ctrl+C 取消... "
echo ""

START_TIME=$(date +%s)

# ===== 执行 =====
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')] 开始 $ds"
    echo "══════════════════════════════════════════════════════"

    LOG_FILE="outputs/flat/minilm-embedding/${ds}/openie_run.log"

    $PIPELINE "$ds" openie \
        --llm-model "$LLM_MODEL" \
        --llm-base-url "$LLM_BASE_URL" \
        --force-openie \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "  ✅ [$(date '+%H:%M:%S')] $ds 完成"
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "══════════════════════════════════════════════════════"
echo "  全部完成！耗时 ${ELAPSED} 分钟"
echo "══════════════════════════════════════════════════════"
echo ""
echo "结果文件:"
for ds in "${DATASETS[@]}"; do
    OUT="outputs/flat/minilm-embedding/${ds}/openie_results_deepseek-v4-flash.json"
    if [ -f "$OUT" ]; then
        SIZE_MB=$(du -h "$OUT" | cut -f1)
        echo "  ✅ $ds: $OUT ($SIZE_MB)"
    else
        echo "  ❌ $ds: 文件不存在！"
    fi
done
