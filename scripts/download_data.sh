#!/usr/bin/env bash
# ============================================================
# download_data.sh — 下载 HotpotQA / 2WikiMultihopQA / MuSiQue
#
# 本项目使用 HippoRAG 格式的数据：{dataset}.json（问题）+
# {dataset}_corpus.json（语料）。官方来源：
#   - 样例/部分数据: https://github.com/OSU-NLP-Group/HippoRAG (reproduce/dataset)
#   - 完整数据集:    https://huggingface.co/datasets/osunlp/HippoRAG_v2
#
# 用法:
#   bash scripts/download_data.sh          # 下载到 ./data
#   bash scripts/download_data.sh <目标目录>
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-data}"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 下载数据到 $DEST =="
git clone --depth 1 https://github.com/OSU-NLP-Group/HippoRAG.git "$TMP/hipporag"

for ds in hotpotqa 2wikimultihopqa musique; do
  for suf in "" "_corpus"; do
    src="$TMP/hipporag/reproduce/dataset/${ds}${suf}.json"
    if [ -f "$src" ]; then
      cp "$src" "$DEST/"
      echo "  + ${ds}${suf}.json"
    else
      echo "  ! 未在仓库中找到，请从 https://huggingface.co/datasets/osunlp/HippoRAG_v2 下载 ${ds}${suf}.json 放到 $DEST/"
    fi
  done
done

echo "完成：$DEST 下现有 $(ls -1 "$DEST"/*.json 2>/dev/null | wc -l) 个 json 文件"
