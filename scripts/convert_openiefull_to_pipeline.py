#!/usr/bin/env python3
"""
将 openie_results_full/ 中的 OpenIE 结果（chunk hash 为 key）
转换为 pipeline 所需的格式（passage index 为 key）。

用法:
    python3 scripts/convert_openiefull_to_pipeline.py <dataset>

会自动写入 outputs/flat/minilm-embedding/{dataset}/openie_results_deepseek-v4-flash.json
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE, 'data')
OUTPUT_BASE = os.path.join(BASE, 'outputs')
OPENIE_FULL_DIR = os.path.join(OUTPUT_BASE, 'openie_results_full')

EMBED_SLUG = "minilm-embedding"
LLM_SLUG = "deepseek-v4-flash"


def aggregate_by_passage(chunk_hash_results: dict, chunk_to_passage: dict, num_passages: int, field_name: str):
    """将 chunk hash 级别的结果按 chunk_to_passage 聚合回 passage 级别。"""
    passage_results = [set() for _ in range(num_passages)]

    for chash, items in chunk_hash_results.items():
        p_idx = chunk_to_passage.get(chash)
        if p_idx is None:
            continue
        if field_name == "ner":
            for item in items:
                passage_results[p_idx].add(json.dumps(item, sort_keys=True))
        elif field_name == "triple":
            for triple in items:
                passage_results[p_idx].add(json.dumps(triple, sort_keys=True))

    # 去重后转回原始格式
    result = {}
    for p_idx in range(num_passages):
        if field_name == "ner":
            result[str(p_idx)] = [json.loads(s) for s in passage_results[p_idx]] if passage_results[p_idx] else []
        elif field_name == "triple":
            result[str(p_idx)] = [json.loads(s) for s in passage_results[p_idx]] if passage_results[p_idx] else []
    return result


def convert(dataset: str):
    openie_full_path = os.path.join(OPENIE_FULL_DIR, f"{dataset}_openie_full.json")
    if not os.path.exists(openie_full_path):
        print(f"[ERROR] 找不到: {openie_full_path}")
        sys.exit(1)

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] 找不到: {corpus_path}")
        sys.exit(1)

    working_dir = os.path.join(OUTPUT_BASE, "flat", EMBED_SLUG, dataset)
    chunk_map_path = os.path.join(working_dir, "chunk_map.json")

    if not os.path.exists(chunk_map_path):
        print(f"[ERROR] 找不到: {chunk_map_path}")
        print(f"  请先运行 build_graph 生成 chunk_map.json")
        print(f"  python3 scripts/pipeline.py {dataset} build_graph --embed-backend hash")
        sys.exit(1)

    # 加载
    with open(openie_full_path) as f:
        openie_full = json.load(f)
    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(chunk_map_path) as f:
        cmap = json.load(f)

    chunk_to_passage = cmap.get("chunk_to_passage", cmap)
    num_passages = len(corpus)

    full_ner = openie_full["results"]["ner"]
    full_triple = openie_full["results"]["triple"]

    print(f"[Convert] {dataset}")
    print(f"  corpus passages: {num_passages}")
    print(f"  openie_full chunk hashes (ner): {len(full_ner)}")
    print(f"  openie_full chunk hashes (triple): {len(full_triple)}")
    print(f"  chunk_map entries: {len(chunk_to_passage)}")

    # 聚合
    ner_passage = aggregate_by_passage(full_ner, chunk_to_passage, num_passages, "ner")
    triple_passage = aggregate_by_passage(full_triple, chunk_to_passage, num_passages, "triple")

    # 检查覆盖
    ner_empty = sum(1 for v in ner_passage.values() if not v)
    triple_empty = sum(1 for v in triple_passage.values() if not v)
    print(f"  聚合后 ner: {num_passages - ner_empty}/{num_passages} 个 passage 有 NER 结果")
    print(f"  聚合后 triple: {num_passages - triple_empty}/{num_passages} 个 passage 有 Triple 结果")

    # 写入
    out_path = os.path.join(working_dir, f"openie_results_{LLM_SLUG}.json")
    out_data = {
        "config": {
            "llm_model": LLM_SLUG,
            "dataset": dataset,
            "source": f"openie_results_full/{dataset}_openie_full.json",
        },
        "ner": ner_passage,
        "triple": triple_passage,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"  ✅ 写入: {out_path}")
    print(f"     大小: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 scripts/convert_openiefull_to_pipeline.py <dataset>")
        print("  dataset: hotpotqa | 2wikimultihopqa | musique")
        sys.exit(1)
    convert(sys.argv[1])
