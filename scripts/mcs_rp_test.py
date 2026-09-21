#!/usr/bin/env python3
"""
MCS-RP 检索测试脚本。

支持通过命令行参数指定 query、数据集、top_k 等参数，输出检索到的 passage 信息。

Usage:
    python3 scripts/mcs_rp_test.py --dataset musique --query "Where was Lady Godiva born?" --top-k 5
    python3 scripts/mcs_rp_test.py --dataset hotpotqa --query "Who is Ralph Rapson" --top-k 10 --llm-model deepseek-v4-flash
"""

import json
import os
import sys
import re
import argparse
import hashlib as _hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import List, Tuple, Optional

# ── 目录配置 ──
DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(DIR, '..'))
DATA_DIR = os.path.join(BASE, 'data')
CACHE_BASE = os.path.join(BASE, 'outputs')

DATASET_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
}


def _make_model_slug(model_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model_name).rstrip('._-')


def _get_api_key(llm_base_url: str) -> str:
    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        return os.environ.get("TENCENT_API_KEY", "")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def main():
    parser = argparse.ArgumentParser(description="MCS-RP 检索测试")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_MAP.keys()),
                        help="数据集名称")
    parser.add_argument("--query", required=True, type=str,
                        help="检索 query")
    parser.add_argument("--top-k", type=int, default=10,
                        help="检索 top-k passage 数 (默认: 10)")
    parser.add_argument("--embed-backend", choices=["hash", "local", "http"], default="hash",
                        help="Embedding 后端 (默认: hash)")
    parser.add_argument("--embed-base-url", default="",
                        help="HTTP embedding 服务 base url")
    parser.add_argument("--embed-model", default="minilm-embedding",
                        help="Embedding 模型名称")
    parser.add_argument("--llm-model", default="deepseek-v3-0324",
                        help="LLM 模型名称")
    parser.add_argument("--llm-base-url", default="https://tokenhub.tencentmaas.com/v1",
                        help="LLM API base URL")
    parser.add_argument("--reranker-base-url", default="http://localhost:16144",
                        help="Reranker API base URL")
    parser.add_argument("--ablation-mode",
                        choices=["none", "no_reranker", "no_mcs", "no_ppr", "no_dpr_blend", "dpr_only"],
                        default="none",
                        help="消融模式 (默认: none = 完整 MCS-RP)")

    args = parser.parse_args()
    dataset = DATASET_MAP[args.dataset]
    query = args.query
    top_k = args.top_k
    llm_slug = _make_model_slug(args.llm_model)
    embed_slug = _make_model_slug(args.embed_model)

    working_dir = os.path.join(CACHE_BASE, "flat", embed_slug, dataset)
    if not os.path.exists(working_dir):
        print(f"[ERROR] Working dir not found: {working_dir}")
        print(f"请先运行 build_graph 和 retrieve 生成缓存")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"MCS-RP 检索测试")
    print(f"{'='*70}")
    print(f"Dataset:  {dataset}")
    print(f"Query:    {query}")
    print(f"Top-K:    {top_k}")
    print(f"LLM:      {args.llm_model}")
    print(f"Embed:    {args.embed_model} ({args.embed_backend})")
    print(f"WorkDir:  {working_dir}")

    # ── 加载 KG + Stores ──
    print(f"\n[1/4] 加载 KG 和 embedding stores...")
    from infra.knowledge_graph import KnowledgeGraph
    from infra.embedding_store import EmbeddingStore
    from infra.retrieval import Passage

    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    if not os.path.exists(kg_path):
        # fallback: 尝试不带 slug 的
        kg_path = os.path.join(working_dir, "graph_flat.pkl")
    if not os.path.exists(kg_path):
        print(f"[ERROR] KG 文件不存在: {kg_path}")
        sys.exit(1)

    print(f"  KG: {kg_path}")
    kg = KnowledgeGraph.load(kg_path)

    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)
    print(f"  Chunks: {len(chunk_store.texts)}, Entities: {len(entity_store.texts)}, Facts: {len(fact_store.texts)}")

    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    chunk_to_passage = {}
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path) as f:
            cmap = json.load(f)
            chunk_to_passage = cmap.get("chunk_to_passage", cmap)

    chunks_list = [(_hashlib.md5(t.encode()).hexdigest(), t) for t in chunk_store.texts]

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(DATA_DIR, f"{dataset}.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    if isinstance(corpus, dict):
        passage_list = [Passage(id=t, title=t, text=' '.join(paras) if isinstance(paras, list) else str(paras))
                        for t, paras in corpus.items()]
    else:
        passage_list = [Passage(id=p.get("id", p["title"]), title=p["title"], text=p["text"]) for p in corpus]
    print(f"  Corpus passages: {len(passage_list)}")

    index_result = {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage,
        "chunks_list": chunks_list, "passages": passage_list,
    }

    # ── Embedding + Reranker Client ──
    print(f"\n[2/4] 初始化 Embedding 和 Reranker...")
    from infra.embeddings import EmbeddingClient, RerankerClient
    embed_client = EmbeddingClient(
        mode=args.embed_backend, base_url=args.embed_base_url, model=args.embed_model,
    )
    reranker_client = RerankerClient(
        mode='http', model='bge-reranker-v2-m3',
        base_url=args.reranker_base_url,
        api_key='vllm', threshold=0.0,
    )
    print(f"  Embedding dim: {embed_client.dim}")
    print(f"  Reranker: {args.reranker_base_url}")

    # ── LLM + Retriever ──
    print(f"\n[3/4] 初始化 MCS-RP Retriever...")
    from langchain_openai import ChatOpenAI
    from retrievers.mcs_rp import MCSRerankPathRetriever

    _api_key = _get_api_key(args.llm_base_url)
    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=_api_key,
        base_url=args.llm_base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    retriever = MCSRerankPathRetriever(
        index_result=index_result, embedding_client=embed_client,
        reranker_client=reranker_client, llm=llm,
        ablation_mode=args.ablation_mode,
    )

    # ── 检索 ──
    print(f"\n[4/4] 检索 (top_k={top_k})...")
    import time
    t0 = time.time()
    retrieved = retriever.retrieve(query, top_k=top_k)
    t1 = time.time()
    print(f"  检索耗时: {t1-t0:.2f}s")
    print(f"  返回: {len(retrieved)} 条 passage")

    print(f"\n{'='*70}")
    print(f"检索结果 (top-{len(retrieved)}):")
    print(f"{'='*70}")

    for i, (p, score) in enumerate(retrieved):
        title = p.title
        text_preview = p.text[:200].replace('\n', ' ')
        print(f"\n  [{i+1}] Score={score:.4f}")
        print(f"      Title: {title}")
        print(f"      Text:  {text_preview}...")

    print(f"\n{'='*70}")
    print("完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
