#!/usr/bin/env python3
"""
Pipeline: 四步独立执行的 MCS-RerankPath 索引和评估流程。

Steps:
  openie:      加载 corpus → passage 级 OpenIE (NER+Triple) → 保存结果
  build_graph: 加载 OpenIE 结果 → chunk embedding → entity/fact → 建图
  retrieval:   加载图 + stores → MCS-RP 检索 recall 评估
  qa:          加载图 + stores → MCS-RP 端到端 QA 评估

Usage:
    # 完整流程（分步）
    python3 scripts/pipeline.py hotpotqa openie
    python3 scripts/pipeline.py hotpotqa build_graph --embed-backend http --embed-base-url http://localhost:8000/v1
    python3 scripts/pipeline.py hotpotqa retrieval --embed-backend http --embed-base-url http://localhost:8000/v1
    python3 scripts/pipeline.py hotpotqa qa --embed-backend http --embed-base-url http://localhost:8000/v1
"""

import json
import os
import sys
import time
import re
import argparse
import asyncio
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from langchain_openai import ChatOpenAI

from infra.embeddings import EmbeddingClient, RerankerClient
from infra.openie import OpenIE
from infra.retrieval import Passage, load_passages_from_corpus
from infra.embedding_store import EmbeddingStore
from infra.knowledge_graph import KnowledgeGraph
from infra.base import QUERY_PREFIX, DOC_PREFIX, ChunkTuple
from index.helpers import chunk_text, hash_text, build_entity_data_from_passages
from graph.builder import GraphBuilder

# ── 目录配置 ───────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(DIR, '..'))
DATA_DIR = os.path.join(BASE, 'data')
CACHE_BASE = os.path.join(BASE, 'outputs')
EXP_DIR = os.path.join(BASE, 'exp')

DATASET_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
}

DEFAULT_LLM_MODEL = os.environ.get("PIPELINE_LLM_MODEL", "deepseek-v3-0324")
DEFAULT_LLM_BASE_URL = os.environ.get("PIPELINE_LLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")


def log(msg: str):
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _make_model_slug(model_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model_name).rstrip('._-')


# ── Step 1: OpenIE ─────────────────────────────────────────────────────────
def step_openie(dataset: str, working_dir: str,
                llm_model: str = DEFAULT_LLM_MODEL,
                llm_base_url: str = DEFAULT_LLM_BASE_URL,
                force: bool = False) -> Dict:
    llm_slug = _make_model_slug(llm_model)
    openie_path = os.path.join(working_dir, f"openie_results_{llm_slug}.json")

    if not force and os.path.exists(openie_path):
        log(f"[OpenIE] 从缓存加载: {openie_path}")
        with open(openie_path) as f:
            data = json.load(f)
        return data

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    passages = load_passages_from_corpus(corpus_path)
    log(f"[OpenIE] 加载了 {len(passages)} 个 passage (LLM: {llm_model})")

    passages_text = {str(i): p.title + " " + p.text for i, p in enumerate(passages)}

    # 根据 base_url 自动选择对应的 API key
    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        _openie_api_key = os.environ.get("TENCENT_API_KEY", "")
    else:
        _openie_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    llm = ChatOpenAI(
        model=llm_model,
        api_key=_openie_api_key,
        base_url=llm_base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    openie = OpenIE(llm, max_workers=30)

    log(f"[OpenIE] 开始 Passage 级 OpenIE ({len(passages_text)} passages)...")
    ner_results, triple_results = openie.batch_process_passages(passages_text)

    data = {
        "config": {"llm_model": llm_model, "llm_base_url": llm_base_url},
        "ner": ner_results,
        "triple": triple_results,
    }
    os.makedirs(working_dir, exist_ok=True)
    with open(openie_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    log(f"[OpenIE] 已保存到 {openie_path}")

    return data


# ── Step 2: Build Graph ────────────────────────────────────────────────────
def step_build_graph(
    dataset: str, working_dir: str,
    embed_client: EmbeddingClient,
    llm_model: str = DEFAULT_LLM_MODEL,
    chunk_size: int = 512, chunk_overlap: int = 64,
    force: bool = False,
) -> Dict:
    llm_slug = _make_model_slug(llm_model)
    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    openie_path = os.path.join(working_dir, f"openie_results_{llm_slug}.json")

    if not force and os.path.exists(kg_path):
        log(f"[BuildGraph] 加载已有 KG: {kg_path}")
        kg = KnowledgeGraph.load(kg_path)
        chunk_store = EmbeddingStore("chunk", working_dir)
        entity_store = EmbeddingStore("entity", working_dir)
        fact_store = EmbeddingStore("fact", working_dir)
        chunk_map_path = os.path.join(working_dir, "chunk_map.json")
        chunk_to_passage = {}
        if os.path.exists(chunk_map_path):
            with open(chunk_map_path) as f:
                cmap = json.load(f)
                chunk_to_passage = cmap.get("chunk_to_passage", cmap)
        return {
            "kg": kg, "chunk_store": chunk_store,
            "entity_store": entity_store, "fact_store": fact_store,
            "chunk_to_passage": chunk_to_passage,
        }

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    passages = load_passages_from_corpus(corpus_path)
    log(f"[BuildGraph] 加载了 {len(passages)} 个 passage")

    if not os.path.exists(openie_path):
        old_path = os.path.join(working_dir, "openie_results.json")
        if os.path.exists(old_path):
            openie_path = old_path
        else:
            raise FileNotFoundError(
                f"请先运行 OpenIE 步骤: {openie_path} 不存在\n"
                f"  python3 scripts/pipeline.py {dataset} openie"
            )
    with open(openie_path) as f:
        openie_data = json.load(f)

    if "config" in openie_data:
        ner_results = openie_data.get("ner", {})
        triple_results = openie_data.get("triple", {})
    else:
        ner_results = openie_data.get("ner", {})
        triple_results = openie_data.get("triple", {})

    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)

    # Chunking
    chunks: List[ChunkTuple] = []
    chunk_to_passage: Dict[str, int] = {}
    for p_idx, passage in enumerate(tqdm(passages, desc="Chunking", unit="p", leave=False)):
        text = passage.title + " " + passage.text
        chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for c_text in chunk_texts:
            c_hash = hash_text(c_text)
            chunks.append((c_hash, c_text))
            chunk_to_passage[c_hash] = p_idx

    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    with open(chunk_map_path, "w") as f:
        json.dump({"chunk_to_passage": chunk_to_passage}, f)
    log(f"[BuildGraph] 生成 {len(chunks)} 个 chunk")

    chunk_texts = [t for _, t in chunks]
    chunk_embs = embed_client.encode_batch([DOC_PREFIX + t for t in chunk_texts])
    chunk_store.insert(chunk_texts, chunk_embs)

    # Entity
    entity_names, entity_name_to_desc, entity_to_passages = \
        build_entity_data_from_passages(passages, ner_results, triple_results)
    entity_embs = None
    if entity_names:
        entity_embs = embed_client.encode_batch(entity_names)
        entity_store.insert(entity_names, entity_embs)
    log(f"[BuildGraph] Entity embedding: {entity_store.count()} 条")

    # Fact
    all_triples: List[list] = []
    fact_texts = []
    for pid in tqdm(triple_results, desc="Extract facts", unit="p", leave=False):
        for triple in triple_results[pid]:
            if len(triple) == 3:
                all_triples.append(triple)
                fact_texts.append(json.dumps(triple, ensure_ascii=False))

    if fact_texts:
        fact_embs = embed_client.encode_batch(fact_texts)
        fact_store.insert(fact_texts, fact_embs)
    log(f"[BuildGraph] Fact embedding: {fact_store.count()} 条")

    # KG
    log(f"[BuildGraph] 构建 flat 图...")
    builder = GraphBuilder()
    kg = builder.build(
        entity_names=entity_names,
        entity_name_to_desc=entity_name_to_desc,
        entity_to_passages=entity_to_passages,
        entity_embs=entity_embs,
        all_triples=all_triples,
        passages=passages,
    )
    kg.save(kg_path)
    log(f"[BuildGraph] KG 已保存 ({kg.graph.vcount()} 节点, {kg.graph.ecount()} 边)")

    return {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage, "chunks_list": chunks,
    }


# ── Step 3: Retrieval ──────────────────────────────────────────────────────
def step_retrieval(
    dataset: str, working_dir: str,
    embed_client: EmbeddingClient,
    reranker_client: RerankerClient,
    num_samples: int = 0,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    ablation_mode: str = "none",
    inner_depth: int = 3,
    inner_width: int = 8,
    ppr_damping: float = 0.4,
):
    from retrievers.mcs_rp import MCSRerankPathRetriever

    log(f"[Retrieval] 加载 {dataset} 的 KG 和 stores...")
    llm_slug = _make_model_slug(llm_model)
    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 build_graph: {kg_path} 不存在")

    kg = KnowledgeGraph.load(kg_path)
    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)

    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    chunk_to_passage = {}
    chunks_list = []
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path) as f:
            cmap = json.load(f)
            chunk_to_passage = cmap.get("chunk_to_passage", cmap)

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(DATA_DIR, f"{dataset}.json")
    passages = load_passages_from_corpus(corpus_path)

    for p_idx, passage in enumerate(passages):
        text = passage.title + " " + passage.text
        for c_text in chunk_text(text, chunk_size=512, overlap=64):
            c_hash = hash_text(c_text)
            if c_hash in chunk_to_passage:
                chunks_list.append((c_hash, c_text))

    index_result = {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage,
        "chunks_list": chunks_list, "passages": passages,
    }

    log(f"[Retrieval] KG: {kg.graph.vcount()} 节点, {len(kg.passage_node_idxs)} passage")

    samples_path = os.path.join(DATA_DIR, f"{dataset}.json")
    if not os.path.exists(samples_path):
        log(f"[Retrieval] 没有 samples 文件: {samples_path}，跳过")
        return

    with open(samples_path) as f:
        all_samples = json.load(f)
    if num_samples > 0:
        all_samples = all_samples[:num_samples]
    log(f"[Retrieval] {len(all_samples)} 个 sample")

    # 根据 base_url 自动选择对应的 API key
    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        _ret_api_key = os.environ.get("TENCENT_API_KEY", "")
    else:
        _ret_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    llm = ChatOpenAI(
        model=llm_model,
        api_key=_ret_api_key,
        base_url=llm_base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    retriever = MCSRerankPathRetriever(
        index_result=index_result, embedding_client=embed_client,
        reranker_client=reranker_client, llm=llm,
        ablation_mode=ablation_mode,
    )

    log(f"\n[Retrieval] Running MCS-RerankPath...")
    results: list = [None] * len(all_samples)
    lock = Lock()

    def _process_one(idx: int):
        sample = all_samples[idx]
        q = sample["question"]
        retrieved = retriever.retrieve(q, top_k=20, inner_depth=inner_depth, inner_width=inner_width, ppr_damping=ppr_damping)
        seen_fulltexts = set()
        deduped = []
        for p, score in retrieved:
            ft = p.title + '\n' + p.text
            if ft not in seen_fulltexts:
                seen_fulltexts.add(ft)
                deduped.append((p, score))
        return idx, [(p.title, p.title + '\n' + p.text) for p, _ in deduped]

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_process_one, i): i for i in range(len(all_samples))}
        for future in as_completed(futures):
            idx, entries = future.result()
            with lock:
                results[idx] = entries
                done = sum(1 for r in results if r is not None)
                if done == len(all_samples) or done % max(len(all_samples) // 20, 1) == 0:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(all_samples) - done) / rate if rate > 0 else 0
                    log(f"    [{done}/{len(all_samples)}]  {rate:.1f}q/s  ETA={eta:.0f}s")

    param_suffix = f"_d{inner_depth}w{inner_width}p{str(ppr_damping).replace('.','_')}"
    abl_suffix = f"_{ablation_mode}" if ablation_mode != "none" else ""
    out_path = os.path.join(working_dir, f"retrieval_MCS_RerankPath{abl_suffix}{param_suffix}_{len(all_samples)}.json")
    # 向后兼容：存为 [(title, full_text), ...]
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    log(f"[Retrieval] 结果已保存到 {out_path}")

    # ── Recall 指标计算（对齐 HippoRAG2：使用全文匹配） ──
    # Recall@k = sum(|gold_i ∩ retrieved_i[:k]|) / sum(|gold_i|)
    # 每个 gold passage 是 "title\ncontent" 形式
    def _normalize_text(t):
        return ' '.join(t.split())
    
    def get_gold_fulltexts(sample):
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            gold_titles = set(sf[0] for sf in sample.get("supporting_facts", []))
            results = []
            for title, paras in sample.get("context", []):
                if title in gold_titles:
                    # hotpotqa: ''.join 即可（段落自带空格），2wiki: ' '.join（段落无空格）
                    if dataset == 'hotpotqa':
                        full = title + '\n' + ''.join(paras)
                    else:
                        full = title + '\n' + ' '.join(paras)
                    results.append(_normalize_text(full))
            return results
        elif "paragraphs" in sample:  # musique 等（使用 is_supporting 字段）
            paras = sample.get("paragraphs", [])
            results = []
            for p in paras:
                if p.get("is_supporting", False):
                    full = p["title"] + '\n' + (p.get("text", p.get("paragraph_text", "")))
                    results.append(_normalize_text(full))
            return results
        else:
            # fallback: question_decomposition
            paras = sample.get("paragraphs", [])
            titles = []
            for qd in sample.get("question_decomposition", []):
                idx = qd.get("paragraph_support_idx")
                if idx is not None and idx < len(paras):
                    titles.append(paras[idx]["title"])
            gold_titles = set(titles)
            results = []
            for p in paras:
                if p["title"] in gold_titles:
                    full = p["title"] + '\n' + p.get("text", p.get("paragraph_text", ""))
                    results.append(_normalize_text(full))
            return results
        return []

    TOP_K_LIST = [1, 5, 10, 20]
    total_gold = 0
    hits_by_k = {k: 0 for k in TOP_K_LIST}

    for result_entries, sample in zip(results, all_samples):
        gold_fulltexts = get_gold_fulltexts(sample)
        gold_set = set(gold_fulltexts)
        total_gold += len(gold_set)
        for k in TOP_K_LIST:
            pred_fulltexts = [_normalize_text(ft) for _, ft in result_entries[:k]]
            pred_set = set(pred_fulltexts)
            hits_by_k[k] += len(gold_set & pred_set)

    log(f"")
    log(f"{'='*55}")
    log(f"Recall (passage-level, {len(all_samples)} samples, {total_gold} gold passages):")
    for k in TOP_K_LIST:
        recall = hits_by_k[k] / total_gold if total_gold > 0 else 0
        log(f"  R@{k:<2d} = {recall:.4f}  ({hits_by_k[k]}/{total_gold})")
    log(f"{'='*55}")
    log(f"")

    # 同时保存指标到 JSON
    metrics = {
        "dataset": dataset,
        "method": "MCS_RerankPath",
        "num_samples": len(all_samples),
        "total_gold": total_gold,
        "recall_at_k": {str(k): round(hits_by_k[k] / total_gold, 4) if total_gold > 0 else 0.0
                         for k in TOP_K_LIST},
        "hits_at_k": {str(k): hits_by_k[k] for k in TOP_K_LIST},
    }
    abl_suffix = f"_{ablation_mode}" if ablation_mode != "none" else ""
    metrics_path = os.path.join(working_dir, f"retrieval_MCS_RerankPath{abl_suffix}{param_suffix}_{len(all_samples)}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"[Retrieval] 指标已保存到 {metrics_path}")


# ── Step 4: QA ─────────────────────────────────────────────────────────────
def step_qa(
    dataset: str, working_dir: str,
    embed_client: EmbeddingClient,
    reranker_client: RerankerClient,
    num_samples: int = 0,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    qa_eval_mode: str = "llm-judge",
    llm_judge_mode: str = "open-book",
    ablation_mode: str = "none",
):
    import hashlib as _hashlib
    from collections import Counter
    import string as _string
    from retrievers.mcs_rp import MCSRerankPathRetriever

    # ── EM/F1 工具函数（用于 hipporag-em-f1 模式） ──
    def _normalize_answer(text: str) -> str:
        def remove_articles(t):
            return re.sub(r"\b(a|an|the)\b", " ", t)
        def white_space_fix(t):
            return " ".join(t.split())
        def remove_punc(t):
            exclude = set(_string.punctuation)
            return "".join(ch for ch in t if ch not in exclude)
        def lower(t):
            return t.lower()
        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def _exact_match(gold: str, predicted: str) -> float:
        gn = _normalize_answer(gold)
        pn = _normalize_answer(predicted)
        return 1.0 if gn == pn else 0.0

    def _exact_match_multi(gold_list: list, predicted: str, aggregation_fn=np.max) -> float:
        scores = [_exact_match(g, predicted) for g in gold_list]
        return float(aggregation_fn(scores))

    def _f1_score(gold: str, predicted: str) -> float:
        gn = _normalize_answer(gold)
        pn = _normalize_answer(predicted)
        gt = gn.split()
        pt = pn.split()
        common = Counter(pt) & Counter(gt)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pt)
        recall = 1.0 * num_same / len(gt)
        return 2.0 * (precision * recall) / (precision + recall)

    def _f1_score_multi(gold_list: list, predicted: str, aggregation_fn=np.max) -> float:
        scores = [_f1_score(g, predicted) for g in gold_list]
        return float(aggregation_fn(scores))

    # ── Prompt 模板 ──
    # llm-judge 模式（根据 llm_judge_mode 选择 open-book 或 closed-book）
    if llm_judge_mode == 'closed-book':
        ANSWER_TEMPLATE = """You are an expert knowledge assistant. Your task is to answer the question based on the provided knowledge context.

1. Use ONLY the information from the provided knowledge context and try your best to answer the question.
2. If the knowledge is insufficient, reject to answer the question.
3. Be precise, concise, and answer the question directly and completely. Start your answer with the answer itself, not with preambles like "Based on..." or "The knowledge context...".
4. For factual questions, provide the specific fact or entity name.
5. For temporal questions, provide the specific date, year, or time period.

Question: {question}

Knowledge Context:
{context}

Answer:
"""
    else:
        ANSWER_TEMPLATE = """You are an expert knowledge assistant. Your task is to answer the question based on the provided knowledge context.

1. If the knowledge is insufficient, answer the question based on your own knowledge.
2. Be precise and concise in your answer.
3. For factual questions, provide the specific fact or entity name.
4. For temporal questions, provide the specific date, year, or time period.

Question: {question}

Knowledge Context:
{context}

Answer (be specific and direct):
"""

    JUDGE_PROMPT = """You are an expert evaluator. Determine if the predicted answer is correct based on the question and gold answer.
Be lenient — if the predicted answer contains the gold answer or is semantically equivalent, consider it correct.
Ignore extra words, prefixes, or complete sentences.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted}

Return only "1" (correct) or "0" (incorrect):"""

    # hipporag-em-f1 模式：对齐 HippoRAG2 的 prompt 模板
    HIPPO_SYSTEM_PROMPT = (
        "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
        "Your response start after \"Thought: \", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. "
        "Conclude with \"Answer: \" to present a concise, definitive response, devoid of additional elaborations."
    )
    HIPPO_FEW_SHOT_PASSAGES = (
        "Wikipedia Title: The Last Horse\n"
        "The Last Horse (Spanish:El \u00faltimo caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fern\u00e1n G\u00f3mez.\n\n"
        "Wikipedia Title: Southampton\n"
        "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. "
        "The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. "
        "In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. "
        "The university considers itself one of the top 5 research universities in the UK. "
        "The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) "
        "It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n\n"
        "Wikipedia Title: Stanton Township, Champaign County, Illinois\n"
        "Stanton Township is a township in Champaign County, Illinois, USA. "
        "As of the 2010 census, its population was 505 and it contained 202 housing units.\n\n"
        "Wikipedia Title: Neville A. Stanton\n"
        "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. "
        "Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). "
        "He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. "
        "Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. "
        "He has been published in academic journals including \"Nature\". "
        "He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n\n"
        "Wikipedia Title: Finding Nemo\n"
        "Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton "
        "Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter "
        "Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) "
        "Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"
    )
    HIPPO_FEW_SHOT_QUESTION = "When was Neville A. Stanton's employer founded?"
    HIPPO_FEW_SHOT_ANSWER = (
        "The employer of Neville A. Stanton is University of Southampton. "
        "The University of Southampton was founded in 1862. \nSo the answer is: 1862."
    )

    # ── 加载数据 ──
    log(f"[QA] 加载 {dataset} 的 KG 和 stores...")
    llm_slug = _make_model_slug(llm_model)
    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 build_graph: {kg_path} 不存在")

    kg = KnowledgeGraph.load(kg_path)
    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)

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
        # {title: [paragraphs]}
        passage_list = [Passage(id=t, title=t, text=' '.join(paras) if isinstance(paras, list) else str(paras))
                        for t, paras in corpus.items()]
    else:
        # [{title, text}]
        passage_list = [Passage(id=p.get("id", p["title"]), title=p["title"], text=p["text"]) for p in corpus]
    log(f"[QA] Total passages: {len(passage_list)}")

    index_result = {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage,
        "chunks_list": chunks_list, "passages": passage_list,
        "embedding_client": embed_client,
    }

    samples_path = os.path.join(DATA_DIR, f"{dataset}.json")
    if not os.path.exists(samples_path):
        log(f"[QA] 没有 samples 文件: {samples_path}，跳过")
        return

    with open(samples_path) as f:
        all_samples = json.load(f)
    if num_samples > 0:
        all_samples = all_samples[:num_samples]
    n = len(all_samples)
    log(f"[QA] {n} 个 sample (LLM: {llm_model}) | eval mode: {qa_eval_mode}")

    retriever = MCSRerankPathRetriever(
        index_result=index_result, embedding_client=embed_client,
        reranker_client=reranker_client, llm=None,
        ablation_mode=ablation_mode,
    )

    # 加载 retrieval 缓存，避免 QA 阶段重复检索
    abl_suffix = f"_{ablation_mode}" if ablation_mode != "none" else ""
    retrieval_cache_path = os.path.join(working_dir, f"retrieval_MCS_RerankPath{abl_suffix}_1000.json")
    cached_retrieval = []
    title_to_text = {}
    if os.path.exists(retrieval_cache_path):
        with open(retrieval_cache_path) as f:
            cached_retrieval = json.load(f)
        log(f"[QA] 加载 retrieval 缓存: {len(cached_retrieval)} 个 query")
        # 从 samples 构建 title -> text 映射
        for s in all_samples:
            for p in s.get("paragraphs", []):
                t = p.get("title", "")
                txt = p.get("text", p.get("paragraph_text", ""))
                if t:
                    title_to_text[t] = txt
        log(f"[QA] title->text 映射: {len(title_to_text)} 条")
    else:
        log(f"[QA] 未找到 retrieval 缓存，将实时检索")

    from openai import AsyncOpenAI

    # 根据 base_url 自动选择对应的 API key
    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        _api_key = os.environ.get("TENCENT_API_KEY", "")
    else:
        _api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    _async_client = AsyncOpenAI(
        base_url=llm_base_url,
        api_key=_api_key,
    )

    def _clean_llm_content(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        m = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE).match(t)
        if m:
            t = m.group("body").strip()
        elif t.startswith("```") and t.endswith("```") and len(t) >= 6:
            t = t[3:-3].strip()
        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()
        return t

    def get_gold_titles(sample):
        if "supporting_facts" in sample:
            return [sf[0] for sf in sample.get("supporting_facts", [])]
        elif "question_decomposition" in sample:
            paras = sample.get("paragraphs", [])
            titles = []
            for qd in sample.get("question_decomposition", []):
                idx = qd.get("paragraph_support_idx")
                if idx is not None and idx < len(paras):
                    titles.append(paras[idx]["title"])
            return titles
        return []

    out_filename = f"qa_results_{llm_model}_mcsrp_{qa_eval_mode}.json"
    out_path = os.path.join(EXP_DIR, out_filename)
    os.makedirs(EXP_DIR, exist_ok=True)

    results = [None] * n

    # ── llm-judge 模式：LLM 调用（temperature=0.3） ──
    async def _llm_judge_ask(prompt: str) -> str:
        try:
            completion = await _async_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = completion.choices[0].message.content or ""
            return _clean_llm_content(raw)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"[ERROR] _llm_judge_ask failed: {e}")
            return ""

    # ── hipporag-em-f1 模式：LLM 调用（temperature=0.0，HippoRAG2 prompt 模板） ──
    async def _hipporag_llm_ask(context_passages: list, question: str) -> str:
        # 取前 qa_top_k=10 个 passage
        top_passages = context_passages[:10]
        prompt_user = ""
        for p in top_passages:
            prompt_user += f"Wikipedia Title: {p.title}\n{p.text}\n\n"
        prompt_user += f"Question: {question}\nThought: "
        messages = [
            {"role": "system", "content": HIPPO_SYSTEM_PROMPT},
            {"role": "user", "content": HIPPO_FEW_SHOT_PASSAGES + "\n\nQuestion: " + HIPPO_FEW_SHOT_QUESTION + "\nThought: "},
            {"role": "assistant", "content": HIPPO_FEW_SHOT_ANSWER},
            {"role": "user", "content": prompt_user},
        ]
        try:
            completion = await _async_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = completion.choices[0].message.content or ""
            return _clean_llm_content(raw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"[ERROR] _hipporag_llm_ask failed: {e}")
            return ""

    def _extract_hipporag_answer(text: str) -> str:
        """从 "Thought: ... \nAnswer: ..." 中提取答案"""
        if not text:
            return ""
        try:
            return text.split("Answer:")[1].strip()
        except Exception:
            return text.strip()

    def _build_gold_list(sample):
        gold_raw = sample["answer"]
        gold_list = gold_raw if isinstance(gold_raw, list) else [gold_raw]
        gold_set = set(gold_list)
        if "answer_aliases" in sample and isinstance(sample["answer_aliases"], list):
            gold_set.update(sample["answer_aliases"])
        return list(gold_set)

    async def process_one(idx_sample):
        idx, s = idx_sample
        q = s["question"]
        gold_list = _build_gold_list(s)

        t0 = time.time()
        # 复用 retrieval 缓存（新格式: [(title, full_text), ...] 或旧格式: [title, ...]）
        cached_entry = cached_retrieval[idx] if idx < len(cached_retrieval) else []
        retrieved = []
        if cached_entry and isinstance(cached_entry[0], list):
            # 新格式: [(title, full_text), ...]
            for t, ft in cached_entry:
                retrieved.append((type('Passage', (), {"title": t, "text": ft.split('\n', 1)[1] if '\n' in ft else ft})(), 0.0))
        else:
            # 旧格式: [title, ...]
            for t in cached_entry:
                if t in title_to_text:
                    retrieved.append((type('Passage', (), {"title": t, "text": title_to_text[t]})(), 0.0))
        t1 = time.time()
        retrieval_time = t1 - t0

        if qa_eval_mode == "llm-judge":
            # PAGR 原始方式：20 passage + temperature=0.3 + LLM judge
            context = "\n\n".join([p.text for p, _ in retrieved])
            pred = await _llm_judge_ask(ANSWER_TEMPLATE.format(question=q, context=context))
            t2 = time.time()
            qa_time = t2 - t1
            is_correct = False
            if pred:
                judge_raw = await _llm_judge_ask(
                    JUDGE_PROMPT.format(question=q, gold_answer=gold_list[0], predicted=pred))
                is_correct = judge_raw == "1"
            em_score = 1.0 if is_correct else 0.0
            f1_score_val = em_score
            accuracy_flag = is_correct
        else:
            # hipporag-em-f1：5 passage + temperature=0.0 + HippoRAG2 prompt + 规则 EM/F1
            raw_response = await _hipporag_llm_ask([p for p, _ in retrieved], q)
            t2 = time.time()
            qa_time = t2 - t1
            pred = _extract_hipporag_answer(raw_response)
            em_score = _exact_match_multi(gold_list, pred) if pred else 0.0
            f1_score_val = _f1_score_multi(gold_list, pred) if pred else 0.0
            accuracy_flag = is_correct = em_score >= 1.0

        retrieved_titles = [p.title for p, _ in retrieved]
        gold_titles = get_gold_titles(s)

        return idx, {
            "id": s.get("id", s.get("_id", str(idx))),
            "question": q,
            "gold": gold_list,
            "predicted": pred,
            "correct": is_correct,
            "em": em_score,
            "f1": f1_score_val,
            "retrieved_titles": retrieved_titles,
            "gold_titles": gold_titles,
            "retrieval_time": retrieval_time,
            "qa_time": qa_time,
        }

    def _save_checkpoint():
        done_list = [r for r in results if r is not None]
        if not done_list:
            return
        correct_done = sum(1 for r in done_list if r["correct"])
        avg_em = float(np.mean([r["em"] for r in done_list]))
        avg_f1 = float(np.mean([r["f1"] for r in done_list]))
        save_results = [
            {k: r[k] for k in ["id", "question", "gold", "predicted", "correct",
                               "em", "f1", "retrieved_titles", "gold_titles"]}
            for r in done_list
        ]
        with open(out_path, "w") as f:
            per_query = [
                {"question": r["question"], "gold_answers": r["gold"], "prediction": r["predicted"]}
                for r in done_list
            ]
            json.dump({
                "method": "MCS_RerankPath",
                "qa_eval_mode": qa_eval_mode,
                "llm_model": llm_model,
                "dataset": dataset,
                "qa": {
                    "ExactMatch": round(avg_em, 4),
                    "F1": round(avg_f1, 4),
                },
                "accuracy": correct_done / max(len(done_list), 1) * 100,
                "correct": correct_done,
                "total": len(done_list),
                "avg_retrieval_time": float(np.mean([r["retrieval_time"] for r in done_list])),
                "avg_qa_time": float(np.mean([r["qa_time"] for r in done_list])),
                "results": save_results,
                "per_query": per_query,
            }, f, indent=2, ensure_ascii=False)

    async def run_all():
        sem = asyncio.Semaphore(32)  # 限制并发数，避免 API 限流
        async def throttled(t):
            async with sem:
                return await process_one(t)
        tasks = [(i, s) for i, s in enumerate(all_samples)]
        coros = [throttled(t) for t in tasks]
        for coro in atqdm.as_completed(coros, desc="QA", unit="q"):
            idx, result = await coro
            results[idx] = result
            done = sum(1 for r in results if r is not None)
            if done % 100 == 0 or done == n:
                _save_checkpoint()

    log(f"[QA] 开始评估...")
    asyncio.run(run_all())
    results = [r for r in results if r is not None]
    _save_checkpoint()

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_em = float(np.mean([r["em"] for r in results]))
    avg_f1 = float(np.mean([r["f1"] for r in results]))
    accuracy = correct / max(total, 1) * 100

    log(f"\n{'='*50}")
    if qa_eval_mode == "hipporag-em-f1":
        log(f"Final (hipporag-em-f1):  EM={avg_em:.4f}  F1={avg_f1:.4f}  Acc={accuracy:.1f}%  ({correct}/{total})")
    else:
        log(f"Final (llm-judge):  Acc={accuracy:.1f}%  ({correct}/{total})")
    log(f"{'='*50}")
    log(f"Saved to {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MCS-RerankPath Pipeline")
    parser.add_argument("dataset", choices=list(DATASET_MAP.keys()), help="数据集名称")
    parser.add_argument("step", choices=["openie", "build_graph", "retrieval", "qa"],
                        help="步骤")
    parser.add_argument("--force-openie", action="store_true", help="强制重跑 OpenIE")
    parser.add_argument("--force-graph", action="store_true", help="强制重建图")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk token 大小 (默认: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk 重叠 token 数 (默认: 64)")
    parser.add_argument("--embed-backend", choices=["hash", "local", "http"], default="hash",
                        help="Embedding 后端 (默认: hash)")
    parser.add_argument("--embed-base-url", default="", help="HTTP embedding 服务 base url")
    parser.add_argument("--embed-model", default="minilm-embedding", help="Embedding 模型名称")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="LLM 模型名称")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL, help="LLM API base URL")
    parser.add_argument("--num-samples", type=int, default=0, help="样本数 (0=全量)")
    parser.add_argument("--qa-eval", choices=["llm-judge", "hipporag-em-f1"], default="llm-judge",
                        help="QA 评估方式: llm-judge | hipporag-em-f1 (默认: llm-judge)")
    parser.add_argument("--llm-judge-mode", choices=["open-book", "closed-book"], default="open-book",
                        help="llm-judge 的检索源使用策略: open-book (可用自身知识) | closed-book (仅限 passage, 不足时拒绝)")
    parser.add_argument("--ablation-mode",
                        choices=["none", "no_reranker", "no_mcs", "rerank_only", "no_ppr", "no_dpr_blend", "dpr_only"],
                        default="none",
                        help="消融模式 (默认: none = 完整 MCS-RP)")
    parser.add_argument("--inner-depth", type=int, default=3,
                        help="MCS BFS 搜索最大深度 (默认: 3)")
    parser.add_argument("--inner-width", type=int, default=8,
                        help="MCS BFS 搜索每层最大宽度 (默认: 8)")
    parser.add_argument("--ppr-damping", type=float, default=0.4,
                        help="PPR damping/teleportation factor (默认: 0.4)")

    args = parser.parse_args()
    dataset = DATASET_MAP[args.dataset]

    embed_slug = _make_model_slug(args.embed_model)
    working_dir = os.path.join(CACHE_BASE, "flat", embed_slug, dataset)
    os.makedirs(working_dir, exist_ok=True)

    embed_client = EmbeddingClient(
        mode=args.embed_backend, base_url=args.embed_base_url, model=args.embed_model,
    )

    log(f"Pipeline: {dataset} / {args.step}")
    log(f"LLM: {args.llm_model} | Embedding: {args.embed_model} (dim={embed_client.dim})")
    log(f"Working dir: {working_dir}")

    if args.step == "openie":
        step_openie(dataset, working_dir, llm_model=args.llm_model,
                    llm_base_url=args.llm_base_url, force=args.force_openie)
        log("[Done] OpenIE")

    elif args.step == "build_graph":
        step_build_graph(dataset, working_dir, embed_client,
                         llm_model=args.llm_model,
                         chunk_size=args.chunk_size,
                         chunk_overlap=args.chunk_overlap,
                         force=args.force_graph)
        log("[Done] Build Graph")

    elif args.step == "retrieval":
        reranker_client = RerankerClient(
            mode='http', model='bge-reranker-v2-m3',
            base_url='http://localhost:16144',
            api_key='vllm', threshold=0.0,
        )
        step_retrieval(dataset, working_dir, embed_client, reranker_client,
                       num_samples=args.num_samples,
                       llm_model=args.llm_model, llm_base_url=args.llm_base_url,
                       ablation_mode=args.ablation_mode,
                       inner_depth=args.inner_depth,
                       inner_width=args.inner_width,
                       ppr_damping=args.ppr_damping)
        log("[Done] Retrieval")

    elif args.step == "qa":
        reranker_client = RerankerClient(
            mode='http', model='bge-reranker-v2-m3',
            base_url='http://localhost:16144',
            api_key='vllm', threshold=0.0,
        )
        step_qa(dataset, working_dir, embed_client, reranker_client,
                num_samples=args.num_samples,
                llm_model=args.llm_model, llm_base_url=args.llm_base_url,
                qa_eval_mode=args.qa_eval,
                llm_judge_mode=getattr(args, 'llm_judge_mode', 'open-book'),
                ablation_mode=args.ablation_mode)
        log("[Done] QA")


if __name__ == "__main__":
    main()
