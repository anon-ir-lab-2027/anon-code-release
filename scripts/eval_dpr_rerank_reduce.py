#!/usr/bin/env python3
"""
Evaluate DPR → reranker recall reduction: DPR retrieves top-20, reranker re-ranks to top-10.

Usage:
    python3 scripts/eval_dpr_rerank_reduce.py hotpotqa
    python3 scripts/eval_dpr_rerank_reduce.py 2wikimultihopqa
    python3 scripts/eval_dpr_rerank_reduce.py musique
"""

import json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infra.embeddings import EmbeddingClient, RerankerClient
from infra.retrieval import Passage, load_passages_from_corpus
from infra.embedding_store import EmbeddingStore
from infra.base import QUERY_PREFIX, DOC_PREFIX
from index.helpers import chunk_text, hash_text

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'flat', 'local_embedding')
TOP_K_LIST = [1, 2, 5, 10, 20]

DATASET_CONFIG = {
    "hotpotqa": {"corpus": "hotpotqa_corpus.json", "samples": "hotpotqa.json"},
    "2wikimultihopqa": {"corpus": "2wikimultihopqa_corpus.json", "samples": "2wikimultihopqa.json"},
    "musique": {"corpus": "musique_corpus.json", "samples": "musique.json"},
}

def load_cache(dataset):
    """Load cached embedding stores."""
    working_dir = os.path.join(CACHE_DIR, dataset)
    chunk_store = EmbeddingStore("chunk", working_dir)
    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    chunk_to_passage = {}
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path) as f:
            cmap = json.load(f)
            chunk_to_passage = cmap.get("chunk_to_passage", cmap)
    return chunk_store, chunk_to_passage

def build_chunk_text_to_passage(chunk_store, chunk_to_passage, passages):
    m = {}
    for p_idx, passage in enumerate(passages):
        text = passage.title + " " + passage.text
        for c_text in chunk_text(text, 512, 64):
            c_hash = hash_text(c_text)
            if c_hash in chunk_to_passage:
                m[c_text] = chunk_to_passage[c_hash]
    return m

def get_gold_fulltexts(sample, dataset):
    def _norm(t):
        return ' '.join(t.split())
    if "supporting_facts" in sample:
        gold_titles = set(sf[0] for sf in sample.get("supporting_facts", []))
        results = []
        for title, paras in sample.get("context", []):
            if title in gold_titles:
                if dataset == 'hotpotqa':
                    full = title + '\n' + ''.join(paras)
                else:
                    full = title + '\n' + ' '.join(paras)
                results.append(_norm(full))
        return results
    elif "paragraphs" in sample:
        paras = sample.get("paragraphs", [])
        results = []
        for p in paras:
            if p.get("is_supporting", False):
                full = p["title"] + '\n' + (p.get("text", p.get("paragraph_text", "")))
                results.append(_norm(full))
        return results
    return []

def process_query(args):
    idx, sample, chunk_store, chunk_text_to_passage, embed_client, reranker_client, passages = args
    query = sample["question"]
    query_vec = embed_client.encode(QUERY_PREFIX + query)
    
    # Step 1: DPR top-20
    results = chunk_store.search(query_vec, top_k=20 * 3)
    seen = set()
    dpr_top20 = []
    for chunk_text, score, _ in results:
        pass_idx = chunk_text_to_passage.get(chunk_text)
        if pass_idx is not None and pass_idx not in seen:
            seen.add(pass_idx)
            dpr_top20.append((passages[pass_idx], score))
        if len(dpr_top20) >= 20:
            break
    
    # Step 2: Reranker re-rank to top-10
    if reranker_client and dpr_top20:
        try:
            texts = [p.title + '\n' + p.text for p, _ in dpr_top20]
            rerank_scores = reranker_client.score_batch(query, texts)
            reranked = sorted(zip(dpr_top20, rerank_scores), key=lambda x: -x[1])
            dpr_top20 = [(p, s) for (p, _), s in reranked]
        except Exception:
            pass
    
    # dedup by full text
    seen_ft = set()
    deduped = []
    for p, s in dpr_top20:
        ft = p.title + '\n' + p.text
        if ft not in seen_ft:
            seen_ft.add(ft)
            deduped.append((p, s))
    
    return idx, [(p.title, p.title + '\n' + p.text) for p, _ in deduped[:10]]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    
    dataset = args.dataset
    config = DATASET_CONFIG[dataset]
    
    embed_client = EmbeddingClient(mode='http', model='local_embedding', base_url='http://localhost:8000/v1', api_key='vllm')
    reranker_client = RerankerClient(mode='http', model='bge-reranker-v2-m3', base_url='http://localhost:16144', api_key='vllm', threshold=0.0)
    
    print(f"[{dataset}] Loading passages and cache...")
    corpus_path = os.path.join(DATA_DIR, config["corpus"])
    passages = load_passages_from_corpus(corpus_path)
    chunk_store, chunk_to_passage = load_cache(dataset)
    chunk_text_to_passage = build_chunk_text_to_passage(chunk_store, chunk_to_passage, passages)
    
    samples_path = os.path.join(DATA_DIR, config["samples"])
    with open(samples_path) as f:
        all_samples = json.load(f)[:1000]
    
    print(f"[{dataset}] {len(all_samples)} samples, {len(passages)} passages")
    
    n = len(all_samples)
    results = [None] * n
    lock = Lock()
    t_start = time.time()
    
    batch_args = [(i, all_samples[i], chunk_store, chunk_text_to_passage, embed_client, reranker_client, passages) for i in range(n)]
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_query, arg): i for i, arg in enumerate(batch_args)}
        for future in as_completed(futures):
            idx, entries = future.result()
            with lock:
                results[idx] = entries
                done = sum(1 for r in results if r is not None)
                if done % max(n // 20, 1) == 0 or done == n:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{n}]  {rate:.1f}q/s  ETA={eta:.0f}s", flush=True)
    
    # Compute recall
    total_gold = 0
    hits_by_k = {k: 0 for k in TOP_K_LIST}
    
    for idx, entries in enumerate(results):
        gold = get_gold_fulltexts(all_samples[idx], dataset)
        gold_set = set(gold)
        total_gold += len(gold_set)
        for k in TOP_K_LIST:
            pred = set(' '.join(ft.split()) for _, ft in entries[:k])
            hits_by_k[k] += len(gold_set & pred)
    
    print()
    print("=" * 55)
    print(f"DPR→Rerank Reduce: top-20 → rerank top-10")
    print(f"Recall (passage-level, {n} samples, {total_gold} gold):")
    for k in TOP_K_LIST:
        recall = hits_by_k[k] / total_gold if total_gold > 0 else 0
        print(f"  R@{k:<2d} = {recall:.4f}  ({hits_by_k[k]}/{total_gold})")
    print("=" * 55)
    
    out = os.path.join(os.path.dirname(__file__), '..', 'outputs', f"{dataset}_dpr_rerank_reduce.json")
    metrics = {
        "dataset": dataset,
        "method": "dpr_rerank_reduce",
        "n": n,
        "total_gold": total_gold,
        "recall_at_k": {str(k): round(hits_by_k[k] / total_gold, 4) if total_gold > 0 else 0.0 for k in TOP_K_LIST},
        "hits_at_k": {str(k): hits_by_k[k] for k in TOP_K_LIST},
    }
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
