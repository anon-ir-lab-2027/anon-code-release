"""
HippoRAG Indexer: complete index construction executor.

Input: corpus_path + cache_dir + external dependencies -> produces IndexResult.
Completely separated from retrieval logic.

Passage-level OpenIE: NER and Triple extraction are done on full passages,
not chunks. Chunking is only used for the chunk embedding store (DPR).
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from infra.knowledge_graph import KnowledgeGraph
from infra.embedding_store import EmbeddingStore
from infra.embeddings import EmbeddingClient, RerankerClient
from infra.openie import OpenIE
from infra.retrieval import Passage, load_passages_from_corpus
from infra.base import (
    QUERY_PREFIX, DOC_PREFIX,
    FACT_RETRIEVE_TOP_K, FACT_RERANK_TOP_K,
    SYNONYM_TOP_K, SYNONYM_THRESHOLD,
    PPR_DAMPING, PASSAGE_NODE_WEIGHT,
    ChunkTuple, IndexResult,
)
from graph.builder import GraphBuilder
from index.helpers import chunk_text, hash_text, build_entity_data, build_entity_data_from_passages


class HippoRAGIndexer:
    """Complete index construction executor.

    Builds all stores, knowledge graph, and metadata. Returns IndexResult.
    Does NOT contain any retrieval logic.

    Key design decisions:
    - OpenIE (NER/Triple) runs on full passages, NOT chunks.
    - Chunking is only used for DPR-level operations.
    - Graph nodes are indexed by passage index (0-based), not chunk hash.
    """

    def __init__(
        self,
        corpus_path: str,
        cache_dir: str,
        openie: OpenIE,
        embedding_client: EmbeddingClient,
    ):
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.openie = openie
        self.embedding_client = embedding_client
        self.graph_builder = GraphBuilder()

    def index(
        self,
        force_reindex: bool = False,
        force_reopenie: bool = False,
        graph_mode: str = "flat",
    ) -> IndexResult:
        """Full indexing pipeline (passage-level OpenIE).

        Step 1: Load passages
        Step 2: Passage-level OpenIE (NER + Triple extraction)
        Step 3: Chunking + Chunk embedding (only for DPR)
        Step 4: Entity from passage-level OpenIE -> entity_store
        Step 5: Fact construction -> fact_store
        Step 6: Passage-level Graph construction

        Cache logic: if graph.pkl + openie_results.json exist, load cached.
        force_reindex=True: rebuild everything.
        force_reopenie=True: re-run OpenIE even with existing cache.

        Returns:
            IndexResult dict with keys:
              kg, chunk_store, entity_store, fact_store,
              chunk_to_passage, chunks_list, passages,
              ner_results, triple_results
        """
        ds_name = os.path.basename(self.corpus_path).replace("_corpus.json", "").replace(".json", "")
        # graph_mode is always "flat" (H2 removed)
        working_dir = os.path.join(self.cache_dir, ds_name)
        os.makedirs(working_dir, exist_ok=True)

        kg_path = os.path.join(working_dir, "graph.pkl")
        openie_path = os.path.join(working_dir, "openie_results.json")
        chunk_map_path = os.path.join(working_dir, "chunk_map.json")

        # Initialize stores
        chunk_store = EmbeddingStore("chunk", working_dir)
        entity_store = EmbeddingStore("entity", working_dir)
        fact_store = EmbeddingStore("fact", working_dir)

        # Load passages
        passages = load_passages_from_corpus(self.corpus_path)
        print(f"[HippoRAG] Loaded {len(passages)} passages")

        # Check if graph exists and skip indexing
        if not force_reindex and os.path.exists(kg_path) and os.path.exists(openie_path):
            print(f"[HippoRAG] Loading existing graph from {kg_path}")
            kg = KnowledgeGraph.load(kg_path)
            chunk_to_passage = {}
            if os.path.exists(chunk_map_path):
                with open(chunk_map_path) as f:
                    data = json.load(f)
                    chunk_to_passage = {k: v for k, v in data["chunk_to_passage"].items()}
            chunks_list = []
            for p_idx, passage in enumerate(passages):
                text = passage.title + " " + passage.text
                chunk_texts = chunk_text(text)
                for c_text in chunk_texts:
                    c_hash = hash_text(c_text)
                    if c_hash in chunk_to_passage:
                        chunks_list.append((c_hash, c_text))
            with open(openie_path) as f:
                openie_data = json.load(f)
            return {
                "kg": kg,
                "chunk_store": chunk_store,
                "entity_store": entity_store,
                "fact_store": fact_store,
                "chunk_to_passage": chunk_to_passage,
                "chunks_list": chunks_list,
                "passages": passages,
                "ner_results": openie_data.get("ner", {}),
                "triple_results": openie_data.get("triple", {}),
            }

        # Step 1: Passage-level OpenIE
        ner_results = None
        triple_results = None

        if not force_reopenie and os.path.exists(openie_path):
            with open(openie_path) as f:
                openie_data = json.load(f)
                ner_results = openie_data.get("ner", {})
                triple_results = openie_data.get("triple", {})
            print(f"[HippoRAG] Loaded existing OpenIE results ({len(ner_results)} passages)")
        else:
            if force_reopenie:
                print("[HippoRAG] force_reopenie=True — re-running OpenIE")
            passages_text = {str(i): p.title + " " + p.text for i, p in enumerate(passages)}
            ner_results, triple_results = self.openie.batch_process_passages(passages_text)
            with open(openie_path, "w") as f:
                json.dump({"ner": ner_results, "triple": triple_results}, f, ensure_ascii=False)
            print(f"[HippoRAG] Saved OpenIE results to {openie_path}")

        # Step 2: Chunking + Chunk embedding
        print("[HippoRAG] Chunking passages...")
        chunks: List[ChunkTuple] = []
        chunk_to_passage: Dict[str, int] = {}
        for p_idx, passage in enumerate(passages):
            text = passage.title + " " + passage.text
            chunk_texts = chunk_text(text)
            for c_text in chunk_texts:
                c_hash = hash_text(c_text)
                chunks.append((c_hash, c_text))
                chunk_to_passage[c_hash] = p_idx

        chunks_list = chunks
        print(f"[HippoRAG] Generated {len(chunks)} chunks from {len(passages)} passages")

        with open(chunk_map_path, "w") as f:
            json.dump({"chunk_to_passage": chunk_to_passage}, f)

        print("[HippoRAG] Computing chunk embeddings...")
        chunk_texts_list = [t for _, t in chunks]
        chunk_embs = self.embedding_client.encode_batch(
            [DOC_PREFIX + t for t in chunk_texts_list]
        )
        chunk_store.insert(chunk_texts_list, chunk_embs)

        # Step 3: Entity from passage-level OpenIE
        print("[HippoRAG] Computing entity embeddings...")
        entity_names, entity_name_to_desc, entity_to_passages = build_entity_data_from_passages(
            passages, ner_results, triple_results
        )

        entity_embs = None
        if entity_names:
            entity_embs = self.embedding_client.encode_batch(entity_names)
            entity_store.insert(entity_names, entity_embs)

        print(f"[HippoRAG] Extracted {len(entity_names)} unique entities")

        # Step 4: Build fact data and compute fact embeddings
        print("[HippoRAG] Computing fact embeddings...")
        all_triples: List[list] = []
        fact_texts = []

        for pid, triples_list in triple_results.items():
            for triple in triples_list:
                if len(triple) == 3:
                    all_triples.append(triple)
                    fact_texts.append(json.dumps(triple, ensure_ascii=False))

        if fact_texts:
            fact_embs = self.embedding_client.encode_batch(fact_texts)
            fact_store.insert(fact_texts, fact_embs)

        print(f"[HippoRAG] Extracted {len(fact_texts)} facts")

        # Step 5: Build Knowledge Graph
        print("[HippoRAG] Building knowledge graph...")
        kg = self.graph_builder.build(
            entity_names=entity_names,
            entity_name_to_desc=entity_name_to_desc,
            entity_to_passages=entity_to_passages,
            entity_embs=entity_embs,
            all_triples=all_triples,
            passages=passages,
        )

        # Persist
        kg.save(kg_path)
        print("[HippoRAG] Indexing complete!")

        return {
            "kg": kg,
            "chunk_store": chunk_store,
            "entity_store": entity_store,
            "fact_store": fact_store,
            "chunk_to_passage": chunk_to_passage,
            "chunks_list": chunks_list,
            "passages": passages,
            "ner_results": ner_results,
            "triple_results": triple_results,
        }
