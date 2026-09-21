"""
MCS-RerankPath Retriever: MCS + reranker path scoring + PPR seed expansion.

Pipeline:
  S1. Fact store top-15 -> fact triples
  S2. Build minimum connected subgraph from S1 triple entities -> MST entity chains
  S3. (removed)
  S4. Build reasoning chains -> reranker -> top chains
  S5. Extract seed entities from top chains
  S6. PPR on global graph with DPR blending
  S7. Sort passages -> return top_k
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_openai import ChatOpenAI

from infra.knowledge_graph import KnowledgeGraph
from infra.embedding_store import EmbeddingStore
from infra.embeddings import EmbeddingClient, RerankerClient
from infra.retrieval import Passage
from infra.base import (
    QUERY_PREFIX,
    PPR_DAMPING, PASSAGE_NODE_WEIGHT,
    IndexResult,
)
from graph.ppr import run_ppr as _run_ppr
from graph.search import mcs_connected_subgraph, mcs_graph_search
from index.runner import HippoRAGIndexer


class MCSRerankPathRetriever:
    """MCS-RerankPath: reasoning chain reranking retrieval."""

    def __init__(
        self,
        index_result: IndexResult | None = None,
        indexer: HippoRAGIndexer | None = None,
        embedding_client: EmbeddingClient | None = None,
        reranker_client: RerankerClient | None = None,
        llm: ChatOpenAI | None = None,
        ablation_mode: str = "none",
    ):
        if index_result is not None:
            self._init_from_result(index_result)
        elif indexer is not None:
            self._init_from_indexer(indexer)
        else:
            raise ValueError("Either index_result or indexer must be provided")

        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.llm = llm
        self.ablation_mode = ablation_mode
        assert ablation_mode in ("none", "no_reranker", "no_mcs", "rerank_only", "no_ppr", "no_dpr_blend", "dpr_only"), \
            f"Unknown ablation_mode: {ablation_mode}"

    def _init_from_result(self, index_result: IndexResult):
        self.kg: KnowledgeGraph = index_result["kg"]
        self.chunk_store: EmbeddingStore = index_result["chunk_store"]
        self.entity_store: EmbeddingStore = index_result["entity_store"]
        self.fact_store: EmbeddingStore = index_result["fact_store"]
        self._chunk_to_passage: Dict[str, int] = index_result.get("chunk_to_passage", {})
        self._chunks_list: List[Tuple[str, str]] = index_result.get("chunks_list", [])
        self.passages: List[Passage] = index_result["passages"]
        self.ready = True

    def _init_from_indexer(self, indexer: HippoRAGIndexer):
        index_result = indexer.index()
        self._init_from_result(index_result)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _entity_readable_name(kg: KnowledgeGraph, vidx: int) -> str:
        """Get human-readable entity name from graph vertex index."""
        node_name = kg.graph.vs[vidx]["name"]
        meta = kg.entity_meta.get(node_name, {})
        if isinstance(meta, dict):
            return meta.get("name", node_name[:40])
        return node_name[:40]

    def _build_chunk_text_to_passage_map(self) -> Dict[str, int]:
        """Build chunk_text -> passage_idx lookup (O(1) instead of O(n))."""
        if not hasattr(self, '__chunk_text_to_passage'):
            m = {}
            for c_hash, c_text in self._chunks_list:
                if c_hash in self._chunk_to_passage:
                    m[c_text] = self._chunk_to_passage[c_hash]
            self.__chunk_text_to_passage = m
        return self.__chunk_text_to_passage

    # ── PPR (owned by MCS-RP) ──────────────────────────────────────────────

    def _mcsrp_ppr(
        self,
        seed_entity_names: List[str],
        seed_entity_scores: List[float],
        passage_dpr_scores: np.ndarray | None = None,
        passage_node_weight: float = PASSAGE_NODE_WEIGHT,
        damping: float = PPR_DAMPING,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MCS-RP's PPR: seed entities -> node weights + DPR blending -> run_ppr.

        Returns (sorted_passage_indices, sorted_passage_scores).
        """
        n = self.kg.graph.vcount()
        node_weights = np.zeros(n)

        for ent_name, score in zip(seed_entity_names, seed_entity_scores):
            ent_key = self.kg.get_entity_hash(ent_name)
            if ent_key in self.kg.node_name_to_idx:
                idx = self.kg.node_name_to_idx[ent_key]
                node_weights[idx] += score

        if passage_dpr_scores is not None and passage_node_weight > 0:
            for i, passage_idx in enumerate(self.kg.passage_node_idxs):
                node_weights[passage_idx] += passage_dpr_scores[i] * passage_node_weight

        if np.sum(node_weights) == 0:
            if passage_dpr_scores is not None and np.sum(passage_dpr_scores) > 0:
                sorted_idx = np.argsort(-passage_dpr_scores)
                return sorted_idx, passage_dpr_scores[sorted_idx]
            passage_scores = np.ones(len(self.kg.passage_node_idxs))
            return np.argsort(-passage_scores), passage_scores

        ppr_scores = _run_ppr(self.kg, node_weights, damping=damping)
        passage_scores = ppr_scores[self.kg.passage_node_idxs]

        sorted_indices = np.argsort(-passage_scores)
        sorted_scores = passage_scores[sorted_indices]
        return sorted_indices, sorted_scores

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        inner_depth: int = 3,
        inner_width: int = 8,
        ppr_damping: float = 0.4,
        inner_bidirectional: bool = True,
        top_chains: int = 10,
        **kwargs,
    ) -> List[Tuple[Passage, float]]:
        """MCS-RerankPath retrieval pipeline (S1-S7).

        S1: Fact store top-15 -> fact triples
        S2: Minimum connected subgraph from S1 entities -> MST entity chains
        S3: (removed)
        S4: Build reasoning chains -> reranker -> top chains (keeps reranker scores)
        S5: Extract seed entities from top chains
        S6: PPR on global graph with DPR weak prior
        S7: Sort passages -> return top_k
        """
        if not self.ready:
            raise RuntimeError("MCSRerankPath retriever not indexed.")
        if self.embedding_client is None:
            raise RuntimeError("embedding_client is required")

        query_vec = self.embedding_client.encode(QUERY_PREFIX + query)

        # ── Ablation: dpr_only — skip all graph-based retrieval ──────────
        if self.ablation_mode == "dpr_only":
            return self._dpr_retrieval(query, top_k)

        # ── S1: Fact store — direct top-15 search (no reranker) ────────────
        if self.ablation_mode != "rerank_only":
            fact_results = self.fact_store.search(query_vec, top_k=15)
        else:
            # rerank_only: fact store top-20 -> reranker -> top-10
            fact_results = self.fact_store.search(query_vec, top_k=20)
            if fact_results and self.reranker_client:
                try:
                    fact_texts = [f[0] for f in fact_results]
                    rerank_scores = self.reranker_client.score_batch(query, fact_texts)
                    reranked = sorted(zip(fact_results, rerank_scores), key=lambda x: -x[1])
                    fact_results = [(f[0], f[1], f[2]) for f, _ in reranked[:10]]
                except Exception:
                    fact_results = fact_results[:15]

        triple_seed_entities: List[str] = []
        for (fact_str, _, _) in fact_results:
            try:
                triple = json.loads(fact_str)
                if len(triple) >= 3:
                    triple_seed_entities.extend([triple[0], triple[2]])
            except (json.JSONDecodeError, IndexError):
                pass

        # 去重
        triple_seed_entities = list(dict.fromkeys(triple_seed_entities))

        if not triple_seed_entities:
            return self._dpr_retrieval(query, top_k)

        # ── Ablation: no_mcs / rerank_only — skip S2-S5, feed seeds directly to PPR ────
        if self.ablation_mode in ("no_mcs", "rerank_only"):
            seed_scores = [0.5] * len(triple_seed_entities)
            dpr_results = self._dpr_retrieval(query, top_k=50)
            dpr_scores_array = self._build_dpr_scores_array(dpr_results)
            sorted_idx, sorted_scores = self._mcsrp_ppr(
                triple_seed_entities, seed_scores,
                passage_dpr_scores=dpr_scores_array,
                passage_node_weight=PASSAGE_NODE_WEIGHT,
                damping=ppr_damping,
            )
            result = self._map_nodes_to_passages(sorted_idx, sorted_scores, top_k)
            if not result:
                return self._dpr_retrieval(query, top_k)
            return result

        # ── S2: MCS connected subgraph ────────────────────────────────────
        s2_entity_chains = mcs_connected_subgraph(
            self.kg, triple_seed_entities,
            max_depth=inner_depth, max_width=inner_width,
            bidirectional=inner_bidirectional,
        )
        if not s2_entity_chains:
            return self._dpr_retrieval(query, top_k)

        # S3 removed (no forward explore)

        # ── S4: Build reasoning chains + reranker ─────────────────────────
        all_entity_paths: List[List[int]] = [
            chain for chain in s2_entity_chains if len(chain) >= 2
        ]

        chain_info: List[Tuple[str, List[int]]] = []
        for path_ents in all_entity_paths:
            chain_text = _build_reasoning_chain(self.kg, path_ents)
            if chain_text:
                chain_info.append((chain_text, path_ents))

        if not chain_info:
            return self._dpr_retrieval(query, top_k)

        chain_texts = [c[0] for c in chain_info]
        # ── Ablation: no_reranker — skip reranker, use all chains equally ──
        if self.ablation_mode == "no_reranker":
            chain_scores = [1.0 / (1.0 + i) for i in range(len(chain_texts))]
        else:
            try:
                if self.reranker_client:
                    chain_scores = self.reranker_client.score_batch(query, chain_texts)
                else:
                    chain_scores = [0.5] * len(chain_texts)
            except Exception:
                chain_scores = [0.5] * len(chain_texts)

        # Keep reranker scores directly — no reciprocal ranking override
        # Reranker already gives meaningful relative scores
        scored_chains = sorted(zip(chain_info, chain_scores), key=lambda x: -x[1])
        top_chains_list = scored_chains[:top_chains]

        # ── S5: Extract entities from top chains ──────────────────────────
        entity_chain_scores: Dict[str, float] = {}
        for (chain_text, path_ents), chain_score in top_chains_list:
            for ent_vidx in path_ents:
                readable_name = self._entity_readable_name(self.kg, ent_vidx)
                entity_chain_scores[readable_name] = max(
                    entity_chain_scores.get(readable_name, 0.0), chain_score
                )

        if not entity_chain_scores:
            return self._dpr_retrieval(query, top_k)

        seed_entity_names_s5 = list(entity_chain_scores.keys())
        seed_entity_scores_s5 = list(entity_chain_scores.values())

        # Ablation: no_reranker — apply the same min-max norm as other modes
        #   (but the chain scores are 1/(1+i) so normalization is needed for consistent
        #    seed weight distribution compared to reranker-scored chains)
        scores_arr = np.array(list(entity_chain_scores.values()))
        score_min, score_max = scores_arr.min(), scores_arr.max()
        if score_max > score_min:
            seed_entity_scores_s5 = list((scores_arr - score_min) / (score_max - score_min))

        # ── Ablation: no_ppr — skip PPR, use BFS graph search instead ───────
        #   Instead of PPR global diffusion, use mcs_graph_search which does
        #   deterministic BFS from seed entities over graph edges to find passages.
        #   This is more robust than entity_node_to_chunk lookup because it
        #   handles entity hash discrepancies and supports multi-hop paths.
        if self.ablation_mode == "no_ppr":
            dpr_results = self._dpr_retrieval(query, top_k=50)
            dpr_scores_array = self._build_dpr_scores_array(dpr_results)
            sorted_idx, sorted_scores = mcs_graph_search(
                self.kg,
                seed_entity_names_s5,
                seed_entity_scores_s5,
                dpr_scores_array,
                max_depth=3,
                max_width=20,
                bidirectional=True,
            )
            result = self._map_nodes_to_passages(sorted_idx, sorted_scores, top_k)
            if not result:
                return self._dpr_retrieval(query, top_k)
            return result

        # ── S6: DPR + PPR ────────────────────────────────────────────────
        dpr_results = self._dpr_retrieval(query, top_k=50)
        dpr_scores_array = self._build_dpr_scores_array(dpr_results)

        # ── Ablation: no_dpr_blend — PPR without DPR weak prior ─────────
        dpr_blend_weight = 0.0 if self.ablation_mode == "no_dpr_blend" else PASSAGE_NODE_WEIGHT

        sorted_idx, sorted_scores = self._mcsrp_ppr(
            seed_entity_names_s5, seed_entity_scores_s5,
            passage_dpr_scores=dpr_scores_array,
            passage_node_weight=dpr_blend_weight,
            damping=ppr_damping,
        )

        # ── S7: Map to passages ──────────────────────────────────────────
        result = self._map_nodes_to_passages(sorted_idx, sorted_scores, top_k)
        if not result:
            return self._dpr_retrieval(query, top_k)
        return result

    def _dpr_retrieval(self, query: str, top_k: int) -> List[Tuple[Passage, float]]:
        """Dense passage retrieval (fallback / DPR baseline)."""
        if self.embedding_client is None:
            return []
        query_vec = self.embedding_client.encode(QUERY_PREFIX + query)
        results = self.chunk_store.search(query_vec, top_k=top_k * 3)

        chunk_text_to_passage = self._build_chunk_text_to_passage_map()

        seen: set = set()
        final = []
        for chunk_text, score, _ in results:
            pass_idx = chunk_text_to_passage.get(chunk_text)
            if pass_idx is not None and pass_idx not in seen:
                final.append((self.passages[pass_idx], float(score)))
                seen.add(pass_idx)
            if len(final) >= top_k:
                break

        return final[:top_k]

    def _build_dpr_scores_array(
        self, dpr_results: List[Tuple[Passage, float]]
    ) -> np.ndarray:
        """Build DPR score array aligned with kg.passage_node_idxs."""
        n_passage_nodes = len(self.kg.passage_node_idxs)
        dpr_scores = np.zeros(n_passage_nodes)

        if not dpr_results:
            return dpr_scores

        passage_score_map: Dict[int, float] = {}
        for passage, score in dpr_results:
            for p_idx in range(len(self.passages)):
                if (
                    self.passages[p_idx].title == passage.title
                    and self.passages[p_idx].text == passage.text
                ):
                    passage_score_map[p_idx] = score
                    break

        for i, vidx in enumerate(self.kg.passage_node_idxs):
            node_name = self.kg.graph.vs[vidx]["name"]
            pass_idx = int(node_name.replace("passage-", ""))
            if pass_idx in passage_score_map:
                dpr_scores[i] = passage_score_map[pass_idx]

        return dpr_scores

    def _map_nodes_to_passages(
        self,
        sorted_idx: np.ndarray,
        sorted_scores: np.ndarray,
        top_k: int,
    ) -> List[Tuple[Passage, float]]:
        """Map sorted passage-node indices to Passage objects."""
        result: List[Tuple[Passage, float]] = []
        seen: set = set()

        for pos, i in enumerate(sorted_idx[: top_k * 2]):
            if i >= len(self.kg.passage_node_idxs):
                continue
            vidx = self.kg.passage_node_idxs[i]
            node_name = self.kg.graph.vs[vidx]["name"]
            pass_idx = int(node_name.replace("passage-", ""))

            if pass_idx not in seen and pass_idx < len(self.passages):
                result.append((self.passages[pass_idx], float(sorted_scores[pos])))
                seen.add(pass_idx)
            if len(result) >= top_k:
                break

        return result


def _get_edge_label(kg: KnowledgeGraph, src_vidx: int, dst_vidx: int) -> Optional[str]:
    """Get edge label between two vertices (checks both directions)."""
    try:
        eids = kg.graph.get_eids([(src_vidx, dst_vidx)], directed=True, error=False)
    except TypeError:
        eids = []
    if eids and eids[0] >= 0:
        try:
            return kg.graph.es[eids[0]]["label"]
        except (KeyError, IndexError):
            pass
    try:
        rev_eids = kg.graph.get_eids([(dst_vidx, src_vidx)], directed=True, error=False)
    except TypeError:
        return None
    if rev_eids and rev_eids[0] >= 0:
        try:
            return kg.graph.es[rev_eids[0]]["label"]
        except (KeyError, IndexError):
            return None
    return None


def _build_reasoning_chain(kg: KnowledgeGraph, path_ent_vidxs: List[int]) -> Optional[str]:
    """Build a human-readable reasoning chain from entity vertex indices."""
    if len(path_ent_vidxs) < 2:
        return None

    def _ent_readable(vidx: int) -> str:
        node_name = kg.graph.vs[vidx]["name"]
        meta = kg.entity_meta.get(node_name, {})
        if isinstance(meta, dict):
            return meta.get("name", node_name[:40])
        return node_name[:40]

    parts = []
    for i in range(len(path_ent_vidxs) - 1):
        src = path_ent_vidxs[i]
        dst = path_ent_vidxs[i + 1]
        src_name = _ent_readable(src)
        dst_name = _ent_readable(dst)
        pred = _get_edge_label(kg, src, dst)
        if pred:
            parts.append(f"({src_name} {pred} {dst_name})")
        else:
            parts.append(f"({src_name} -> {dst_name})")

    return "".join(parts)
