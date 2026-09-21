"""
Graph construction from OpenIE results and entity data.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from infra.knowledge_graph import KnowledgeGraph


class GraphBuilder:
    """Builds a KnowledgeGraph from OpenIE results + entity lists."""

    def __init__(self, kg_class=None):
        self._kg_class = kg_class or KnowledgeGraph

    def build(
        self,
        entity_names: List[str],
        entity_name_to_desc: Dict[str, str],
        entity_to_passages: Dict[str, List[Tuple[int, str]]],
        entity_embs: np.ndarray | None,
        all_triples: List[list],
        passages: list = None,
    ) -> KnowledgeGraph:
        """Complete graph construction pipeline.

        Uses DBSCAN (eps=0.12, min_samples=2, cosine) for entity alias merging.
        Synonymy edges are intentionally excluded — ablation showed no benefit
        over DBSCAN alone (see docs/实验记录.md 第19章).

        Args:
            entity_names: list of entity names
            entity_name_to_desc: entity name -> description
            entity_to_passages: entity name -> list of (passage_idx, description)
            entity_embs: optional entity embeddings for DBSCAN alias merging
            all_triples: list of [subj, pred, obj] triples
            passages: optional passages list for filling passage_meta

        Returns:
            Fully built KnowledgeGraph (directed=True)
        """
        kg = self._kg_class(directed=True)

        # Step 1: DBSCAN alias merging — semantic similarity based entity dedup
        kg._name_map = {}
        if entity_embs is not None and len(entity_names) > 1:
            db = DBSCAN(eps=0.12, min_samples=2, metric="cosine", n_jobs=-1)
            labels = db.fit_predict(entity_embs)

            name_to_canonical_hash: Dict[str, str] = {}
            for i, name in enumerate(entity_names):
                label = labels[i]
                if label == -1:
                    ch = KnowledgeGraph.entity_hash(name)
                    name_to_canonical_hash[name] = ch
                else:
                    cluster_indices = np.where(labels == label)[0]
                    cluster_names = [entity_names[i] for i in cluster_indices]
                    longest_name = max(cluster_names, key=len)
                    name_to_canonical_hash[name] = name_to_canonical_hash.get(
                        longest_name, KnowledgeGraph.entity_hash(longest_name)
                    )

            kg._name_map = name_to_canonical_hash

            num_unique = len(set(name_to_canonical_hash.values()))
            if num_unique < len(entity_names):
                print(f"  [Builder] Alias merging: {len(entity_names)} names -> {num_unique} unique canonical entities")

        # Add passage edges: entity <-> passage (by passage_idx)
        for name in entity_names:
            for p_idx, _ in entity_to_passages.get(name, []):
                kg.add_passage_edges(
                    [(name, entity_name_to_desc.get(name, ""))],
                    p_idx,
                )

        # Add fact edges: entity -> entity (directed, subj->obj only)
        for triple in all_triples:
            kg.add_fact_edges([triple])

        # Fill passage_meta
        if passages:
            for p_idx, p in enumerate(passages):
                p_key = kg.passage_hash(p_idx)
                kg.passage_meta[p_key] = f"{p.title} {p.text}"

        # Finalize igraph
        kg.build()

        return kg
