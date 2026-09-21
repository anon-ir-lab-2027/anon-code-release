"""
Personalized PageRank (PPR) — raw igraph wrapper.

Only provides the low-level run_ppr. Each retriever implements its own
seed-weight assembly and passage ranking logic.
"""

import numpy as np

from infra.knowledge_graph import KnowledgeGraph


def run_ppr(kg: KnowledgeGraph,
            seed_weights: np.ndarray,
            damping: float = 0.5) -> np.ndarray:
    """Run personalized PageRank on a KnowledgeGraph.

    Args:
        kg: the KnowledgeGraph instance
        seed_weights: 1D array of length vcount(). Non-zero values mark seed nodes.
        damping: restart probability (teleportation factor)

    Returns:
        PPR scores for each node (same order as seed_weights)
    """
    seed_weights = np.where(np.isnan(seed_weights) | (seed_weights < 0), 0, seed_weights)
    scores = kg.graph.personalized_pagerank(
        vertices=range(kg.graph.vcount()),
        damping=damping,
        directed=kg.graph.is_directed(),
        weights="weight",
        reset=seed_weights,
        implementation="prpack"
    )
    return np.array(scores)
