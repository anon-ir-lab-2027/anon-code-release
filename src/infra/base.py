"""
Infrastructure base types and constants for HippoRAG.
"""
from typing import Dict, List, Tuple

QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
FACT_RETRIEVE_TOP_K = 10
FACT_RERANK_TOP_K = 5
SYNONYM_TOP_K = 10
SYNONYM_THRESHOLD = 0.85
PPR_DAMPING = 0.4
PASSAGE_NODE_WEIGHT = 0.1  # DPR blending 权重，使 DPR 分数以 0.1 倍加入 PPR node_weights

ChunkTuple = Tuple[str, str]  # (chunk_hash, chunk_text)

# Passage 级 OpenIE 结果类型
PassageNerResult = Dict[str, List[Dict]]  # passage_id -> [{"name", "description"}]
PassageTripleResult = Dict[str, List[List[str]]]  # passage_id -> [[subj, pred, obj]]

IndexResult = Dict  # keys: kg, chunk_store, entity_store, fact_store, chunk_to_passage, chunks_list, passages, passage_to_chunks
