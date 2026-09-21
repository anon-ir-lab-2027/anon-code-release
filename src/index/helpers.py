"""
Helper functions for chunking, hashing, and entity data construction.

Uses tiktoken (cl100k_base) for token-aware chunking.
This aligns chunk boundaries with what the embedding model and reranker see.
"""

import hashlib
from typing import Dict, List, Tuple

from infra.base import ChunkTuple


def hash_text(text: str) -> str:
    """Return MD5 hex digest of text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Token-aware text chunking using tiktoken (cl100k_base).

    Falls back to character-level chunking if tiktoken is unavailable
    or fails (e.g., SSL certificate issue on WSL).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(enc.decode(chunk_tokens).strip())
            if i + chunk_size >= len(tokens):
                break
        return chunks
    except Exception:
        # Fallback to character-level chunking
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size].strip()
            chunks.append(chunk_text)
            if i + chunk_size >= len(text):
                break
        return chunks


def build_entity_data(
    chunks: List[ChunkTuple],
    ner_results: Dict,
    triple_results: Dict,
) -> Tuple[List[str], Dict[str, str], Dict[str, List[Tuple[str, str]]]]:
    """Build entity names, descriptions, and chunk mappings from NER/Triple results.

    Only includes entities that appear in triple results (as subject or object).

    Returns:
        (entity_names, entity_name_to_desc, entity_to_chunks)
    """
    entity_name_to_desc: Dict[str, str] = {}
    entity_to_chunks: Dict[str, List[Tuple[str, str]]] = {}

    for c_hash, c_text in chunks:
        if c_hash not in ner_results:
            continue
        for entity in ner_results[c_hash]:
            name = entity.get("name", "").strip()
            desc = entity.get("description", "").strip()
            if not name:
                continue
            if name not in entity_name_to_desc:
                entity_name_to_desc[name] = desc
            if name not in entity_to_chunks:
                entity_to_chunks[name] = []
            entity_to_chunks[name].append((c_hash, desc))

    # Step 2: Include triple entities not already covered by NER
    for c_hash, c_text in chunks:
        if c_hash not in triple_results:
            continue
        for triple in triple_results[c_hash]:
            if len(triple) != 3:
                continue
            subj, pred, obj = triple
            for name in [subj, obj]:
                name_stripped = name.strip()
                if not name_stripped or name_stripped in entity_name_to_desc:
                    continue
                # Use triple text as fallback description
                desc = f"{pred}: {obj if name_stripped == subj else subj}"
                entity_name_to_desc[name_stripped] = desc
                if name_stripped not in entity_to_chunks:
                    entity_to_chunks[name_stripped] = []
                entity_to_chunks[name_stripped].append((c_hash, desc))

    entity_names = list(entity_name_to_desc.keys())
    return entity_names, entity_name_to_desc, entity_to_chunks


def build_entity_data_from_passages(
    passages: list,
    ner_results: Dict[str, List[Dict]],
    triple_results: Dict[str, List[List[str]]],
) -> tuple:
    """从 passage 级 NER/Triple 结果构建实体数据。

    Args:
        passages: List[Passage]（使用 passage index 作为 id）
        ner_results: {passage_idx_str: [{"name", "description"}]}
        triple_results: {passage_idx_str: [["subj","pred","obj"]]}

    Returns:
        (entity_names, entity_name_to_desc, entity_to_passages)
        entity_to_passages: {entity_name: [(passage_index, desc)]}
    """
    entity_name_to_desc: Dict[str, str] = {}
    entity_to_passages: Dict[str, List[tuple]] = {}

    for p_idx, passage in enumerate(passages):
        pid = str(p_idx)
        for entity in ner_results.get(pid, []):
            name = entity.get("name", "").strip()
            desc = entity.get("description", "").strip()
            if not name:
                continue
            if name not in entity_name_to_desc:
                entity_name_to_desc[name] = desc
            if name not in entity_to_passages:
                entity_to_passages[name] = []
            entity_to_passages[name].append((p_idx, desc))

        for triple in triple_results.get(pid, []):
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            subj, pred, obj = triple
            for name in [subj, obj]:
                if not isinstance(name, str):
                    continue
                name_stripped = name.strip()
                if not name_stripped or name_stripped in entity_name_to_desc:
                    continue
                desc = f"{pred}: {obj if name_stripped == subj else subj}"
                entity_name_to_desc[name_stripped] = desc
                if name_stripped not in entity_to_passages:
                    entity_to_passages[name_stripped] = []
                entity_to_passages[name_stripped].append((p_idx, desc))

    entity_names = list(entity_name_to_desc.keys())
    return entity_names, entity_name_to_desc, entity_to_passages
