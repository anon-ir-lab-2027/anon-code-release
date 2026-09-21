from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from infra.embeddings import _hash_embed, EmbeddingClient


@dataclass
class Passage:
    id: str
    title: str
    text: str
    source: Optional[str] = None


class CorpusRetriever:
    def __init__(
        self,
        passages: List[Passage],
        mode: str = "keyword",
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        st_model: Optional[Any] = None,
        embedding_client: Any = None,
        query_prefix: str = "",
        doc_prefix: str = "",
        _embeddings: Optional[np.ndarray] = None,
        _bm25: Any = None,
        _bm25_tokenized: Optional[List[List[str]]] = None,
    ) -> None:
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._passages = list(passages)
        self._mode = mode

        if embedding_client is not None:
            self._embed_fn = embedding_client.encode
            self._embed_batch_fn: Optional[Callable[[List[str]], np.ndarray]] = (
                embedding_client.encode_batch
            )
        elif st_model is not None:
            self._embed_fn = lambda t: st_model.encode(t, convert_to_numpy=True)
            self._embed_batch_fn = lambda texts: st_model.encode(
                texts, convert_to_numpy=True
            )
        elif embed_fn is not None:
            self._embed_fn = embed_fn
            self._embed_batch_fn = None
        else:
            self._embed_fn = lambda t: _hash_embed(t)
            self._embed_batch_fn = None

        self._embeddings: Optional[np.ndarray] = None
        if _embeddings is not None:
            self._embeddings = _embeddings
        elif mode in ("vector", "hybrid"):
            texts = [self._doc_prefix + p.title + " " + p.text for p in self._passages]
            if self._embed_batch_fn is not None:
                self._embeddings = self._embed_batch_fn(texts)
            else:
                self._embeddings = np.stack([self._embed_fn(t) for t in texts])

        if _bm25 is not None or _bm25_tokenized is not None:
            self._bm25 = _bm25
            self._bm25_tokenized = _bm25_tokenized if _bm25_tokenized is not None else []
        elif mode in ("keyword", "hybrid"):
            self._bm25 = None
            self._bm25_tokenized: List[List[str]] = []
            self._init_bm25()
        else:
            self._bm25 = None
            self._bm25_tokenized: List[List[str]] = []

    def _init_bm25(self) -> None:
        texts = [p.title + " " + p.text for p in self._passages]
        self._bm25_tokenized = [t.lower().split() for t in texts]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._bm25_tokenized)
        except ImportError:
            self._bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Passage, float]]:
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            max_score = max(scores) if scores.size > 0 else 1.0
            idxs = np.argsort(-scores)[:top_k]
            return [
                (self._passages[i], float(scores[i]) / max(max_score, 1e-8))
                for i in idxs if scores[i] > 0
            ]
        return self._keyword_search(query, top_k)

    def search(
        self,
        query: str,
        top_k: int = 20,
        bm_top_k: Optional[int] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[Passage, float]]:
        _bm_top_k = bm_top_k if bm_top_k is not None else top_k * 2

        if self._mode == "hybrid" and self._embeddings is not None:
            vec_results = self._vector_search(query, top_k * 2, query_embedding)
            bm25_results = self._bm25_search(query, _bm_top_k)
            if _bm_top_k == top_k:
                return self._merge_hybrid_results(vec_results, bm25_results, top_k)
            merged = self._merge_hybrid_results(vec_results, bm25_results, _bm_top_k)
            return merged[:_bm_top_k]
        if self._mode == "vector" and self._embeddings is not None:
            return self._vector_search(query, top_k, query_embedding)
        return self._bm25_search(query, top_k)

    def _merge_hybrid_results(
        self,
        vec_results: List[Tuple[Passage, float]],
        bm25_results: List[Tuple[Passage, float]],
        top_k: int,
    ) -> List[Tuple[Passage, float]]:
        merged: Dict[str, Tuple[Passage, float]] = {}
        for p, score in vec_results:
            merged[p.id] = (p, 0.3 * score)
        for p, score in bm25_results:
            if p.id in merged:
                _, existing = merged[p.id]
                merged[p.id] = (p, existing + 0.7 * score)
            else:
                merged[p.id] = (p, 0.7 * score)
        results = sorted(merged.values(), key=lambda x: -x[1])
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Passage, float]]:
        tokens = set(query.lower().split())
        scored: List[Tuple[Passage, float]] = []
        for p in self._passages:
            text_lower = (p.title + " " + p.text).lower()
            overlap = sum(1 for t in tokens if t in text_lower)
            if overlap > 0:
                score = overlap / max(len(tokens), 1)
                scored.append((p, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _vector_search(
        self,
        query: str,
        top_k: int,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[Passage, float]]:
        if query_embedding is not None:
            q_emb = query_embedding
        else:
            q_emb = self._embed_fn(self._query_prefix + query)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        scores = np.dot(self._embeddings, q_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * q_norm + 1e-8
        )
        idxs = np.argsort(-scores)[:top_k]
        return [(self._passages[i], float(scores[i])) for i in idxs if scores[i] > 0]

    def __len__(self) -> int:
        return len(self._passages)

    def save(self, path: str) -> None:
        data: Dict[str, Any] = {
            "passages": self._passages,
            "mode": self._mode,
        }
        if self._mode in ("keyword", "hybrid"):
            data["bm25"] = self._bm25
            data["bm25_tokenized"] = self._bm25_tokenized
        if self._mode in ("vector", "hybrid") and self._embeddings is not None:
            data["embeddings"] = self._embeddings
            data["embed_dim"] = self._embeddings.shape[1]
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> CorpusRetriever:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            passages=data.get("passages", []),
            mode=data.get("mode", "keyword"),
            _embeddings=data.get("embeddings"),
            _bm25=data.get("bm25"),
            _bm25_tokenized=data.get("bm25_tokenized"),
        )


def load_passages_from_corpus(path: str) -> List[Passage]:
    import json
    with open(path, "r") as f:
        raw = json.load(f)
    passages: List[Passage] = []
    if isinstance(raw, dict):
        for title, sentences in raw.items():
            if isinstance(sentences, list):
                text = " ".join(sentences)
            else:
                text = str(sentences)
            passages.append(Passage(id=title, title=title, text=text))
    elif isinstance(raw, list):
        for item in raw:
            title = item.get("title", "")
            text = item.get("text", "")
            pid = item.get("id", title)
            passages.append(Passage(id=pid, title=title, text=text))
    else:
        raise ValueError(f"Unexpected corpus format: {type(raw)}")
    return passages
