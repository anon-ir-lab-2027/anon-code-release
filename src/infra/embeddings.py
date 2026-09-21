from __future__ import annotations

import hashlib
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

# rough char-to-token ratio for truncation safety
# For vLLM all-MiniLM-L6-v2: max context length = 256 tokens
# Leave 6 token margin for DOC_PREFIX / QUERY_PREFIX
_MODEL_MAX_TOKENS = 256
_EMBEDDING_MAX_TOKENS = _MODEL_MAX_TOKENS - 16  # 240; cl100k != MiniLM WordPiece tokenizer, empirical safe limit


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    h = hashlib.md5(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") % (2**32)
    rng = np.random.RandomState(int(seed))
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class EmbeddingClient:
    def __init__(
        self,
        mode: str = "hash",
        base_url: str = "",
        model: str = "",
        api_key: str = "vllm",
        local_model: Optional[Union[str, Any]] = None,
    ) -> None:
        self._mode = mode
        self._fn: Callable[[str], np.ndarray]
        self._batch_fn: Optional[Callable[[List[str]], np.ndarray]] = None
        self._dim: Optional[int] = None

        if mode == "http":
            self._init_http(base_url, model, api_key)
        elif mode == "local":
            self._init_local(local_model)
        else:
            self._init_hash()

    def _init_hash(self) -> None:
        self._fn = _hash_embed
        self._batch_fn = lambda texts: np.stack([_hash_embed(t) for t in texts])
        self._dim = 384

    def _init_http(self, base_url: str, model: str, api_key: str) -> None:
        self._http_base = base_url.rstrip("/")
        self._http_model = model
        self._http_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            test_vec = self._http_embed_query("dimension test")
            self._dim = len(test_vec)
        except Exception:
            self._dim = 384

        self._fn = lambda t: self._http_embed_query(t)
        self._batch_fn = lambda texts: self._http_embed_batch(texts)

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to fit embedding model's token limit using tiktoken.

        Falls back to character-level truncation if tiktoken is unavailable.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= _EMBEDDING_MAX_TOKENS:
                return text
            return enc.decode(tokens[:_EMBEDDING_MAX_TOKENS])
        except Exception:
            # Fallback: assume ~2 chars per token
            max_chars = _EMBEDDING_MAX_TOKENS * 2
            if len(text) <= max_chars:
                return text
            return text[:max_chars]

    def _http_embed_query(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self._http_base}/embeddings",
            json={"model": self._http_model, "input": self._truncate(text)},
            headers=self._http_headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data["data"][0]["embedding"]
        return np.array(vec, dtype=np.float32)

    def _http_embed_batch(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        import requests
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            truncated = [self._truncate(t) for t in texts[i:i + batch_size]]
            resp = requests.post(
                f"{self._http_base}/embeddings",
                json={"model": self._http_model, "input": truncated},
                headers=self._http_headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            vecs = [d["embedding"] for d in data["data"]]
            all_vecs.extend(vecs)
            if len(all_vecs) % (batch_size * 5) == 0:
                print(f"[Embedding] {len(all_vecs)}/{len(texts)} encoded")
        return np.array(all_vecs, dtype=np.float32)

    def _init_local(self, local_model: Optional[Union[str, Any]]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed. Falling back to hash embedding."
            )
            self._init_hash()
            return

        if isinstance(local_model, str):
            model = SentenceTransformer(local_model)
        elif local_model is not None:
            model = local_model
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        self._dim = model.get_sentence_embedding_dimension()
        self._fn = lambda t: model.encode(t, convert_to_numpy=True)
        self._batch_fn = lambda texts: model.encode(texts, convert_to_numpy=True)

    def encode(self, text: str) -> np.ndarray:
        return self._fn(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if self._batch_fn:
            return self._batch_fn(texts)
        return np.stack([self._fn(t) for t in texts])

    @property
    def dim(self) -> int:
        if self._dim is None:
            return 64
        return self._dim


class RerankerClient:
    def __init__(
        self,
        mode: str = "light",
        base_url: str = "",
        model: str = "",
        api_key: str = "vllm",
        st_model: Any = None,
        threshold: float = 0.0,
    ) -> None:
        self._mode = mode
        self._threshold = threshold

        if mode == "http":
            self._init_http(base_url, model, api_key)
        elif mode == "st" and st_model is not None:
            self._init_st(st_model)
        elif mode == "bge":
            self._init_bge_local(model or "BAAI/bge-reranker-v2-m3")
        else:
            self._init_light()

    def _init_light(self) -> None:
        self._score_fn = _rerank_light

    def _init_http(self, base_url: str, model: str, api_key: str) -> None:
        base = base_url.rstrip("/")
        self._base_url = base
        self._model = model
        self._api_key = api_key
        self._use_openai_rerank = False
        try:
            r = requests.post(
                f"{base}/v1/rerank",
                json={"model": model, "query": "test", "documents": ["test"]},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            if r.status_code < 500:
                self._use_openai_rerank = True
        except Exception:
            pass

        if self._use_openai_rerank:
            self._score_fn = self._score_openai_rerank
        else:
            self._score_fn = self._score_native

    def _score_openai_rerank(self, query: str, passage: str) -> float:
        r = requests.post(
            f"{self._base_url}/v1/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": [passage],
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return self._threshold
        data = r.json()
        results = data.get("results", data.get("data", []))
        if results and isinstance(results, list):
            score = results[0].get("relevance_score", results[0].get("score", 0))
            return float(score)
        return self._threshold

    def _score_native(self, query: str, passage: str) -> float:
        r = requests.post(
            f"{self._base_url}/score",
            json={"text_1": [query], "text_2": [passage]},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return self._threshold
        data = r.json()
        scores = data.get("data", [])
        if scores:
            return float(scores[0].get("score", self._threshold))
        return self._threshold

    def _init_st(self, st_model: Any) -> None:
        self._st_model = st_model
        self._score_fn = self._score_st

    def _score_st(self, query: str, passage: str) -> float:
        emb_q = self._st_model.encode(query, convert_to_numpy=True)
        emb_p = self._st_model.encode(passage, convert_to_numpy=True)
        sim = np.dot(emb_q, emb_p) / (
            np.linalg.norm(emb_q) * np.linalg.norm(emb_p) + 1e-8
        )
        return float(max(0.0, sim))

    def _init_bge_local(self, model_name: str) -> None:
        try:
            from FlagEmbedding import FlagReranker
            self._bge_model = FlagReranker(model_name, use_fp16=True)
            self._score_fn = self._score_bge_local
        except ImportError:
            warnings.warn(
                "FlagEmbedding not installed. Falling back to light reranker."
            )
            self._init_light()

    def _score_bge_local(self, query: str, passage: str) -> float:
        scores = self._bge_model.compute_score([[query, passage]])
        if isinstance(scores, list):
            return float(scores[0])
        return float(scores)

    def score(self, context: str, passage_text: str) -> float:
        return self._score_fn(context, passage_text)

    def score_batch(self, context: str, passages: List[str]) -> List[float]:
        # 如果后端支持 native score，改为 batch 发送减少请求次数
        if self._score_fn.__name__ == "_score_native" or self._score_fn.__name__ == "_score_openai_rerank":
            r = requests.post(
                f"{self._base_url}/score",
                json={"text_1": [context] * len(passages), "text_2": passages},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120,
            )
            if r.status_code == 200:
                data = r.json()
                scores = data.get("data", [])
                return [float(s.get("score", self._threshold)) for s in scores]
        # fallback: 逐条调用
        return [self._score_fn(context, p) for p in passages]

    def rerank(
        self,
        query: str,
        passages: List[Tuple[Any, float]],
        top_k: int,
    ) -> List[Tuple[Any, float]]:
        texts = [p.text for p, _ in passages]
        scores = self.score_batch(query, texts)
        reranked = sorted(
            zip([p for p, _ in passages], scores),
            key=lambda x: -x[1],
        )
        return reranked[:top_k]

    @property
    def threshold(self) -> float:
        return self._threshold


def _rerank_light(context: str, passage_text: str) -> float:
    ctx_tokens = set(context.lower().split())
    psg_tokens = set(passage_text.lower().split())
    if not ctx_tokens:
        return 0.0
    overlap = ctx_tokens & psg_tokens
    return len(overlap) / len(ctx_tokens)


def make_embedding_client(
    embed_backend: str = "hash",
    embedding_base_url: str = "",
    embedding_model: str = "",
    embedding_api_key: str = "vllm",
) -> EmbeddingClient:
    return EmbeddingClient(
        mode=embed_backend,
        base_url=embedding_base_url,
        model=embedding_model,
        api_key=embedding_api_key,
    )


def make_reranker_client(
    reranker_mode: str = "light",
    reranker_base_url: str = "",
    reranker_model: str = "",
    reranker_api_key: str = "vllm",
    st_model: Any = None,
    tau_rerank: float = 0.0,
) -> RerankerClient:
    return RerankerClient(
        mode=reranker_mode,
        base_url=reranker_base_url,
        model=reranker_model,
        api_key=reranker_api_key,
        st_model=st_model,
        threshold=tau_rerank,
    )
