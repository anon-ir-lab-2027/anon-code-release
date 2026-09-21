"""
Persistent embedding storage backed by npz + JSON metadata.

Each EmbeddingStore manages one namespace (chunk/entity/fact) and
can save/load its state to/from a cache directory.
"""

import json
import os
import numpy as np
from typing import List, Optional, Tuple


class EmbeddingStore:
    """A vector store with text and metadata, persisted to disk."""

    def __init__(self, name: str, cache_dir: str):
        """
        Args:
            name: namespace name (e.g., "chunk", "entity", "fact")
            cache_dir: directory to store cache files
        """
        self.name = name
        self.cache_dir = os.path.join(cache_dir, name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # nxd matrix
        self.metadatas: List[dict] = []
        self.text_to_idx: dict = {}  # text hash -> index (for dedup)
        self._load()

    def _text_hash(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load(self):
        """Load from disk cache if exists."""
        texts_path = os.path.join(self.cache_dir, "texts.npy")
        embs_path = os.path.join(self.cache_dir, "embeddings.npy")
        meta_path = os.path.join(self.cache_dir, "metadata.json")
        idx_path = os.path.join(self.cache_dir, "text_to_idx.json")

        if os.path.exists(texts_path) and os.path.exists(embs_path):
            self.texts = np.load(texts_path, allow_pickle=True).tolist()
            self.embeddings = np.load(embs_path)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadatas = json.load(f)
            if os.path.exists(idx_path):
                with open(idx_path) as f:
                    self.text_to_idx = json.load(f)
            print(f"  [EmbeddingStore:{self.name}] Loaded {len(self.texts)} entries")

    def save(self):
        """Persist to disk."""
        np.save(os.path.join(self.cache_dir, "texts.npy"), np.array(self.texts, dtype=object))
        if self.embeddings is not None:
            np.save(os.path.join(self.cache_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.cache_dir, "metadata.json"), "w") as f:
            json.dump(self.metadatas, f, ensure_ascii=False)
        with open(os.path.join(self.cache_dir, "text_to_idx.json"), "w") as f:
            json.dump(self.text_to_idx, f, ensure_ascii=False)

    def insert(self, texts: List[str], embeddings: np.ndarray,
               metadatas: Optional[List[dict]] = None):
        """Insert new texts with their embeddings.

        Deduplicates by text hash: if a text already exists, skip it.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        new_texts, new_embs, new_metas = [], [], []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            h = self._text_hash(text)
            if h not in self.text_to_idx:
                self.text_to_idx[h] = len(self.texts) + len(new_texts)
                new_texts.append(text)
                new_embs.append(emb)
                new_metas.append(meta)

        if not new_texts:
            print(f"  [EmbeddingStore:{self.name}] All {len(texts)} texts already exist, skipped")
            return

        self.texts.extend(new_texts)
        self.metadatas.extend(new_metas)

        new_embs_arr = np.array(new_embs, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embs_arr
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs_arr])

        print(f"  [EmbeddingStore:{self.name}] Inserted {len(new_texts)} new (total: {len(self.texts)})")
        self.save()

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float, dict]]:
        """Search by cosine similarity. Returns [(text, score, metadata), ...]."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Normalize
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embs = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        scores = np.dot(embs, q)
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.texts[idx], float(scores[idx]), self.metadatas[idx]))
        return results

    def get_embeddings(self, indices: Optional[List[int]] = None) -> np.ndarray:
        if indices is None:
            return self.embeddings
        return self.embeddings[indices]

    def count(self) -> int:
        return len(self.texts) if self.embeddings is not None else 0

    def get_all_ids(self) -> List[str]:
        return list(range(len(self.texts)))

    def get_row(self, idx: int) -> dict:
        return {"text": self.texts[idx], "metadata": self.metadatas[idx]}
