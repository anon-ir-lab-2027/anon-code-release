"""
Knowledge Graph for HippoRAG.

Manages an igraph instance with entity and passage nodes,
synonymy edges, fact edges, and passage edges.
Supports Personalized PageRank (PPR) for retrieval.
"""

import hashlib
import json
import os
import pickle
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import numpy as np


class KnowledgeGraph:
    def __init__(self, directed: bool = True):
        self.graph: ig.Graph = ig.Graph(directed=directed)
        self.node_name_to_idx: Dict[str, int] = {}
        self.entity_node_idxs: List[int] = []
        self.passage_node_idxs: List[int] = []

        # Entity metadata: node_name -> {"name": ..., "description": ...}
        self.entity_meta: Dict[str, dict] = {}

        # Passage metadata: node_name -> chunk_text
        self.passage_meta: Dict[str, str] = {}

        # Edge stats for incremental building
        self.node_to_node_stats: Dict[Tuple[str, str], float] = {}

        # Edge labels (predicates) for fact edges: (src, dst) -> list of predicate strings
        self.node_to_node_labels: Dict[Tuple[str, str], List[str]] = {}

        # Entity node -> set of chunk node names
        self.ent_node_to_chunk_ids: Dict[str, Set[str]] = {}

        # Entity name -> canonical hash (alias merging)
        self._name_map: Dict[str, str] = {}

    # ── Entity hash helpers ──
    @staticmethod
    def entity_hash(name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        return hashlib.md5(name.lower().encode('utf-8')).hexdigest()

    def get_entity_hash(self, name: str) -> str:
        return self._name_map.get(name, self.entity_hash(name))

    @staticmethod
    def passage_hash(passage_idx: int) -> str:
        return f"passage-{passage_idx}"

    # ── Graph construction ──
    def add_fact_edges(self, triples: List[List[str]]):
        """Add directed edges between entities based on triples.
        Multiple triples between same entity pair increase edge weight.
        Edge direction follows triple order: subj -> obj only.
        """
        for triple in triples:
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            subj, pred, obj = triple
            if not all(isinstance(x, str) for x in [subj, pred, obj]):
                continue
            subj_key = self.get_entity_hash(subj)
            obj_key = self.get_entity_hash(obj)

            key = (subj_key, obj_key)

            # Increment weight (subj -> obj only, no reverse)
            self.node_to_node_stats[key] = \
                self.node_to_node_stats.get(key, 0.0) + 1

            # Store predicate as edge label
            if key not in self.node_to_node_labels:
                self.node_to_node_labels[key] = []
            if pred not in self.node_to_node_labels[key]:
                self.node_to_node_labels[key].append(pred)

    def add_passage_edges(self, entities: List[Tuple[str, str]], passage_idx: int):
        """Add edges: entity <-> passage node.

        Args:
            entities: list of (entity_name, entity_description)
            passage_idx: the passage index (0-based)
        """
        passage_key = self.passage_hash(passage_idx)
        for ent_name, ent_desc in entities:
            ent_key = self.get_entity_hash(ent_name)
            # Store entity metadata
            if ent_key not in self.entity_meta:
                self.entity_meta[ent_key] = {"name": ent_name, "description": ent_desc}
            # Edge: passage <-> entity (bidirectional)
            # passage->entity (weight=1.0), entity->passage (weight=1.0)
            # Equal weight in both directions for directed PPR
            self.node_to_node_stats[(passage_key, ent_key)] = 1.0
            self.node_to_node_stats[(ent_key, passage_key)] = 1.0
            # Track entity -> chunk mapping
            if ent_key not in self.ent_node_to_chunk_ids:
                self.ent_node_to_chunk_ids[ent_key] = set()
            self.ent_node_to_chunk_ids[ent_key].add(passage_key)

    def add_synonymy_edges(self, entity_names: List[str], entity_embeddings: np.ndarray,
                           top_k: int = 10, threshold: float = 0.85):
        """
        Add synonymy edges between entities based on embedding similarity.

        For each entity, find top_k nearest neighbors by cosine similarity.
        If similarity > threshold, add an edge.
        """
        if len(entity_names) < 2:
            return

        # Normalize embeddings
        norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = entity_embeddings / norms

        # For each entity, find similar ones
        sim_matrix = np.dot(normalized, normalized.T)

        num_added = 0
        for i in range(len(entity_names)):
            # Get top_k (excluding self)
            indices = np.argsort(-sim_matrix[i])[1:top_k+1]
            for j in indices:
                if sim_matrix[i, j] >= threshold:
                    ent_i_key = self.entity_hash(entity_names[i])
                    ent_j_key = self.entity_hash(entity_names[j])
                    if (ent_i_key, ent_j_key) not in self.node_to_node_stats and \
                       (ent_j_key, ent_i_key) not in self.node_to_node_stats:
                        self.node_to_node_stats[(ent_i_key, ent_j_key)] = float(sim_matrix[i, j])
                        num_added += 1

        if num_added > 0:
            print(f"  [KG] Added {num_added} synonymy edges")

    def build(self):
        """Finalize the graph: add all accumulated nodes and edges to igraph.
        The graph's directed mode is inherited from __init__."""
        # Collect all node names
        all_node_names = set()
        for (src, dst) in self.node_to_node_stats:
            all_node_names.add(src)
            all_node_names.add(dst)

        # Add vertices
        node_names_list = sorted(all_node_names)
        self.graph.add_vertices(len(node_names_list))
        self.graph.vs["name"] = node_names_list
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names_list)}

        # Categorize nodes
        entity_keys = set(self.entity_meta.keys())
        for name in node_names_list:
            if name.startswith("passage-"):
                self.passage_node_idxs.append(self.node_name_to_idx[name])
            elif name in entity_keys:
                self.entity_node_idxs.append(self.node_name_to_idx[name])

        # Add edges with weights and labels
        edge_list = []
        weights = []
        edge_labels = []
        for (src, dst), w in self.node_to_node_stats.items():
            if src in self.node_name_to_idx and dst in self.node_name_to_idx:
                edge_list.append((self.node_name_to_idx[src], self.node_name_to_idx[dst]))
                weights.append(w)
                labels = self.node_to_node_labels.get((src, dst), [])
                edge_labels.append(labels[0] if labels else "")

        self.graph.add_edges(edge_list)
        self.graph.es["weight"] = weights
        if edge_labels:
            self.graph.es["label"] = edge_labels

        print(f"  [KG] Built graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        print(f"  [KG]   Entity nodes: {len(self.entity_node_idxs)}")
        print(f"  [KG]   Passage nodes: {len(self.passage_node_idxs)}")

    # ── PPR ──
    def run_ppr(self, seed_weights: np.ndarray, damping: float = 0.5) -> np.ndarray:
        """Run personalized PageRank.

        Args:
            seed_weights: 1D array of length vcount(). Non-zero values mark seed nodes.
            damping: restart probability (teleportation factor)

        Returns:
            PPR scores for each node (same order as seed_weights)
        """
        seed_weights = np.where(np.isnan(seed_weights) | (seed_weights < 0), 0, seed_weights)
        scores = self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()),
            damping=damping,
            directed=self.graph.is_directed(),
            weights="weight",
            reset=seed_weights,
            implementation="prpack"
        )
        return np.array(scores)

    def graph_search(self, seed_entity_names: List[str],
                     seed_entity_scores: List[float],
                     passage_dpr_scores: np.ndarray,
                     passage_node_weight: float = 0.05,
                     damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full HippoRAG graph search pipeline.

        Args:
            seed_entity_names: entity names from reranked facts
            seed_entity_scores: scores for each seed entity
            passage_dpr_scores: DPR scores for all passage nodes (from dense retrieval)
            passage_node_weight: weight to blend DPR scores

        Returns:
            (sorted_passage_indices, sorted_passage_scores)
        """
        n = self.graph.vcount()
        node_weights = np.zeros(n)

        # Assign entity seed weights
        for ent_name, score in zip(seed_entity_names, seed_entity_scores):
            ent_key = self.get_entity_hash(ent_name)
            if ent_key in self.node_name_to_idx:
                idx = self.node_name_to_idx[ent_key]
                node_weights[idx] += score

        # Blend DPR passage scores
        for i, passage_idx in enumerate(self.passage_node_idxs):
            node_weights[passage_idx] += passage_dpr_scores[i] * passage_node_weight

        if np.sum(node_weights) == 0:
            # Fallback: use DPR scores directly
            return np.argsort(-passage_dpr_scores), passage_dpr_scores[np.argsort(-passage_dpr_scores)]

        # Run PPR
        ppr_scores = self.run_ppr(node_weights, damping=damping)

        # Extract passage scores
        passage_scores = ppr_scores[self.passage_node_idxs]

        # Sort
        sorted_indices = np.argsort(-passage_scores)
        sorted_scores = passage_scores[sorted_indices]

        return sorted_indices, sorted_scores

    # ── Persistence ──
    def save(self, path: str, format: str = "graphml"):
        """Save graph and metadata.

        Args:
            path: file path to save to
            format: "graphml" (recommended) or "pickle" (legacy, may have null-byte issues)
        """
        if format == "graphml" and self.graph.vcount() > 0:
            # Write graph to temporary GraphML file, then read bytes
            tmpf = tempfile.NamedTemporaryFile(suffix=".graphml", delete=False)
            tmp_path = tmpf.name
            tmpf.close()
            self.graph.write_graphml(tmp_path)
            with open(tmp_path, "rb") as f:
                graph_bytes = f.read()
            os.unlink(tmp_path)
            data = {
                "graph_format": "graphml",
                "graph_data": graph_bytes.decode("utf-8"),
                "node_name_to_idx": self.node_name_to_idx,
                "entity_node_idxs": self.entity_node_idxs,
                "passage_node_idxs": self.passage_node_idxs,
                "entity_meta": self.entity_meta,
                "passage_meta": self.passage_meta,
                "name_map": self._name_map,
                "node_to_node_stats": {f"{k[0]}||{k[1]}": v for k, v in self.node_to_node_stats.items()},
                "node_to_node_labels": {f"{k[0]}||{k[1]}": v for k, v in self.node_to_node_labels.items()},
                "ent_node_to_chunk_ids": {k: list(v) for k, v in self.ent_node_to_chunk_ids.items()},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            data = {
                "graph_format": "pickle",
                "graph_pickle": self.graph.write_pickle() if self.graph.vcount() > 0 else None,
                "node_name_to_idx": self.node_name_to_idx,
                "entity_node_idxs": self.entity_node_idxs,
                "passage_node_idxs": self.passage_node_idxs,
                "entity_meta": self.entity_meta,
                "passage_meta": self.passage_meta,
                "name_map": self._name_map,
                "node_to_node_stats": {f"{k[0]}||{k[1]}": v for k, v in self.node_to_node_stats.items()},
                "node_to_node_labels": {f"{k[0]}||{k[1]}": v for k, v in self.node_to_node_labels.items()},
                "ent_node_to_chunk_ids": {k: list(v) for k, v in self.ent_node_to_chunk_ids.items()},
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(data, f)
        print(f"  [KG] Saved to {path} (format={format})")

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load graph and metadata. Supports both JSON/GraphML and legacy pickle formats."""
        # Detect format
        with open(path, "rb") as f:
            header = f.read(4)
            f.seek(0)
            if header.startswith(b"{") or header.startswith(b"\xef\xbb\xbf{"):
                # JSON format
                data = json.loads(f.read().decode("utf-8"))
            else:
                # Legacy pickle format
                data = pickle.load(f)

        kg = cls()
        graph_format = data.get("graph_format", "pickle")

        if graph_format == "graphml":
            # Read_GraphML expects a file path, not XML string
            tmpf = tempfile.NamedTemporaryFile(suffix=".graphml", delete=False, mode="w")
            tmpf.write(data["graph_data"]); tmp_path = tmpf.name; tmpf.close()
            kg.graph = ig.Graph.Read_GraphML(tmp_path)
            os.unlink(tmp_path)
        elif graph_format == "pickle" and data.get("graph_pickle") is not None:
            gp = data["graph_pickle"]
            if isinstance(gp, bytes) and b"\x00" in gp[:100]:
                # Null bytes present: write to temp file, then read
                tmpf = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                tmpf.write(gp); tmp_path = tmpf.name; tmpf.close()
                kg.graph = ig.Graph.Read_Pickle(tmp_path)
                os.unlink(tmp_path)
            else:
                kg.graph = ig.Graph.Read_Pickle(gp)

        kg.node_name_to_idx = data["node_name_to_idx"]
        kg.entity_node_idxs = data["entity_node_idxs"]
        kg.passage_node_idxs = data["passage_node_idxs"]
        kg.entity_meta = data.get("entity_meta", [])
        kg.passage_meta = data.get("passage_meta", [])
        kg._name_map = data.get("name_map", {})
        raw = data.get("node_to_node_stats", {})
        kg.node_to_node_stats = {
            (k.split("||")[0], k.split("||")[1]): v
            for k, v in raw.items()
        } if isinstance(raw, dict) else {}
        raw_labels = data.get("node_to_node_labels", {})
        kg.node_to_node_labels = {
            (k.split("||")[0], k.split("||")[1]): v
            for k, v in raw_labels.items()
        } if isinstance(raw_labels, dict) else {}
        raw_ent = data.get("ent_node_to_chunk_ids", {})
        kg.ent_node_to_chunk_ids = {
            k: set(v) for k, v in raw_ent.items()
        } if isinstance(raw_ent, dict) else {}
        print(f"  [KG] Loaded from {path}")
        print(f"  [KG]   {kg.graph.vcount()} nodes, {kg.graph.ecount()} edges (format={graph_format})")
        import sys; sys.stdout.flush()
        return kg
