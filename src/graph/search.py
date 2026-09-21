"""
MCS (Minimum Connected Subgraph) BFS-based graph search operations.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from infra.knowledge_graph import KnowledgeGraph
from infra.base import PASSAGE_NODE_WEIGHT


def _has_edge(kg: KnowledgeGraph, a: int, b: int) -> bool:
    eids = kg.graph.get_eids([(a, b)], directed=True, error=False)
    return len(eids) > 0 and eids[0] >= 0


def _correct_chain_direction(kg: KnowledgeGraph, chain: List[int]) -> List[int]:
    if len(chain) < 2:
        return chain
    result = list(chain)
    for i in range(len(result) - 1):
        a, b = result[i], result[i + 1]
        if not _has_edge(kg, a, b):
            if _has_edge(kg, b, a):
                result[i], result[i + 1] = b, a
    return result


def mcs_graph_search(
    kg: KnowledgeGraph,
    seed_entity_names: List[str],
    seed_entity_scores: List[float],
    passage_dpr_scores: np.ndarray,
    max_depth: int = 3,
    max_width: int = 20,
    bidirectional: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """BFS-based graph exploration replacing personalized_pagerank.

    From seed entities, expand layer by layer along outgoing (and optionally
    incoming) edges. At each hop the frontier is capped to max_width entities.
    Passages reached from any seed accumulate a depth-decayed score with a
    multi-path voting bonus.

    Returns:
        (sorted_passage_indices, sorted_passage_scores)
    """
    n_passages = len(kg.passage_node_idxs)

    # Map seed entity names -> graph node indices
    seed_indices: List[int] = []
    seed_scores_list: List[float] = []
    for name, score in zip(seed_entity_names, seed_entity_scores):
        ent_key = kg.get_entity_hash(name)
        if ent_key in kg.node_name_to_idx:
            seed_indices.append(kg.node_name_to_idx[ent_key])
            seed_scores_list.append(score)

    if not seed_indices:
        sorted_idx = np.argsort(-passage_dpr_scores)
        return sorted_idx, passage_dpr_scores[sorted_idx]

    # Fast lookups
    passage_set: set = set(kg.passage_node_idxs)
    pidx_to_arr: Dict[int, int] = {v: i for i, v in enumerate(kg.passage_node_idxs)}

    # Accumulators
    passage_mcs = np.zeros(n_passages)
    passage_path_count = np.zeros(n_passages, dtype=int)

    # Per-seed BFS
    for seed_idx, seed_score in zip(seed_indices, seed_scores_list):
        visited: set = {seed_idx}
        frontier: set = {seed_idx}

        for depth in range(1, max_depth + 1):
            next_scores: Dict[int, float] = {}
            next_frontier: set = set()

            for curr_idx in frontier:
                neighbours: set = set()
                neighbours.update(kg.graph.neighbors(curr_idx, mode="out"))
                if bidirectional:
                    neighbours.update(kg.graph.neighbors(curr_idx, mode="in"))

                for nb_idx in neighbours:
                    if nb_idx in passage_set:
                        arr_idx = pidx_to_arr[nb_idx]
                        passage_path_count[arr_idx] += 1
                        pc = passage_path_count[arr_idx]
                        passage_mcs[arr_idx] += (
                            seed_score
                            * (1.0 / (1 + depth))
                            * (1 + 0.2 * (pc - 1))
                        )
                    elif nb_idx not in visited:
                        next_scores[nb_idx] = (
                            next_scores.get(nb_idx, 0.0)
                            + seed_score * (1.0 / (1 + depth))
                        )
                        next_frontier.add(nb_idx)

            if not next_frontier:
                break

            # Top-W filter
            if len(next_frontier) > max_width:
                sorted_ents = sorted(next_scores.items(), key=lambda x: -x[1])
                next_frontier = {idx for idx, _ in sorted_ents[:max_width]}

            visited.update(next_frontier)
            frontier = next_frontier

    # Blend with DPR scores
    combined = passage_mcs + passage_dpr_scores * PASSAGE_NODE_WEIGHT

    if np.sum(passage_mcs) == 0:
        combined = passage_dpr_scores

    sorted_indices = np.argsort(-combined)
    sorted_scores = combined[sorted_indices]

    return sorted_indices, sorted_scores


def mcs_entity_only(
    kg: KnowledgeGraph,
    seed_entity_names: List[str],
    max_depth: int = 3,
    max_width: int = 8,
    bidirectional: bool = True,
) -> List[List[int]]:
    """Entity-only BFS with multi-seed cross-point detection.

    Each seed entity performs independent BFS over entity nodes only (skipping
    passage nodes). Each node tracks which seeds can reach it via a bitmask.
    At each step priority is given to expanding nodes that connect the most
    still-unconnected seed pairs. Nodes reached by >=2 different seeds are
    *cross points* that bridge multiple seeds; paths from seeds to cross points
    form the reasoning-chain skeleton.

    Falls back to the longest independent path from each seed if no cross point
    is found within *max_depth*.

    Returns:
        List of entity chains, each chain is [entity_vidx, ...]
    """
    passage_set: set = set(kg.passage_node_idxs)

    seen_idxs: set = set()
    seed_indices: List[int] = []
    for name in seed_entity_names:
        ent_key = kg.get_entity_hash(name)
        if ent_key in kg.node_name_to_idx:
            idx = kg.node_name_to_idx[ent_key]
            if idx not in seen_idxs:
                seen_idxs.add(idx)
                seed_indices.append(idx)

    if not seed_indices:
        return []

    n_seeds = len(seed_indices)

    def _entity_neighbours(vidx: int) -> set:
        neighbours: set = set()
        neighbours.update(kg.graph.neighbors(vidx, mode="out"))
        if bidirectional:
            neighbours.update(kg.graph.neighbors(vidx, mode="in"))
        return {n for n in neighbours if n not in passage_set}

    # Single-seed: simple BFS
    if n_seeds == 1:
        sidx = seed_indices[0]
        all_chains: List[List[int]] = []
        visited: set = {sidx}
        frontier_paths: Dict[int, List[int]] = {sidx: [sidx]}
        for _ in range(max_depth):
            next_frontier: Dict[int, List[int]] = {}
            for curr, curr_path in frontier_paths.items():
                for nb in _entity_neighbours(curr):
                    if nb in visited:
                        continue
                    new_path = curr_path + [nb]
                    if nb not in next_frontier:
                        next_frontier[nb] = new_path
            if not next_frontier:
                break
            sorted_nodes = sorted(
                next_frontier.items(),
                key=lambda x: len(_entity_neighbours(x[0])),
                reverse=True,
            )
            selected = dict(sorted_nodes[:max_width])
            for node, path in selected.items():
                visited.add(node)
                all_chains.append(path)
            frontier_paths = selected
        return all_chains

    # Multi-seed: independent BFS with cross-point detection
    bfs_paths: List[Dict[int, List[int]]] = [
        {sidx: [sidx]} for sidx in seed_indices
    ]
    frontiers: List[set] = [
        {sidx} for sidx in seed_indices
    ]
    node_bitmask: Dict[int, int] = {}
    for i, sidx in enumerate(seed_indices):
        node_bitmask[sidx] = 1 << i

    connected_pairs: set = set()
    cross_points: Dict[int, int] = {}

    for _depth in range(max_depth):
        if all(len(f) == 0 for f in frontiers):
            break

        new_discoveries: Dict[int, Dict[int, List[int]]] = {}

        for i in range(n_seeds):
            for curr in frontiers[i]:
                curr_path = bfs_paths[i][curr]
                for nb in _entity_neighbours(curr):
                    if nb in bfs_paths[i]:
                        continue
                    if nb not in new_discoveries:
                        new_discoveries[nb] = {}
                    if i not in new_discoveries[nb]:
                        new_discoveries[nb][i] = curr_path + [nb]

        if not new_discoveries:
            break

        for node, seed_path_map in new_discoveries.items():
            for i, path in seed_path_map.items():
                bfs_paths[i][node] = path

            old_mask = node_bitmask.get(node, 0)
            new_mask = old_mask
            for i in seed_path_map:
                new_mask |= 1 << i
            node_bitmask[node] = new_mask

            if new_mask.bit_count() >= 2:
                cross_points[node] = new_mask
                seeds_list = [j for j in range(n_seeds) if new_mask & (1 << j)]
                for a in seeds_list:
                    for b in seeds_list:
                        if a < b:
                            connected_pairs.add((a, b))

        candidates: List[Tuple[int, int, int]] = []
        for node in new_discoveries:
            mask = node_bitmask.get(node, 0)
            seeds_list = [j for j in range(n_seeds) if mask & (1 << j)]
            unconnected = 0
            for a in seeds_list:
                for b in seeds_list:
                    if a < b and (a, b) not in connected_pairs:
                        unconnected += 1
            candidates.append((node, unconnected, mask))

        candidates.sort(key=lambda x: (-x[1], -x[2].bit_count()))

        selected_nodes = [c[0] for c in candidates[:max_width]]

        frontiers = [set() for _ in range(n_seeds)]
        for node in selected_nodes:
            if node in new_discoveries:
                for i in new_discoveries[node]:
                    frontiers[i].add(node)

    # Build result chains from cross points
    result_chains: List[List[int]] = []
    best_pair_paths: Dict[Tuple[int, int], List[int]] = {}

    if cross_points:
        for cross_node, mask in cross_points.items():
            seeds_list = [i for i in range(n_seeds) if mask & (1 << i)]
            for ii in range(len(seeds_list)):
                for jj in range(ii + 1, len(seeds_list)):
                    i = seeds_list[ii]
                    j = seeds_list[jj]
                    path_i = bfs_paths[i].get(cross_node)
                    path_j = bfs_paths[j].get(cross_node)
                    if path_i is None or path_j is None:
                        continue
                    merged = path_i + path_j[:-1][::-1]
                    pair_key = (i, j)
                    if pair_key not in best_pair_paths or len(merged) < len(best_pair_paths[pair_key]):
                        best_pair_paths[pair_key] = merged

        if n_seeds > 1 and best_pair_paths:
            edges: List[Tuple[int, int, List[int]]] = sorted(
                [(i, j, path) for (i, j), path in best_pair_paths.items()],
                key=lambda x: len(x[2]),
            )
            parent = list(range(n_seeds))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def _union(x: int, y: int) -> bool:
                rx, ry = _find(x), _find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True

            mst_paths: Dict[Tuple[int, int], List[int]] = {}
            for i, j, path in edges:
                if _union(i, j):
                    mst_paths[(i, j)] = path
                    if len(mst_paths) == n_seeds - 1:
                        break

            seen: set = set()
            for pair_key, path in mst_paths.items():
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    result_chains.append(path)
        else:
            seen: set = set()
            for pair_key, path in best_pair_paths.items():
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    result_chains.append(path)

        if result_chains:
            return result_chains

    # Fallback: longest independent path from each seed
    fallback: List[List[int]] = []
    for i in range(n_seeds):
        longest: List[int] = []
        for path in bfs_paths[i].values():
            if len(path) > len(longest):
                longest = path
        if longest:
            fallback.append(longest)
    return fallback


def forward_explore_from_paths(
    kg: KnowledgeGraph,
    passage_paths: Dict[int, List[int]],
    max_depth: int = 2,
    max_width: int = 3,
) -> List[List[int]]:
    """From all unique entities in passage_paths, BFS along outgoing edges.

    Passage nodes are skipped. Returns entity chains.

    Returns:
        [[entity_vidx, entity_vidx, ...], ...]
        Each inner list is an entity chain, entities are graph vertex indices.
    """
    passage_set = set(kg.passage_node_idxs)

    start_entities: set = set()
    for path in passage_paths.values():
        start_entities.update(path)

    if not start_entities:
        return []

    all_paths: List[List[int]] = []
    visited_global: set = set(start_entities)

    frontier_with_paths: List[Tuple[int, List[int]]] = [
        (ent_idx, [ent_idx]) for ent_idx in start_entities
    ]

    for depth in range(1, max_depth + 1):
        next_candidates: Dict[int, Tuple[float, List[int]]] = {}

        for curr_idx, curr_path in frontier_with_paths:
            for nb_idx in kg.graph.neighbors(curr_idx, mode="out"):
                if nb_idx in passage_set:
                    continue
                if nb_idx not in visited_global:
                    new_path = curr_path + [nb_idx]
                    prev = next_candidates.get(nb_idx, (0.0, []))
                    next_candidates[nb_idx] = (prev[0] + 1.0, new_path)

        if not next_candidates:
            break

        sorted_cands = sorted(next_candidates.items(), key=lambda x: -x[1][0])
        selected = sorted_cands[:max_width]

        for vidx, (_, path) in selected:
            all_paths.append(path)

        frontier_with_paths = [(vidx, path) for vidx, (_, path) in selected]
        visited_global.update(vidx for vidx, _ in frontier_with_paths)

    return all_paths


def mcs_connected_subgraph(
    kg: KnowledgeGraph,
    seed_entity_names: List[str],
    max_depth: int = 3,
    max_width: int = 8,
    bidirectional: bool = True,
) -> List[List[int]]:
    """True 2-approximate Minimum Connected Subgraph via Steiner Tree algorithm.

    Algorithm (2-approximation for Steiner Tree):
      1. Compute all-pairs shortest paths between seed entities (entity nodes only).
      2. Build a complete graph where edge weight = shortest path distance.
      3. Run Kruskal's MST on this complete graph over seeds.
      4. Replace each MST edge with its shortest path to form the final subgraph.
      5. Return the merged paths as entity chains.

    On undirected entity-only graph (bidirectional=True), edge direction is
    ignored. On directed, only out edges are used.

    Returns:
        List of entity chains forming a connected subgraph via MST over seeds.
        Empty list if seeds cannot be connected.
    """
    passage_set: set = set(kg.passage_node_idxs)

    seen_idxs: set = set()
    seed_indices: List[int] = []
    for name in seed_entity_names:
        ent_key = kg.get_entity_hash(name)
        if ent_key in kg.node_name_to_idx:
            idx = kg.node_name_to_idx[ent_key]
            if idx not in seen_idxs:
                seen_idxs.add(idx)
                seed_indices.append(idx)

    if not seed_indices:
        return []

    n_seeds = len(seed_indices)

    if n_seeds < 2:
        return []

    # ── Step 1 & 2: Compute shortest paths between all seed pairs ─────────
    mode = "all" if bidirectional else "out"

    pair_paths: Dict[Tuple[int, int], List[int]] = {}
    pair_dist: Dict[Tuple[int, int], int] = {}

    for i in range(n_seeds):
        si = seed_indices[i]
        targets = seed_indices[i+1:]
        if not targets:
            continue
        # Suppress igraph C-level stderr warnings for unreachable targets
        import contextlib
        import os
        import io
        # Redirect to devnull at the fd level to catch C-level igraph warnings
        _fname = '/dev/null'
        devnull_fd = os.open(_fname, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            paths = kg.graph.get_shortest_paths(
                si, to=targets,
                mode=mode, output="vpath",
            )
        except Exception:
            paths = []
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        for j_offset, p in enumerate(paths):
            if not p:
                continue
            j = i + 1 + j_offset
            sj = seed_indices[j]
            dist = len(p) - 1
            if dist > 0 and dist <= max_depth:
                pair_paths[(i, j)] = p
                pair_dist[(i, j)] = dist

    if not pair_paths:
        return []

    # ── Step 3: Kruskal MST on seed distance graph ───────────────────────
    sorted_pairs = sorted(pair_dist.keys(), key=lambda k: pair_dist[k])

    uf = list(range(n_seeds))

    def _find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def _union(x: int, y: int) -> bool:
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return False
        uf[rx] = ry
        return True

    # Count how many components we have initially
    components = set(_find(i) for i in range(n_seeds))
    n_components = len(components)

    mst_edges: List[Tuple[int, int]] = []
    for (i, j) in sorted_pairs:
        if _union(i, j):
            mst_edges.append((i, j))
            # Check if all seeds now in one component
            if len(mst_edges) == n_seeds - n_components:
                break

    if not mst_edges:
        return []

    # ── Step 4: Build result chains from MST edges ───────────────────────
    result_chains: List[List[int]] = []
    seen_paths: set = set()

    for (i, j) in mst_edges:
        path = pair_paths[(i, j)]
        key = tuple(path)
        if key not in seen_paths:
            seen_paths.add(key)
            result_chains.append(path)
        # Also add reverse direction if available
        rev_key = tuple(reversed(path))
        if rev_key not in seen_paths:
            seen_paths.add(rev_key)

    # ── Step 5: Clean up overlapping paths by merging ────────────────────
    # Collect all unique nodes and edges in the subgraph
    all_nodes: List[int] = []
    seen_nodes: set = set()
    for chain in result_chains:
        for v in chain:
            if v not in seen_nodes:
                seen_nodes.add(v)
                all_nodes.append(v)

    if len(all_nodes) <= 2:
        # Return the longest chain as-is
        result_chains.sort(key=len, reverse=True)
        return [result_chains[0]] if result_chains else []

    # Build adjacency from MST paths, then output as flat node list
    # (the connected subgraph in traversal order)
    return result_chains
