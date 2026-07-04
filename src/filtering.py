"""
Metadata filtering strategies for HNSW.

Three approaches to filtered approximate nearest neighbor search:

1. Pre-filter: Brute-force exact KNN over the predicate-matching subset.
   Perfect recall, but cost is O(selectivity * n * dim). Wins when
   selectivity is very low (tiny matching subset).

2. Post-filter: Standard HNSW search with oversampled ef, then discard
   non-matching results. Fast at high selectivity (most results match),
   but recall collapses at low selectivity because the graph navigates
   toward globally nearest vectors, not nearest matching vectors.

3. Predicate-aware traversal (ACORN-inspired): Modified beam search where
   the candidate frontier ignores the predicate (preserving graph
   connectivity through non-matching "stepping stone" nodes) but the
   result set only accepts matching nodes. This is the core insight
   from ACORN (Patel et al., 2024).
"""

import heapq
from typing import Optional

import numpy as np

from src.distance import l2_distance, l2_distance_batch
from src.hnsw import HNSW


# --------------------------------------------------------------------------
# Strategy 1: Pre-filter (brute-force exact KNN on matching subset)
# --------------------------------------------------------------------------

def pre_filter_search(
    hnsw: HNSW,
    query: np.ndarray,
    k: int,
    metadata: np.ndarray,
    target_category: int,
) -> list[tuple[float, int]]:
    """Brute-force exact KNN over vectors matching the predicate.

    1. Find all vector IDs where metadata == target_category
    2. Compute L2 distance from query to each matching vector
    3. Return top k by distance

    This gives perfect recall but is O(n_match * dim) per query.

    Args:
        hnsw: The HNSW index (used only to access stored vectors).
        query: Query vector, shape (dim,).
        k: Number of nearest neighbors to return.
        metadata: Category labels for all vectors, shape (n,).
        target_category: The category to filter on.

    Returns:
        List of (distance, node_id) sorted by distance ascending,
        at most k elements.
    """
    # Find matching IDs
    matching_ids = np.where(metadata == target_category)[0]

    if len(matching_ids) == 0:
        return []

    if len(matching_ids) <= k:
        # Fewer matches than k — return all, sorted by distance
        results = []
        for mid in matching_ids:
            dist = l2_distance(query, hnsw.vectors[mid])
            results.append((dist, int(mid)))
        results.sort(key=lambda x: x[0])
        return results

    # Compute distances to all matching vectors using batch operation
    matching_vectors = hnsw.vectors[matching_ids]
    distances = l2_distance_batch(query, matching_vectors)

    # Find top-k via argpartition (O(n) instead of O(n log n))
    top_k_local = np.argpartition(distances, k)[:k]

    # Sort the top-k by distance
    sorted_order = np.argsort(distances[top_k_local])
    top_k_local = top_k_local[sorted_order]

    # Map back to global IDs
    results = [
        (float(distances[local_idx]), int(matching_ids[local_idx]))
        for local_idx in top_k_local
    ]

    return results


# --------------------------------------------------------------------------
# Strategy 2: Post-filter (HNSW search + discard non-matching)
# --------------------------------------------------------------------------

def post_filter_search(
    hnsw: HNSW,
    query: np.ndarray,
    k: int,
    metadata: np.ndarray,
    target_category: int,
    oversample_factor: int = 10,
) -> list[tuple[float, int]]:
    """Standard HNSW search with oversampling, then discard non-matching results.

    1. Run HNSW search with ef = k * oversample_factor
    2. From returned candidates, keep only those matching the predicate
    3. Return top k matching results

    Fast at high selectivity (most candidates match anyway), but recall
    collapses at low selectivity because the graph traverses toward
    globally nearest vectors that are mostly non-matching.

    Args:
        hnsw: The HNSW index.
        query: Query vector, shape (dim,).
        k: Number of nearest neighbors to return.
        metadata: Category labels for all vectors, shape (n,).
        target_category: The category to filter on.
        oversample_factor: Multiply k by this for the HNSW search ef.
            Higher = better recall but slower. Typical values: 5-50.

    Returns:
        List of (distance, node_id) sorted by distance ascending.
        May contain fewer than k results if not enough matches found.
    """
    ef = k * oversample_factor

    # Standard HNSW search (unfiltered)
    candidates = hnsw.search(query, k=ef, ef=ef)

    # Filter to matching candidates
    filtered = [
        (dist, node_id)
        for dist, node_id in candidates
        if metadata[node_id] == target_category
    ]

    # Return top k
    return filtered[:k]


# --------------------------------------------------------------------------
# Strategy 3: Predicate-aware traversal (ACORN-inspired)
# --------------------------------------------------------------------------

def predicate_aware_search(
    hnsw: HNSW,
    query: np.ndarray,
    k: int,
    ef: int,
    metadata: np.ndarray,
    target_category: int,
) -> list[tuple[float, int]]:
    """Modified beam search: unfiltered frontier, filtered result set.

    The key ACORN insight: during graph traversal, the candidate frontier
    expansion ignores the predicate (so the search can walk through
    non-matching nodes as "stepping stones"), but the result set only
    accepts nodes that satisfy the predicate.

    This preserves graph connectivity while ensuring results are valid.

    At upper layers (above layer 0), we use standard greedy descent (ef=1)
    with no filtering, since upper layers have few nodes and filtering
    there would cripple navigation. The filtered beam search runs only
    at layer 0 where the actual result set is built.

    Args:
        hnsw: The HNSW index.
        query: Query vector, shape (dim,).
        k: Number of nearest neighbors to return.
        ef: Beam width for the filtered search at layer 0.
            Higher = better recall, slower. Must be >= k.
        metadata: Category labels for all vectors, shape (n,).
        target_category: The category to filter on.

    Returns:
        List of (distance, node_id) sorted by distance ascending.
        All returned nodes satisfy the predicate.
        May contain fewer than k results at very low selectivity.
    """
    if hnsw.entry_point is None:
        return []

    ef = max(ef, k)

    # Phase 1: Standard greedy descent from top layer to layer 1 (no filtering)
    current_ep = hnsw.entry_point
    for lyr in range(hnsw.max_level, 0, -1):
        result = hnsw._search_layer(query, [current_ep], ef=1, layer=lyr)
        if result:
            current_ep = result[0][1]

    # Phase 2: Filtered beam search at layer 0
    results = _search_layer_filtered(
        hnsw, query, [current_ep], ef, layer=0,
        metadata=metadata, target_category=target_category,
    )

    return results[:k]


def _search_layer_filtered(
    hnsw: HNSW,
    query: np.ndarray,
    entry_points: list[int],
    ef: int,
    layer: int,
    metadata: np.ndarray,
    target_category: int,
) -> list[tuple[float, int]]:
    """Beam search with predicate-aware result set filtering.

    Differs from standard _search_layer in one critical way:
      - Candidate frontier (C): ALL neighbors are added, regardless of predicate
      - Result set (W): ONLY nodes matching the predicate are accepted

    Non-matching nodes serve as "stepping stones" — the search walks through
    them to maintain graph connectivity, but they never appear in results.

    The stopping condition is also adapted: we stop when the closest
    remaining candidate is farther than the farthest MATCHING node in
    the result set AND we have at least ef matching results. We also
    track total nodes visited and stop if we exhaust the reachable graph.

    Args:
        hnsw: The HNSW index.
        query: Query vector, shape (dim,).
        entry_points: Starting node IDs for the search.
        ef: Beam width (max matching results to collect).
        layer: Graph layer to search.
        metadata: Category labels array.
        target_category: The predicate value.

    Returns:
        List of (distance, node_id) for matching nodes only,
        sorted by distance ascending.
    """
    visited = set(entry_points)

    # candidates: min-heap of (distance, node_id) — frontier, ALL nodes
    candidates = []

    # results: max-heap of (-distance, node_id) — ONLY matching nodes
    results = []

    for ep in entry_points:
        dist = l2_distance(query, hnsw.vectors[ep])
        heapq.heappush(candidates, (dist, ep))

        # Only add to results if it matches the predicate
        if metadata[ep] == target_category:
            heapq.heappush(results, (-dist, ep))

    while candidates:
        # Pop closest candidate (matching or not — it's a stepping stone)
        c_dist, c_id = heapq.heappop(candidates)

        # Stopping condition: if we have enough matching results and the
        # closest candidate is farther than the farthest matching result,
        # no remaining candidate can improve the result set
        if results and len(results) >= ef:
            f_dist = -results[0][0]  # farthest matching result
            if c_dist > f_dist:
                break

        # Expand neighbors — ALL neighbors, regardless of predicate
        neighbors = hnsw.graphs.get(layer, {}).get(c_id, [])
        for n_id in neighbors:
            if n_id in visited:
                continue
            visited.add(n_id)

            n_dist = l2_distance(query, hnsw.vectors[n_id])

            # Determine if this neighbor should enter the candidate frontier.
            # It should if: (a) we don't have enough matching results yet, or
            # (b) it's closer than the farthest matching result
            should_add = len(results) < ef
            if not should_add and results:
                f_dist = -results[0][0]
                should_add = n_dist < f_dist

            if should_add:
                # Always add to candidates (stepping stone potential)
                heapq.heappush(candidates, (n_dist, n_id))

                # Only add to results if it matches the predicate
                if metadata[n_id] == target_category:
                    heapq.heappush(results, (-n_dist, n_id))
                    if len(results) > ef:
                        heapq.heappop(results)  # trim to ef

    # Convert results to sorted list (ascending distance)
    output = [(-neg_dist, node_id) for neg_dist, node_id in results]
    output.sort(key=lambda x: x[0])
    return output