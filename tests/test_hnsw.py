"""Tests for core HNSW implementation."""

import sys
import os
import math
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hnsw import HNSW
from src.distance import l2_distance


def brute_force_knn(query, vectors, k):
    """Exact KNN by brute force, returns list of (dist, id)."""
    dists = [
        (float(np.sum((query.astype(np.float64) - vectors[i].astype(np.float64)) ** 2)), i)
        for i in range(len(vectors))
    ]
    dists.sort()
    return dists[:k]


def test_layer_assignment_distribution():
    """Layer distribution should be approximately geometric."""
    index = HNSW(M=16, ef_construction=50, seed=42)
    n_samples = 50000
    levels = [index._get_random_level() for _ in range(n_samples)]

    counts = Counter(levels)
    total = sum(counts.values())

    print("  Layer distribution:")
    for lyr in sorted(counts.keys()):
        frac = counts[lyr] / total
        # Expected: P(level=l) ≈ (1 - 1/M) * (1/M)^l for geometric
        # But our formula is floor(-ln(U) * mL), so the distribution
        # is roughly: P(level >= l) = exp(-l / mL) = (1/M)^l
        print(f"    Layer {lyr}: {counts[lyr]:>6d} ({frac:.3f})")

    # Most nodes should be at layer 0
    assert counts[0] / total > 0.5, "Layer 0 should have majority of nodes"
    # Layer 0+1 should cover almost all
    assert (counts[0] + counts.get(1, 0)) / total > 0.85
    # Should have some higher layers
    assert max(counts.keys()) >= 2, "Should have at least a few nodes at layer 2+"
    print("  ✓ test_layer_assignment_distribution")


def test_insert_first_node():
    """First inserted node becomes entry point."""
    index = HNSW(M=4, ef_construction=10, seed=42)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    index.insert(vec, 0)

    assert index.entry_point == 0
    assert index.n_nodes == 1
    assert index.max_level >= 0
    print("  ✓ test_insert_first_node")


def test_insert_multiple_nodes():
    """Build graph with 100 nodes, check structural invariants."""
    rng = np.random.RandomState(42)
    n, dim = 100, 16
    vectors = rng.randn(n, dim).astype(np.float32)

    index = HNSW(M=8, ef_construction=50, seed=42)
    for i in range(n):
        index.insert(vectors[i], i)

    assert index.n_nodes == n
    assert index.entry_point is not None

    # Check all nodes exist at layer 0
    assert len(index.graphs[0]) == n, (
        f"Layer 0 should have all {n} nodes, has {len(index.graphs[0])}"
    )

    # Check degree bounds
    M0 = index.M0  # 2*M = 16
    M = index.M    # 8
    for lyr, adj in index.graphs.items():
        max_conn = M0 if lyr == 0 else M
        for node_id, neighbors in adj.items():
            assert len(neighbors) <= max_conn, (
                f"Node {node_id} at layer {lyr} has {len(neighbors)} "
                f"connections, max allowed is {max_conn}"
            )

    # Entry point should be at the highest layer
    ep_level = index._node_level[index.entry_point]
    assert ep_level == index.max_level
    print("  ✓ test_insert_multiple_nodes (degree bounds, layer structure)")


def test_search_exact_small_dataset():
    """On a tiny dataset, HNSW search should match brute force exactly."""
    rng = np.random.RandomState(42)
    n, dim = 50, 8
    vectors = rng.randn(n, dim).astype(np.float32)

    index = HNSW(M=8, ef_construction=100, seed=42)
    for i in range(n):
        index.insert(vectors[i], i)

    # With ef high enough on 50 vectors, should get perfect recall
    query = rng.randn(dim).astype(np.float32)
    k = 5

    hnsw_results = index.search(query, k=k, ef=50)
    bf_results = brute_force_knn(query, vectors, k)

    hnsw_ids = set(r[1] for r in hnsw_results)
    bf_ids = set(r[1] for r in bf_results)

    recall = len(hnsw_ids & bf_ids) / k
    assert recall == 1.0, (
        f"Expected perfect recall on 50 vectors with ef=50, got {recall}. "
        f"HNSW: {sorted(hnsw_ids)}, BF: {sorted(bf_ids)}"
    )
    print("  ✓ test_search_exact_small_dataset (recall=1.0 on 50 vectors)")


def test_search_recall_medium_dataset():
    """On 1000 vectors, recall@10 should be > 0.9 with reasonable ef."""
    rng = np.random.RandomState(42)
    n, dim = 1000, 32
    vectors = rng.randn(n, dim).astype(np.float32)

    index = HNSW(M=16, ef_construction=100, seed=42)
    for i in range(n):
        index.insert(vectors[i], i)

    n_queries = 50
    queries = rng.randn(n_queries, dim).astype(np.float32)
    k = 10

    total_recall = 0.0
    for q in queries:
        hnsw_results = index.search(q, k=k, ef=64)
        bf_results = brute_force_knn(q, vectors, k)

        hnsw_ids = set(r[1] for r in hnsw_results)
        bf_ids = set(r[1] for r in bf_results)
        total_recall += len(hnsw_ids & bf_ids) / k

    mean_recall = total_recall / n_queries
    print(f"  ✓ test_search_recall_medium_dataset "
          f"(recall@10 = {mean_recall:.3f} on 1000 vectors, 50 queries)")
    assert mean_recall > 0.9, f"Recall too low: {mean_recall:.3f}"


def test_build_vs_insert():
    """build() should produce same results as sequential insert()."""
    rng = np.random.RandomState(42)
    n, dim = 200, 16
    vectors = rng.randn(n, dim).astype(np.float32)

    # Sequential insert
    idx1 = HNSW(M=8, ef_construction=50, seed=42)
    for i in range(n):
        idx1.insert(vectors[i], i)

    # Bulk build
    idx2 = HNSW(M=8, ef_construction=50, seed=42)
    idx2.build(vectors, show_progress=False)

    # Both should have same structure
    assert idx1.n_nodes == idx2.n_nodes
    assert idx1.max_level == idx2.max_level
    assert idx1.entry_point == idx2.entry_point

    # Search results should be identical (same seed → same graph)
    query = rng.randn(dim).astype(np.float32)
    r1 = idx1.search(query, k=10, ef=50)
    r2 = idx2.search(query, k=10, ef=50)

    ids1 = [r[1] for r in r1]
    ids2 = [r[1] for r in r2]
    assert ids1 == ids2, f"build() and insert() produce different results"
    print("  ✓ test_build_vs_insert")


def test_search_returns_k_results():
    """Search should return exactly k results when enough nodes exist."""
    rng = np.random.RandomState(42)
    n, dim = 100, 8
    vectors = rng.randn(n, dim).astype(np.float32)

    index = HNSW(M=8, ef_construction=50, seed=42)
    for i in range(n):
        index.insert(vectors[i], i)

    query = rng.randn(dim).astype(np.float32)

    for k in [1, 5, 10, 20]:
        results = index.search(query, k=k, ef=max(k, 50))
        assert len(results) == k, (
            f"Expected {k} results, got {len(results)}"
        )
    print("  ✓ test_search_returns_k_results")


def test_search_distances_sorted():
    """Results should be sorted by distance ascending."""
    rng = np.random.RandomState(42)
    n, dim = 200, 16
    vectors = rng.randn(n, dim).astype(np.float32)

    index = HNSW(M=8, ef_construction=50, seed=42)
    for i in range(n):
        index.insert(vectors[i], i)

    query = rng.randn(dim).astype(np.float32)
    results = index.search(query, k=10, ef=50)

    for i in range(len(results) - 1):
        assert results[i][0] <= results[i + 1][0], (
            f"Results not sorted at position {i}: "
            f"{results[i][0]} > {results[i + 1][0]}"
        )
    print("  ✓ test_search_distances_sorted")


def test_neighbor_selection_heuristic():
    """The heuristic should prune redundant edges."""
    rng = np.random.RandomState(42)
    vectors = np.array([
        [0.0, 0.0],  # node 0 (insertion node)
        [1.0, 0.0],  # node 1 — close
        [1.1, 0.0],  # node 2 — close to node 1 (redundant)
        [0.0, 5.0],  # node 3 — far but in different direction (diverse)
    ], dtype=np.float32)

    index = HNSW(M=4, ef_construction=10, seed=42)
    index.vectors = vectors

    # Candidates for node 0, sorted by distance to node 0
    candidates = [
        (1.0, 1),    # dist(0, 1) = 1.0
        (1.21, 2),   # dist(0, 2) = 1.21
        (25.0, 3),   # dist(0, 3) = 25.0
    ]

    selected = index._select_neighbors_heuristic(0, candidates, M=2)
    selected_ids = [s[1] for s in selected]

    # Should keep node 1 (closest) and node 3 (diverse direction).
    # Node 2 should be pruned: dist(2, 1) = 0.01 < dist(2, 0) = 1.21
    assert 1 in selected_ids, "Node 1 should be selected (closest)"
    assert 2 not in selected_ids, "Node 2 should be pruned (redundant with 1)"
    assert 3 in selected_ids, "Node 3 should be selected (diverse direction)"
    print("  ✓ test_neighbor_selection_heuristic (prunes redundant edges)")


if __name__ == "__main__":
    print("Running HNSW tests...\n")
    test_layer_assignment_distribution()
    test_insert_first_node()
    test_insert_multiple_nodes()
    test_search_exact_small_dataset()
    test_search_recall_medium_dataset()
    test_build_vs_insert()
    test_search_returns_k_results()
    test_search_distances_sorted()
    test_neighbor_selection_heuristic()
    print("\nAll HNSW tests passed! ✓")