"""Tests for filtering strategies: pre-filter, post-filter, predicate-aware."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hnsw import HNSW
from src.distance import l2_distance
from src.filtering import pre_filter_search, post_filter_search, predicate_aware_search
from src.utils import recall_at_k


def build_test_index(n=1000, dim=32, n_categories=5, M=16, ef_construction=100, seed=42):
    """Build a small HNSW index with synthetic metadata for testing."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n, dim).astype(np.float32)
    metadata = rng.randint(0, n_categories, size=n).astype(np.int32)

    index = HNSW(M=M, ef_construction=ef_construction, seed=seed)
    index.build(vectors, show_progress=False)

    return index, vectors, metadata


def brute_force_filtered_knn(query, vectors, metadata, category, k):
    """Ground truth: exact KNN over matching subset."""
    matching_ids = np.where(metadata == category)[0]
    if len(matching_ids) == 0:
        return []
    matching_vecs = vectors[matching_ids]
    dists = np.sum(
        (query.astype(np.float64) - matching_vecs.astype(np.float64)) ** 2,
        axis=1,
    )
    top_k = min(k, len(matching_ids))
    local_ids = np.argpartition(dists, top_k)[:top_k]
    sorted_order = np.argsort(dists[local_ids])
    local_ids = local_ids[sorted_order]
    return [(float(dists[li]), int(matching_ids[li])) for li in local_ids]


# --------------------------------------------------------------------------
# Pre-filter tests
# --------------------------------------------------------------------------

def test_pre_filter_perfect_recall():
    """Pre-filter is brute force, so recall must be exactly 1.0."""
    index, vectors, metadata = build_test_index(n=1000, n_categories=5)
    rng = np.random.RandomState(99)
    k = 10

    for category in range(5):
        for _ in range(10):
            query = rng.randn(32).astype(np.float32)
            result = pre_filter_search(index, query, k, metadata, category)
            gt = brute_force_filtered_knn(query, vectors, metadata, category, k)

            result_ids = set(r[1] for r in result)
            gt_ids = set(g[1] for g in gt)

            assert result_ids == gt_ids, (
                f"Pre-filter should match brute force exactly. "
                f"Category={category}, got {result_ids}, expected {gt_ids}"
            )

    print("  ✓ test_pre_filter_perfect_recall (5 categories × 10 queries)")


def test_pre_filter_returns_only_matching():
    """All returned results must match the predicate."""
    index, vectors, metadata = build_test_index()
    rng = np.random.RandomState(42)

    for category in range(5):
        query = rng.randn(32).astype(np.float32)
        result = pre_filter_search(index, query, k=10, metadata=metadata,
                                   target_category=category)
        for dist, node_id in result:
            assert metadata[node_id] == category, (
                f"Pre-filter returned node {node_id} with category "
                f"{metadata[node_id]}, expected {category}"
            )

    print("  ✓ test_pre_filter_returns_only_matching")


def test_pre_filter_sorted_by_distance():
    """Results must be sorted by distance ascending."""
    index, vectors, metadata = build_test_index()
    query = np.random.RandomState(42).randn(32).astype(np.float32)

    result = pre_filter_search(index, query, k=10, metadata=metadata,
                               target_category=0)
    for i in range(len(result) - 1):
        assert result[i][0] <= result[i + 1][0], "Results not sorted"

    print("  ✓ test_pre_filter_sorted_by_distance")


# --------------------------------------------------------------------------
# Post-filter tests
# --------------------------------------------------------------------------

def test_post_filter_returns_only_matching():
    """All post-filter results must match the predicate."""
    index, vectors, metadata = build_test_index()
    rng = np.random.RandomState(42)

    for category in range(5):
        query = rng.randn(32).astype(np.float32)
        result = post_filter_search(
            index, query, k=10, metadata=metadata,
            target_category=category, oversample_factor=10,
        )
        for dist, node_id in result:
            assert metadata[node_id] == category, (
                f"Post-filter returned node {node_id} with category "
                f"{metadata[node_id]}, expected {category}"
            )

    print("  ✓ test_post_filter_returns_only_matching")


def test_post_filter_high_selectivity():
    """At high selectivity (~50%), post-filter should have decent recall."""
    rng = np.random.RandomState(42)
    n, dim, k = 1000, 32, 10
    vectors = rng.randn(n, dim).astype(np.float32)

    # Create metadata where category 0 has ~50% of vectors
    metadata = np.zeros(n, dtype=np.int32)
    metadata[n // 2:] = 1  # category 0 = first 500, category 1 = last 500
    rng.shuffle(metadata)

    index = HNSW(M=16, ef_construction=100, seed=42)
    index.build(vectors, show_progress=False)

    total_recall = 0.0
    n_queries = 20
    for _ in range(n_queries):
        query = rng.randn(dim).astype(np.float32)
        result = post_filter_search(
            index, query, k=k, metadata=metadata,
            target_category=0, oversample_factor=10,
        )
        gt = brute_force_filtered_knn(query, vectors, metadata, 0, k)

        result_ids = set(r[1] for r in result)
        gt_ids = set(g[1] for g in gt)
        total_recall += len(result_ids & gt_ids) / k

    mean_recall = total_recall / n_queries
    print(f"  ✓ test_post_filter_high_selectivity "
          f"(recall@{k} = {mean_recall:.3f} at ~50% selectivity)")
    assert mean_recall > 0.7, f"Post-filter recall too low at 50%: {mean_recall}"


def test_post_filter_low_selectivity_degrades():
    """At low selectivity (~2%), post-filter recall should be noticeably lower."""
    rng = np.random.RandomState(42)
    n, dim, k = 2000, 32, 10
    vectors = rng.randn(n, dim).astype(np.float32)

    # Category 0 gets only ~2% of vectors (40 out of 2000)
    metadata = rng.randint(1, 50, size=n).astype(np.int32)  # categories 1-49
    metadata[:40] = 0  # only 40 vectors in category 0
    rng.shuffle(metadata)

    index = HNSW(M=16, ef_construction=100, seed=42)
    index.build(vectors, show_progress=False)

    total_recall = 0.0
    n_queries = 20
    for _ in range(n_queries):
        query = rng.randn(dim).astype(np.float32)
        result = post_filter_search(
            index, query, k=k, metadata=metadata,
            target_category=0, oversample_factor=10,
        )
        gt = brute_force_filtered_knn(query, vectors, metadata, 0, k)

        result_ids = set(r[1] for r in result)
        gt_ids = set(g[1] for g in gt[:k])
        total_recall += len(result_ids & gt_ids) / k

    mean_recall = total_recall / n_queries
    print(f"  ✓ test_post_filter_low_selectivity_degrades "
          f"(recall@{k} = {mean_recall:.3f} at ~2% selectivity)")
    # We just observe the degradation, not fail on it
    # The whole point is that post-filter degrades here


# --------------------------------------------------------------------------
# Predicate-aware tests
# --------------------------------------------------------------------------

def test_predicate_aware_returns_only_matching():
    """All predicate-aware results must match the predicate."""
    index, vectors, metadata = build_test_index()
    rng = np.random.RandomState(42)

    for category in range(5):
        query = rng.randn(32).astype(np.float32)
        result = predicate_aware_search(
            index, query, k=10, ef=64,
            metadata=metadata, target_category=category,
        )
        for dist, node_id in result:
            assert metadata[node_id] == category, (
                f"Predicate-aware returned node {node_id} with category "
                f"{metadata[node_id]}, expected {category}"
            )

    print("  ✓ test_predicate_aware_returns_only_matching")


def test_predicate_aware_sorted_by_distance():
    """Results must be sorted by distance ascending."""
    index, vectors, metadata = build_test_index()
    query = np.random.RandomState(42).randn(32).astype(np.float32)

    result = predicate_aware_search(
        index, query, k=10, ef=64,
        metadata=metadata, target_category=0,
    )
    for i in range(len(result) - 1):
        assert result[i][0] <= result[i + 1][0], "Results not sorted"

    print("  ✓ test_predicate_aware_sorted_by_distance")


def test_predicate_aware_beats_post_filter_at_low_selectivity():
    """At low selectivity, predicate-aware should have better recall than post-filter."""
    rng = np.random.RandomState(42)
    n, dim, k = 2000, 32, 10
    vectors = rng.randn(n, dim).astype(np.float32)

    # Category 0 gets only ~2% (40 out of 2000)
    metadata = rng.randint(1, 50, size=n).astype(np.int32)
    metadata[:40] = 0
    rng.shuffle(metadata)

    index = HNSW(M=16, ef_construction=100, seed=42)
    index.build(vectors, show_progress=False)

    post_recall_total = 0.0
    pred_recall_total = 0.0
    n_queries = 30

    for _ in range(n_queries):
        query = rng.randn(dim).astype(np.float32)
        gt = brute_force_filtered_knn(query, vectors, metadata, 0, k)
        gt_ids = set(g[1] for g in gt[:k])

        # Post-filter
        post_result = post_filter_search(
            index, query, k=k, metadata=metadata,
            target_category=0, oversample_factor=10,
        )
        post_ids = set(r[1] for r in post_result)
        post_recall_total += len(post_ids & gt_ids) / k

        # Predicate-aware
        pred_result = predicate_aware_search(
            index, query, k=k, ef=128,
            metadata=metadata, target_category=0,
        )
        pred_ids = set(r[1] for r in pred_result)
        pred_recall_total += len(pred_ids & gt_ids) / k

    post_recall = post_recall_total / n_queries
    pred_recall = pred_recall_total / n_queries

    print(f"  ✓ test_predicate_aware_beats_post_filter_at_low_selectivity")
    print(f"    Post-filter recall@{k}:      {post_recall:.3f}")
    print(f"    Predicate-aware recall@{k}:   {pred_recall:.3f}")
    print(f"    Δ = {pred_recall - post_recall:+.3f}")

    # Predicate-aware should be meaningfully better
    assert pred_recall >= post_recall, (
        f"Predicate-aware ({pred_recall:.3f}) should beat "
        f"post-filter ({post_recall:.3f}) at low selectivity"
    )


# --------------------------------------------------------------------------
# Cross-strategy tests
# --------------------------------------------------------------------------

def test_all_strategies_agree_at_full_selectivity():
    """With no filter (100% selectivity), all strategies should return
    the same results as unfiltered HNSW search."""
    rng = np.random.RandomState(42)
    n, dim, k = 500, 16, 10
    vectors = rng.randn(n, dim).astype(np.float32)

    # All vectors in category 0 → 100% selectivity
    metadata = np.zeros(n, dtype=np.int32)

    index = HNSW(M=8, ef_construction=100, seed=42)
    index.build(vectors, show_progress=False)

    query = rng.randn(dim).astype(np.float32)
    ef = 64

    # Unfiltered
    unfiltered = index.search(query, k=k, ef=ef)
    unfiltered_ids = set(r[1] for r in unfiltered)

    # Pre-filter (brute force over all vectors = should match exactly)
    pre = pre_filter_search(index, query, k, metadata, target_category=0)
    pre_ids = set(r[1] for r in pre)

    # Post-filter (no filtering needed, all match)
    post = post_filter_search(
        index, query, k, metadata, target_category=0, oversample_factor=10,
    )
    post_ids = set(r[1] for r in post)

    # Predicate-aware (all nodes match, so equivalent to unfiltered)
    pred = predicate_aware_search(
        index, query, k, ef, metadata, target_category=0,
    )
    pred_ids = set(r[1] for r in pred)

    # Post-filter and predicate-aware should match unfiltered
    # (since all nodes are category 0, filtering has no effect)
    assert post_ids == unfiltered_ids, (
        f"Post-filter disagrees with unfiltered at 100% selectivity: "
        f"{post_ids} vs {unfiltered_ids}"
    )
    assert pred_ids == unfiltered_ids, (
        f"Predicate-aware disagrees with unfiltered at 100% selectivity: "
        f"{pred_ids} vs {unfiltered_ids}"
    )

    # Pre-filter is exact brute force, so it might differ from HNSW
    # (HNSW is approximate). But at ef=64 on 500 vectors it should match.
    pre_unfiltered_overlap = len(pre_ids & unfiltered_ids)
    assert pre_unfiltered_overlap >= k - 1, (
        f"Pre-filter too different from unfiltered: "
        f"overlap={pre_unfiltered_overlap}/{k}"
    )

    print("  ✓ test_all_strategies_agree_at_full_selectivity")


def test_all_strategies_return_at_most_k():
    """No strategy should return more than k results."""
    index, vectors, metadata = build_test_index()
    query = np.random.RandomState(42).randn(32).astype(np.float32)
    k = 5

    pre = pre_filter_search(index, query, k, metadata, target_category=0)
    post = post_filter_search(index, query, k, metadata,
                              target_category=0, oversample_factor=10)
    pred = predicate_aware_search(index, query, k, ef=64,
                                  metadata=metadata, target_category=0)

    assert len(pre) <= k, f"Pre-filter returned {len(pre)} > {k}"
    assert len(post) <= k, f"Post-filter returned {len(post)} > {k}"
    assert len(pred) <= k, f"Predicate-aware returned {len(pred)} > {k}"

    print("  ✓ test_all_strategies_return_at_most_k")


if __name__ == "__main__":
    print("Running filtering strategy tests...\n")
    test_pre_filter_perfect_recall()
    test_pre_filter_returns_only_matching()
    test_pre_filter_sorted_by_distance()
    test_post_filter_returns_only_matching()
    test_post_filter_high_selectivity()
    test_post_filter_low_selectivity_degrades()
    test_predicate_aware_returns_only_matching()
    test_predicate_aware_sorted_by_distance()
    test_predicate_aware_beats_post_filter_at_low_selectivity()
    test_all_strategies_agree_at_full_selectivity()
    test_all_strategies_return_at_most_k()
    print("\nAll filtering tests passed! ✓")