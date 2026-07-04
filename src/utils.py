"""
Utility functions for the Filter-Aware HNSW project.

Provides:
  - read_fvecs / read_ivecs: read SIFT1M binary vector files
  - generate_metadata: assign synthetic categorical attributes to vectors
    with controllable selectivity
  - compute_ground_truth: brute-force exact filtered KNN for evaluation
"""

import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# File I/O for .fvecs / .ivecs format
# ---------------------------------------------------------------------------

def read_fvecs(filepath: str) -> np.ndarray:
    """Read a .fvecs file (float32 vectors).

    Binary format per vector: [dim: int32] [v1: float32] ... [v_dim: float32]
    Each 128-dim vector occupies 4 + 128*4 = 516 bytes.

    Args:
        filepath: Path to .fvecs file.

    Returns:
        Contiguous float32 array of shape (n_vectors, dim).
    """
    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        raw = np.fromfile(f, dtype=np.int32)

    vectors_per_row = dim + 1
    n_vectors = raw.shape[0] // vectors_per_row
    raw = raw.reshape(n_vectors, vectors_per_row)
    vectors = raw[:, 1:].copy().view(np.float32)
    return np.ascontiguousarray(vectors)


def read_ivecs(filepath: str) -> np.ndarray:
    """Read a .ivecs file (int32 vectors, used for ground truth indices).

    Same binary layout as .fvecs but all values are int32.

    Args:
        filepath: Path to .ivecs file.

    Returns:
        Contiguous int32 array of shape (n_vectors, dim).
    """
    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        raw = np.fromfile(f, dtype=np.int32)

    vectors_per_row = dim + 1
    n_vectors = raw.shape[0] // vectors_per_row
    raw = raw.reshape(n_vectors, vectors_per_row)
    vectors = raw[:, 1:].copy()
    return np.ascontiguousarray(vectors)


# ---------------------------------------------------------------------------
# Synthetic metadata generation
# ---------------------------------------------------------------------------

def generate_metadata(
    n_vectors: int,
    n_categories: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Assign each vector a categorical label from 0..n_categories-1.

    Uses a skewed distribution so that querying specific categories gives
    different selectivity levels. The category sizes are designed to hit
    these approximate selectivity targets:

        Category 0  →  0.1%  (1,000 vectors for n=1M)
        Category 1  →  1.0%  (10,000 vectors)
        Category 2  →  5.0%  (50,000 vectors)
        Category 3  →  10.0% (100,000 vectors)
        Category 4  →  25.0% (250,000 vectors)
        Category 5  →  50.0% (500,000 vectors)
        Categories 6-19 → remaining vectors split equally

    This design lets us sweep selectivity by simply changing which
    category we query, without regenerating metadata.

    Args:
        n_vectors: Total number of vectors (e.g. 1,000,000).
        n_categories: Total categories (default 20).
        seed: Random seed for reproducibility.

    Returns:
        int32 array of shape (n_vectors,) with category labels.
    """
    rng = np.random.RandomState(seed)

    # Target selectivities for the first 6 categories
    target_fractions = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    target_counts = [int(f * n_vectors) for f in target_fractions]

    # Remaining vectors split equally among categories 6..n_categories-1
    assigned = sum(target_counts)
    remaining = n_vectors - assigned
    n_remaining_cats = n_categories - len(target_fractions)

    if remaining < 0:
        raise ValueError(
            f"Target selectivity fractions sum to {sum(target_fractions):.2f}, "
            f"which exceeds 1.0 minus the overhead for remaining categories. "
            f"Reduce targets or increase n_vectors."
        )

    if n_remaining_cats > 0:
        per_remaining = remaining // n_remaining_cats
        leftover = remaining - per_remaining * n_remaining_cats
        remaining_counts = [per_remaining] * n_remaining_cats
        # Distribute leftover across first few remaining categories
        for i in range(leftover):
            remaining_counts[i] += 1
    else:
        remaining_counts = []

    all_counts = target_counts + remaining_counts

    # Verify total
    assert sum(all_counts) == n_vectors, (
        f"Count mismatch: {sum(all_counts)} != {n_vectors}"
    )

    # Build the label array: category i gets all_counts[i] entries
    labels = np.empty(n_vectors, dtype=np.int32)
    offset = 0
    for cat_id, count in enumerate(all_counts):
        labels[offset : offset + count] = cat_id
        offset += count

    # Shuffle so that categories are distributed randomly, not in contiguous blocks
    rng.shuffle(labels)

    return labels


def get_selectivity_map(metadata: np.ndarray, n_vectors: int) -> dict:
    """Return a mapping from approximate selectivity % to category ID.

    Useful for benchmark scripts to quickly look up which category
    to query for a given target selectivity.

    Args:
        metadata: Category labels array from generate_metadata().
        n_vectors: Total number of vectors.

    Returns:
        Dict mapping selectivity string (e.g. '0.1%') to
        (category_id, actual_count, actual_selectivity).
    """
    targets = {
        "0.1%": 0,
        "1%": 1,
        "5%": 2,
        "10%": 3,
        "25%": 4,
        "50%": 5,
    }
    result = {}
    for label, cat_id in targets.items():
        count = int(np.sum(metadata == cat_id))
        selectivity = count / n_vectors
        result[label] = {
            "category_id": cat_id,
            "count": count,
            "actual_selectivity": selectivity,
        }
    return result


# ---------------------------------------------------------------------------
# Filtered ground truth computation
# ---------------------------------------------------------------------------

def compute_ground_truth(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metadata: np.ndarray,
    category_id: int,
    k: int = 10,
    batch_size: int = 100,
) -> np.ndarray:
    """Compute brute-force exact KNN over the filtered subset.

    For each query, finds the k nearest neighbors among base vectors
    where metadata == category_id. This is the ground truth for
    evaluating filtered search recall.

    Args:
        base_vectors: All base vectors, shape (n, dim).
        query_vectors: Query vectors, shape (n_queries, dim).
        metadata: Category labels, shape (n,).
        category_id: The category to filter on.
        k: Number of nearest neighbors.
        batch_size: Process queries in batches to manage memory.

    Returns:
        int32 array of shape (n_queries, k) with ground truth neighbor IDs.
    """
    # Find indices of matching vectors
    matching_mask = metadata == category_id
    matching_ids = np.where(matching_mask)[0]
    matching_vectors = base_vectors[matching_ids]  # shape: (n_match, dim)

    n_queries = query_vectors.shape[0]
    n_match = matching_vectors.shape[0]

    if n_match < k:
        raise ValueError(
            f"Only {n_match} vectors match category {category_id}, "
            f"but k={k}. Need at least k matching vectors."
        )

    ground_truth = np.empty((n_queries, k), dtype=np.int32)

    # Process in batches to avoid OOM on large subsets
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_queries = query_vectors[start:end]  # (batch, dim)

        # Compute squared L2 distances: (batch, n_match)
        # ||q - v||^2 = ||q||^2 + ||v||^2 - 2*q·v
        q_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)  # (batch, 1)
        v_norms = np.sum(matching_vectors ** 2, axis=1, keepdims=True).T  # (1, n_match)
        dots = batch_queries @ matching_vectors.T  # (batch, n_match)
        dists = q_norms + v_norms - 2 * dots  # (batch, n_match)

        # For each query, find indices of k smallest distances
        # argpartition is O(n) vs O(n log n) for full sort
        top_k_local = np.argpartition(dists, k, axis=1)[:, :k]

        # Sort the top-k by actual distance for consistent ordering
        for i in range(end - start):
            local_ids = top_k_local[i]
            sorted_order = np.argsort(dists[i, local_ids])
            top_k_local[i] = local_ids[sorted_order]

        # Map local indices back to global vector IDs
        ground_truth[start:end] = matching_ids[top_k_local]

    return ground_truth


def recall_at_k(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> float:
    """Compute recall@k averaged over all queries.

    For each query, recall@k = |predicted ∩ ground_truth| / k.

    Args:
        predicted: Predicted neighbor IDs, shape (n_queries, k) or list of arrays.
        ground_truth: Ground truth neighbor IDs, shape (n_queries, k).
        k: Number of neighbors to evaluate.

    Returns:
        Mean recall@k across all queries (float in [0, 1]).
    """
    n_queries = ground_truth.shape[0]
    total_recall = 0.0

    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k])
        if isinstance(predicted, np.ndarray):
            pred_set = set(predicted[i, :k])
        else:
            # Handle list of variable-length arrays
            pred_set = set(predicted[i][:k])
        total_recall += len(pred_set & gt_set) / k

    return total_recall / n_queries