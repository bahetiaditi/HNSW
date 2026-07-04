"""
Numba-accelerated L2 distance kernels for HNSW.

All functions compute SQUARED L2 distance (no square root).
This is monotonic with true L2, so ranking is preserved,
and we save one sqrt per distance computation.

Functions:
  - l2_distance: squared L2 between two vectors
  - l2_distance_batch: squared L2 from one query to an array of vectors
  - l2_distance_single_to_many_ids: squared L2 from query to specific
    vectors selected by ID from a flat array
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance between two float32 vectors.

    Args:
        a: First vector, shape (dim,).
        b: Second vector, shape (dim,).

    Returns:
        sum((a - b)^2), a float64 scalar.
    """
    d = 0.0
    for i in range(a.shape[0]):
        diff = float(a[i]) - float(b[i])
        d += diff * diff
    return d


@njit(fastmath=True, cache=True)
def l2_distance_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Squared L2 distance from one query to each row in a matrix.

    Args:
        query: Query vector, shape (dim,).
        vectors: Matrix of vectors, shape (n, dim).

    Returns:
        Array of squared L2 distances, shape (n,).
    """
    n = vectors.shape[0]
    dim = query.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = 0.0
        for j in range(dim):
            diff = float(query[j]) - float(vectors[i, j])
            d += diff * diff
        result[i] = d
    return result


@njit(fastmath=True, cache=True)
def l2_distance_single_to_many_ids(
    query: np.ndarray,
    all_vectors: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    """Squared L2 distance from query to vectors at specific IDs.

    Useful for computing distances to a node's neighbors without
    gathering their vectors into a contiguous array first.

    Args:
        query: Query vector, shape (dim,).
        all_vectors: Full vector store, shape (N, dim).
        ids: Array of integer IDs to compute distances for, shape (m,).

    Returns:
        Array of squared L2 distances, shape (m,).
    """
    m = ids.shape[0]
    dim = query.shape[0]
    result = np.empty(m, dtype=np.float64)
    for i in range(m):
        d = 0.0
        vid = ids[i]
        for j in range(dim):
            diff = float(query[j]) - float(all_vectors[vid, j])
            d += diff * diff
        result[i] = d
    return result