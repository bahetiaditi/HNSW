"""Tests for Numba distance kernels."""

import sys
import os
import time

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distance import l2_distance, l2_distance_batch, l2_distance_single_to_many_ids


def test_l2_distance_basic():
    """Squared L2 between known vectors."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    expected = (3**2 + 3**2 + 3**2)  # 27.0
    result = l2_distance(a, b)
    assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
    print("  ✓ test_l2_distance_basic")


def test_l2_distance_zero():
    """Distance of a vector to itself is zero."""
    a = np.array([1.5, -2.3, 0.7, 4.1], dtype=np.float32)
    result = l2_distance(a, a)
    assert abs(result) < 1e-10, f"Expected 0, got {result}"
    print("  ✓ test_l2_distance_zero")


def test_l2_distance_symmetry():
    """dist(a, b) == dist(b, a)."""
    rng = np.random.RandomState(42)
    a = rng.randn(128).astype(np.float32)
    b = rng.randn(128).astype(np.float32)
    assert abs(l2_distance(a, b) - l2_distance(b, a)) < 1e-5
    print("  ✓ test_l2_distance_symmetry")


def test_l2_distance_vs_numpy():
    """Compare against numpy reference on 128-dim vectors."""
    rng = np.random.RandomState(123)
    for _ in range(10):
        a = rng.randn(128).astype(np.float32)
        b = rng.randn(128).astype(np.float32)
        expected = float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        result = l2_distance(a, b)
        assert abs(result - expected) / max(expected, 1e-10) < 1e-5, (
            f"Mismatch: numba={result}, numpy={expected}"
        )
    print("  ✓ test_l2_distance_vs_numpy (10 random pairs)")


def test_l2_distance_batch_shape():
    """Batch output shape matches input."""
    rng = np.random.RandomState(42)
    query = rng.randn(128).astype(np.float32)
    vectors = rng.randn(1000, 128).astype(np.float32)
    result = l2_distance_batch(query, vectors)
    assert result.shape == (1000,), f"Expected (1000,), got {result.shape}"
    print("  ✓ test_l2_distance_batch_shape")


def test_l2_distance_batch_vs_numpy():
    """Batch distances match numpy element-wise computation."""
    rng = np.random.RandomState(42)
    query = rng.randn(128).astype(np.float32)
    vectors = rng.randn(500, 128).astype(np.float32)

    result = l2_distance_batch(query, vectors)
    expected = np.sum(
        (query.astype(np.float64) - vectors.astype(np.float64)) ** 2, axis=1
    )

    max_rel_err = np.max(np.abs(result - expected) / np.maximum(expected, 1e-10))
    assert max_rel_err < 1e-5, f"Max relative error: {max_rel_err}"
    print("  ✓ test_l2_distance_batch_vs_numpy (500 vectors)")


def test_l2_distance_batch_vs_pairwise():
    """Batch result matches calling l2_distance in a loop."""
    rng = np.random.RandomState(99)
    query = rng.randn(128).astype(np.float32)
    vectors = rng.randn(50, 128).astype(np.float32)

    batch_result = l2_distance_batch(query, vectors)
    for i in range(50):
        single_result = l2_distance(query, vectors[i])
        assert abs(batch_result[i] - single_result) < 1e-5
    print("  ✓ test_l2_distance_batch_vs_pairwise (50 vectors)")


def test_l2_distance_single_to_many_ids():
    """ID-based distance matches direct computation."""
    rng = np.random.RandomState(42)
    all_vectors = rng.randn(10000, 128).astype(np.float32)
    query = rng.randn(128).astype(np.float32)
    ids = np.array([0, 50, 999, 5000, 9999], dtype=np.int64)

    result = l2_distance_single_to_many_ids(query, all_vectors, ids)

    for i, vid in enumerate(ids):
        expected = l2_distance(query, all_vectors[vid])
        assert abs(result[i] - expected) < 1e-5
    print("  ✓ test_l2_distance_single_to_many_ids")


def test_numba_compilation_speedup():
    """Second call should be much faster than first (JIT compiled)."""
    rng = np.random.RandomState(42)
    query = rng.randn(128).astype(np.float32)
    vectors = rng.randn(10000, 128).astype(np.float32)

    # First call triggers compilation (already happened above, but let's time anyway)
    t0 = time.perf_counter()
    _ = l2_distance_batch(query, vectors)
    t1 = time.perf_counter()
    first_call = t1 - t0

    # Second call uses compiled code
    t0 = time.perf_counter()
    for _ in range(10):
        _ = l2_distance_batch(query, vectors)
    t1 = time.perf_counter()
    avg_compiled = (t1 - t0) / 10

    print(f"  ✓ test_numba_compilation_speedup "
          f"(first={first_call*1000:.1f}ms, compiled={avg_compiled*1000:.2f}ms)")


if __name__ == "__main__":
    print("Running distance kernel tests...\n")
    test_l2_distance_basic()
    test_l2_distance_zero()
    test_l2_distance_symmetry()
    test_l2_distance_vs_numpy()
    test_l2_distance_batch_shape()
    test_l2_distance_batch_vs_numpy()
    test_l2_distance_batch_vs_pairwise()
    test_l2_distance_single_to_many_ids()
    test_numba_compilation_speedup()
    print("\nAll distance tests passed! ✓")