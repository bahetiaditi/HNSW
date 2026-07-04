"""
Sanity check: compare our HNSW implementation against FAISS IndexHNSWFlat
on SIFT1M with no filtering.

This is the critical validation step. If recall@10 < 0.95 on our implementation,
there's a bug in neighbor selection, layer assignment, or search. Do not proceed
to filtering strategies until this passes.

Usage:
    # Full SIFT1M (1M vectors, ~30-60 min build time)
    python benchmarks/bench_unfiltered.py

    # Quick smoke test on a subset (e.g. 10K vectors, ~30 sec)
    python benchmarks/bench_unfiltered.py --subset 10000

    # Custom parameters
    python benchmarks/bench_unfiltered.py --M 16 --ef-construction 200 --ef-search 64 128 256
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hnsw import HNSW
from src.utils import read_fvecs, read_ivecs, recall_at_k


def load_sift1m(data_dir: str, subset: int = 0):
    """Load SIFT1M dataset.

    Args:
        data_dir: Directory containing sift_base.fvecs, sift_query.fvecs,
            sift_groundtruth.ivecs.
        subset: If > 0, use only the first `subset` base vectors.
            Ground truth is recomputed via brute force in this case.

    Returns:
        (base_vectors, query_vectors, ground_truth)
    """
    print("Loading SIFT1M dataset...")
    base_path = os.path.join(data_dir, "sift_base.fvecs")
    query_path = os.path.join(data_dir, "sift_query.fvecs")
    gt_path = os.path.join(data_dir, "sift_groundtruth.ivecs")

    base = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    gt = read_ivecs(gt_path)

    print(f"  Base vectors: {base.shape}")
    print(f"  Query vectors: {queries.shape}")
    print(f"  Ground truth: {gt.shape}")

    if subset > 0 and subset < base.shape[0]:
        print(f"\n  Using subset of {subset} base vectors.")
        base = base[:subset]

        # Recompute ground truth for the subset via brute force
        print("  Recomputing ground truth for subset (brute force)...")
        k = gt.shape[1]  # typically 100
        gt = _brute_force_gt(base, queries, k)
        print(f"  Subset ground truth: {gt.shape}")

    return base, queries, gt


def _brute_force_gt(base, queries, k):
    """Brute-force exact KNN ground truth."""
    n_queries = queries.shape[0]
    gt = np.empty((n_queries, k), dtype=np.int32)

    for i in range(n_queries):
        dists = np.sum(
            (base.astype(np.float64) - queries[i].astype(np.float64)) ** 2,
            axis=1,
        )
        gt[i] = np.argpartition(dists, k)[:k]
        # Sort top-k by distance
        top_k_dists = dists[gt[i]]
        sorted_order = np.argsort(top_k_dists)
        gt[i] = gt[i][sorted_order]

    return gt


def benchmark_our_hnsw(base, queries, gt, M, ef_construction, ef_search_values, k=10):
    """Build and benchmark our HNSW implementation.

    Args:
        base: Base vectors, shape (n, dim).
        queries: Query vectors, shape (n_queries, dim).
        gt: Ground truth, shape (n_queries, >=k).
        M: HNSW M parameter.
        ef_construction: HNSW efConstruction parameter.
        ef_search_values: List of ef values to sweep.
        k: Number of neighbors for recall computation.

    Returns:
        Dict mapping ef -> {recall, latency_p50_ms, latency_p95_ms, qps}
    """
    n = base.shape[0]

    # Build index
    print(f"\nBuilding our HNSW index (n={n}, M={M}, efConstruction={ef_construction})...")
    index = HNSW(M=M, ef_construction=ef_construction, seed=42)

    t0 = time.time()
    index.build(base, show_progress=True)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Max level: {index.max_level}")
    print(f"  Entry point: {index.entry_point}")

    # Layer statistics
    layer_counts = {}
    for node_id, level in index._node_level.items():
        for lyr in range(level + 1):
            layer_counts[lyr] = layer_counts.get(lyr, 0) + 1
    print("  Nodes per layer:",
          {lyr: layer_counts[lyr] for lyr in sorted(layer_counts.keys())})

    # Benchmark queries
    n_queries = queries.shape[0]
    gt_k = gt[:, :k]
    results = {}

    for ef in ef_search_values:
        print(f"\n  Querying with ef={ef}...")

        # Warmup (100 queries or 10% of total, whichever is smaller)
        n_warmup = min(100, n_queries // 10)
        for i in range(n_warmup):
            index.search(queries[i], k=k, ef=ef)

        # Timed run
        all_latencies = []
        all_predictions = np.empty((n_queries, k), dtype=np.int32)

        for i in range(n_queries):
            t0 = time.perf_counter()
            res = index.search(queries[i], k=k, ef=ef)
            t1 = time.perf_counter()

            all_latencies.append((t1 - t0) * 1000)  # ms
            for j, (dist, node_id) in enumerate(res[:k]):
                all_predictions[i, j] = node_id

        latencies = np.array(all_latencies)
        recall = recall_at_k(all_predictions, gt_k, k=k)

        results[ef] = {
            "recall": recall,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "qps": 1000.0 / float(np.mean(latencies)),
        }

        print(f"    recall@{k} = {recall:.4f}")
        print(f"    p50 latency = {results[ef]['latency_p50_ms']:.2f} ms")
        print(f"    p95 latency = {results[ef]['latency_p95_ms']:.2f} ms")
        print(f"    QPS = {results[ef]['qps']:.1f}")

    return results, build_time


def benchmark_faiss_hnsw(base, queries, gt, M, ef_construction, ef_search_values, k=10):
    """Build and benchmark FAISS IndexHNSWFlat for comparison.

    Args: same as benchmark_our_hnsw.
    Returns: same format.
    """
    try:
        import faiss
    except ImportError:
        print("\n  FAISS not installed. Skipping FAISS baseline.")
        print("  Install with: pip install faiss-cpu")
        return None, None

    n, dim = base.shape

    print(f"\nBuilding FAISS IndexHNSWFlat (n={n}, M={M})...")

    t0 = time.time()
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.1f}s")

    n_queries = queries.shape[0]
    gt_k = gt[:, :k]
    results = {}

    for ef in ef_search_values:
        print(f"\n  Querying with ef={ef}...")
        index.hnsw.efSearch = ef

        # Warmup
        n_warmup = min(100, n_queries // 10)
        _ = index.search(queries[:n_warmup], k)

        # Timed run
        all_latencies = []
        all_predictions = np.empty((n_queries, k), dtype=np.int32)

        for i in range(n_queries):
            t0 = time.perf_counter()
            dists, ids = index.search(queries[i:i + 1], k)
            t1 = time.perf_counter()

            all_latencies.append((t1 - t0) * 1000)
            all_predictions[i] = ids[0]

        latencies = np.array(all_latencies)
        recall = recall_at_k(all_predictions, gt_k, k=k)

        results[ef] = {
            "recall": recall,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "qps": 1000.0 / float(np.mean(latencies)),
        }

        print(f"    recall@{k} = {recall:.4f}")
        print(f"    p50 latency = {results[ef]['latency_p50_ms']:.2f} ms")
        print(f"    p95 latency = {results[ef]['latency_p95_ms']:.2f} ms")
        print(f"    QPS = {results[ef]['qps']:.1f}")

    return results, build_time


def print_comparison(our_results, faiss_results, ef_search_values, k):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 75)
    print(f"{'COMPARISON: Our HNSW vs FAISS IndexHNSWFlat':^75}")
    print("=" * 75)

    header = f"{'ef':>6}  {'Ours recall':>12}  {'Ours p50':>10}"
    if faiss_results:
        header += f"  {'FAISS recall':>13}  {'FAISS p50':>10}  {'Recall Δ':>8}"
    print(header)
    print("-" * len(header))

    for ef in ef_search_values:
        our = our_results[ef]
        line = f"{ef:>6}  {our['recall']:>12.4f}  {our['latency_p50_ms']:>8.2f}ms"

        if faiss_results and ef in faiss_results:
            fa = faiss_results[ef]
            delta = our["recall"] - fa["recall"]
            line += (f"  {fa['recall']:>13.4f}"
                     f"  {fa['latency_p50_ms']:>8.2f}ms"
                     f"  {delta:>+8.4f}")

        print(line)

    # Verdict
    print("\n" + "-" * 75)
    best_ef = max(ef_search_values)
    our_best = our_results[best_ef]["recall"]

    if our_best >= 0.95:
        print(f"✓ PASS: Our HNSW achieves recall@{k} = {our_best:.4f} at ef={best_ef}")
        print("  Graph construction and search are working correctly.")
        print("  Safe to proceed to filtering strategies.")
    elif our_best >= 0.85:
        print(f"⚠ WARNING: recall@{k} = {our_best:.4f} at ef={best_ef}")
        print("  Below 0.95 target but may be acceptable. Inspect neighbor selection.")
    else:
        print(f"✗ FAIL: recall@{k} = {our_best:.4f} at ef={best_ef}")
        print("  Something is wrong with HNSW construction or search.")
        print("  Debug neighbor selection heuristic, layer assignment, or search_layer.")

    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark our HNSW vs FAISS on SIFT1M (unfiltered)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing SIFT1M files (default: data/)",
    )
    parser.add_argument(
        "--subset", type=int, default=0,
        help="Use first N base vectors (0 = full 1M). "
             "Use 10000-50000 for quick tests.",
    )
    parser.add_argument(
        "--M", type=int, default=16,
        help="HNSW M parameter (default: 16)",
    )
    parser.add_argument(
        "--ef-construction", type=int, default=200,
        help="HNSW efConstruction (default: 200)",
    )
    parser.add_argument(
        "--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256],
        help="ef values to sweep (default: 16 32 64 128 256)",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="k for recall@k (default: 10)",
    )
    parser.add_argument(
        "--skip-faiss", action="store_true",
        help="Skip FAISS baseline comparison",
    )
    args = parser.parse_args()

    # Load data
    base, queries, gt = load_sift1m(args.data_dir, subset=args.subset)

    # Benchmark our HNSW
    our_results, our_build = benchmark_our_hnsw(
        base, queries, gt,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search_values=args.ef_search,
        k=args.k,
    )

    # Benchmark FAISS
    faiss_results = None
    if not args.skip_faiss:
        faiss_results, faiss_build = benchmark_faiss_hnsw(
            base, queries, gt,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search_values=args.ef_search,
            k=args.k,
        )

    # Print comparison
    print_comparison(our_results, faiss_results, args.ef_search, args.k)

    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    n_base = base.shape[0]
    csv_path = os.path.join(results_dir, f"unfiltered_n{n_base}.csv")
    with open(csv_path, "w") as f:
        f.write("implementation,ef,recall,latency_p50_ms,latency_p95_ms,qps\n")
        for ef in args.ef_search:
            r = our_results[ef]
            f.write(f"ours,{ef},{r['recall']:.6f},{r['latency_p50_ms']:.3f},"
                    f"{r['latency_p95_ms']:.3f},{r['qps']:.1f}\n")
            if faiss_results and ef in faiss_results:
                r = faiss_results[ef]
                f.write(f"faiss,{ef},{r['recall']:.6f},"
                        f"{r['latency_p50_ms']:.3f},"
                        f"{r['latency_p95_ms']:.3f},{r['qps']:.1f}\n")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()