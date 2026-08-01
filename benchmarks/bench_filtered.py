"""
Main benchmark: three filtering strategies across six selectivity levels.

For each selectivity in {0.1%, 1%, 5%, 10%, 25%, 50%}:
  - Pre-filter: measure latency (recall is always 1.0)
  - Post-filter: sweep oversample_factor in {5, 10, 20, 50}
  - Predicate-aware: sweep ef in {16, 32, 64, 128, 256}
  - FAISS post-filter baseline (if faiss-cpu installed)

Requires:
  1. SIFT1M dataset downloaded (python data/download_sift1m.py)
  2. Ground truth computed (python benchmarks/compute_ground_truth.py)

Usage:
    # Full benchmark on SIFT1M
    python benchmarks/bench_filtered.py

    # Quick test on subset
    python benchmarks/bench_filtered.py --subset 10000

    # Specific selectivities only
    python benchmarks/bench_filtered.py --selectivities 1% 5% 10%
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hnsw import HNSW
from src.utils import (
    read_fvecs,
    read_ivecs,
    generate_metadata,
    get_selectivity_map,
    compute_ground_truth,
    recall_at_k,
)
from src.filtering import pre_filter_search, post_filter_search, predicate_aware_search


# --------------------------------------------------------------------------
# Benchmark helpers
# --------------------------------------------------------------------------

def benchmark_pre_filter(hnsw, queries, metadata, category_id, gt, k=10,
                         n_warmup=100):
    """Benchmark pre-filter search. Recall is always 1.0 by construction."""
    n_queries = queries.shape[0]

    # Warmup
    for i in range(min(n_warmup, n_queries)):
        pre_filter_search(hnsw, queries[i], k, metadata, category_id)

    # Timed run
    latencies = []
    predictions = []

    for i in range(n_queries):
        t0 = time.perf_counter()
        result = pre_filter_search(hnsw, queries[i], k, metadata, category_id)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        pred_ids = [r[1] for r in result[:k]]
        # Pad if fewer than k results
        while len(pred_ids) < k:
            pred_ids.append(-1)
        predictions.append(pred_ids)

    latencies = np.array(latencies)
    pred_array = np.array(predictions, dtype=np.int32)
    recall = recall_at_k(pred_array, gt, k=k)

    return {
        "recall": recall,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def benchmark_post_filter(hnsw, queries, metadata, category_id, gt, k=10,
                           oversample_factor=10, n_warmup=100):
    """Benchmark post-filter search at a given oversample factor."""
    n_queries = queries.shape[0]

    # Warmup
    for i in range(min(n_warmup, n_queries)):
        post_filter_search(hnsw, queries[i], k, metadata, category_id,
                           oversample_factor)

    # Timed run
    latencies = []
    predictions = []

    for i in range(n_queries):
        t0 = time.perf_counter()
        result = post_filter_search(hnsw, queries[i], k, metadata,
                                    category_id, oversample_factor)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        pred_ids = [r[1] for r in result[:k]]
        while len(pred_ids) < k:
            pred_ids.append(-1)
        predictions.append(pred_ids)

    latencies = np.array(latencies)
    pred_array = np.array(predictions, dtype=np.int32)
    recall = recall_at_k(pred_array, gt, k=k)

    return {
        "recall": recall,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def benchmark_predicate_aware(hnsw, queries, metadata, category_id, gt, k=10,
                               ef=64, n_warmup=100):
    """Benchmark predicate-aware search at a given ef."""
    n_queries = queries.shape[0]

    # Warmup
    for i in range(min(n_warmup, n_queries)):
        predicate_aware_search(hnsw, queries[i], k, ef, metadata, category_id)

    # Timed run
    latencies = []
    predictions = []

    for i in range(n_queries):
        t0 = time.perf_counter()
        result = predicate_aware_search(hnsw, queries[i], k, ef, metadata,
                                        category_id)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        pred_ids = [r[1] for r in result[:k]]
        while len(pred_ids) < k:
            pred_ids.append(-1)
        predictions.append(pred_ids)

    latencies = np.array(latencies)
    pred_array = np.array(predictions, dtype=np.int32)
    recall = recall_at_k(pred_array, gt, k=k)

    return {
        "recall": recall,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def benchmark_faiss_post_filter(base, queries, metadata, category_id, gt,
                                 M=16, ef_construction=200, k=10,
                                 oversample_factors=(10, 20, 50)):
    """FAISS IndexHNSWFlat with post-filtering baseline."""
    try:
        import faiss
    except ImportError:
        return None

    n, dim = base.shape
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(base)

    results = {}
    n_queries = queries.shape[0]

    for osf in oversample_factors:
        ef = k * osf
        index.hnsw.efSearch = ef

        # Warmup
        _ = index.search(queries[:100], ef)

        latencies = []
        predictions = []

        for i in range(n_queries):
            t0 = time.perf_counter()
            dists, ids = index.search(queries[i:i + 1], ef)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)

            # Post-filter
            matched = [int(idx) for idx in ids[0] if idx >= 0
                       and metadata[idx] == category_id]
            matched = matched[:k]
            while len(matched) < k:
                matched.append(-1)
            predictions.append(matched)

        latencies = np.array(latencies)
        pred_array = np.array(predictions, dtype=np.int32)
        recall = recall_at_k(pred_array, gt, k=k)

        results[osf] = {
            "recall": recall,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "mean_latency_ms": float(np.mean(latencies)),
        }

    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filtered benchmark: 3 strategies × 6 selectivities"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use first N base vectors (0 = full 1M)")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selectivities", nargs="+",
                        default=["0.1%", "1%", "5%", "10%", "25%", "50%"],
                        help="Selectivity levels to benchmark")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--n-queries", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load dataset
    print("Loading SIFT1M dataset...")
    base = read_fvecs(os.path.join(args.data_dir, "sift_base.fvecs"))
    queries = read_fvecs(os.path.join(args.data_dir, "sift_query.fvecs"))
    print(f"  Base: {base.shape}, Queries: {queries.shape}")

    if args.subset > 0 and args.subset < base.shape[0]:
        print(f"  Using subset of {args.subset} base vectors")
        base = base[:args.subset]

    if args.n_queries > 0:
        queries = queries[:args.n_queries]
        print(f"  Using {args.n_queries} queries")

    n = base.shape[0]

    # Generate metadata
    print("\nGenerating metadata...")
    metadata = generate_metadata(n, n_categories=20, seed=42)
    sel_map = get_selectivity_map(metadata, n)

    # Build HNSW index
    print(f"\nBuilding HNSW index (n={n}, M={args.M}, "
          f"efConstruction={args.ef_construction})...")
    hnsw = HNSW(M=args.M, ef_construction=args.ef_construction, seed=42)
    t0 = time.time()
    hnsw.build(base, show_progress=True)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.1f}s")

    # Sweep parameters
    oversample_factors = [5, 10, 20, 50]
    ef_values = [16, 32, 64, 128, 256]

    # CSV output
    csv_path = os.path.join(args.results_dir, f"filtered_n{n}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "selectivity", "strategy", "parameter", "param_value",
        "recall", "latency_p50_ms", "latency_p95_ms", "mean_latency_ms",
    ])

    # Run benchmarks
    print("\n" + "=" * 75)
    print(f"{'FILTERED BENCHMARK SWEEP':^75}")
    print("=" * 75)

    for sel_label in args.selectivities:
        if sel_label not in sel_map:
            print(f"\n  Skipping unknown selectivity: {sel_label}")
            continue

        info = sel_map[sel_label]
        cat_id = info["category_id"]
        count = info["count"]
        actual_sel = info["actual_selectivity"]

        print(f"\n{'─' * 75}")
        print(f"  Selectivity: {sel_label} (category={cat_id}, "
              f"count={count}, actual={actual_sel*100:.2f}%)")
        print(f"{'─' * 75}")

        # Compute or load ground truth
        gt_filename = f"gt_selectivity_{sel_label.replace('%', 'pct').replace('.', 'p')}.npy"
        gt_path = os.path.join(args.results_dir, gt_filename)

        if os.path.exists(gt_path) and args.subset == 0:
            gt = np.load(gt_path)
            if args.n_queries > 0:
                gt = gt[:args.n_queries]
            print(f"  Ground truth loaded from cache: {gt.shape}")
        else:
            print(f"  Computing ground truth (brute force)...", end="", flush=True)
            t0 = time.time()
            gt = compute_ground_truth(base, queries, metadata, cat_id,
                                      k=args.k, batch_size=100)
            print(f"  {time.time()-t0:.1f}s")
            if args.subset == 0:
                np.save(gt_path, gt)

        # --- Pre-filter ---
        print(f"\n  [Pre-filter]")
        r = benchmark_pre_filter(hnsw, queries, metadata, cat_id, gt, k=args.k)
        print(f"    recall@{args.k}={r['recall']:.4f}  "
              f"p50={r['latency_p50_ms']:.2f}ms  "
              f"p95={r['latency_p95_ms']:.2f}ms")
        writer.writerow([
            sel_label, "pre_filter", "none", "none",
            f"{r['recall']:.6f}", f"{r['latency_p50_ms']:.3f}",
            f"{r['latency_p95_ms']:.3f}", f"{r['mean_latency_ms']:.3f}",
        ])

        # --- Post-filter ---
        print(f"\n  [Post-filter]")
        for osf in oversample_factors:
            r = benchmark_post_filter(hnsw, queries, metadata, cat_id, gt,
                                       k=args.k, oversample_factor=osf)
            print(f"    osf={osf:>3d}  recall@{args.k}={r['recall']:.4f}  "
                  f"p50={r['latency_p50_ms']:.2f}ms  "
                  f"p95={r['latency_p95_ms']:.2f}ms")
            writer.writerow([
                sel_label, "post_filter", "oversample_factor", str(osf),
                f"{r['recall']:.6f}", f"{r['latency_p50_ms']:.3f}",
                f"{r['latency_p95_ms']:.3f}", f"{r['mean_latency_ms']:.3f}",
            ])

        # --- Predicate-aware ---
        print(f"\n  [Predicate-aware]")
        for ef in ef_values:
            r = benchmark_predicate_aware(hnsw, queries, metadata, cat_id, gt,
                                           k=args.k, ef=ef)
            print(f"    ef={ef:>4d}  recall@{args.k}={r['recall']:.4f}  "
                  f"p50={r['latency_p50_ms']:.2f}ms  "
                  f"p95={r['latency_p95_ms']:.2f}ms")
            writer.writerow([
                sel_label, "predicate_aware", "ef", str(ef),
                f"{r['recall']:.6f}", f"{r['latency_p50_ms']:.3f}",
                f"{r['latency_p95_ms']:.3f}", f"{r['mean_latency_ms']:.3f}",
            ])

        # --- FAISS baseline ---
        if not args.skip_faiss:
            print(f"\n  [FAISS post-filter]")
            faiss_results = benchmark_faiss_post_filter(
                base, queries, metadata, cat_id, gt,
                M=args.M, ef_construction=args.ef_construction, k=args.k,
                oversample_factors=oversample_factors,
            )
            if faiss_results:
                for osf, r in faiss_results.items():
                    print(f"    osf={osf:>3d}  recall@{args.k}={r['recall']:.4f}  "
                          f"p50={r['latency_p50_ms']:.2f}ms  "
                          f"p95={r['latency_p95_ms']:.2f}ms")
                    writer.writerow([
                        sel_label, "faiss_post_filter", "oversample_factor",
                        str(osf), f"{r['recall']:.6f}",
                        f"{r['latency_p50_ms']:.3f}",
                        f"{r['latency_p95_ms']:.3f}",
                        f"{r['mean_latency_ms']:.3f}",
                    ])
            else:
                print("    (FAISS not installed, skipping)")

    csv_file.close()
    print(f"\n{'=' * 75}")
    print(f"Results saved to {csv_path}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()