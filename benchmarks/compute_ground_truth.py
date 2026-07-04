"""
Compute brute-force exact filtered KNN ground truth for all selectivity levels.

This is a one-time expensive operation. Results are cached as .npy files in
results/ so benchmark scripts can load them instantly.

For each selectivity in {0.1%, 1%, 5%, 10%, 25%, 50%}, we:
  1. Find all vector IDs matching the target category
  2. For each of the 10,000 queries, compute exact KNN over the matching subset
  3. Save the ground truth array to results/gt_selectivity_{label}.npy

Usage:
    python benchmarks/compute_ground_truth.py
    python benchmarks/compute_ground_truth.py --data-dir data --k 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    read_fvecs,
    generate_metadata,
    get_selectivity_map,
    compute_ground_truth,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute filtered ground truth for all selectivity levels"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing SIFT1M files (default: data/)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save ground truth files (default: results/)",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbors (default: 10)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached files exist",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load dataset
    print("Loading SIFT1M dataset...")
    base = read_fvecs(os.path.join(args.data_dir, "sift_base.fvecs"))
    queries = read_fvecs(os.path.join(args.data_dir, "sift_query.fvecs"))
    print(f"  Base: {base.shape}, Queries: {queries.shape}")

    n = base.shape[0]

    # Generate metadata
    print("\nGenerating metadata labels...")
    metadata = generate_metadata(n, n_categories=20, seed=42)
    sel_map = get_selectivity_map(metadata, n)

    # Save metadata for use by benchmark scripts
    meta_path = os.path.join(args.results_dir, "metadata.npy")
    np.save(meta_path, metadata)
    print(f"  Metadata saved to {meta_path}")

    # Compute ground truth for each selectivity
    print(f"\nComputing filtered ground truth (k={args.k})...")
    print("=" * 65)

    total_time = 0.0

    for label, info in sel_map.items():
        cat_id = info["category_id"]
        count = info["count"]
        selectivity = info["actual_selectivity"]

        gt_filename = f"gt_selectivity_{label.replace('%', 'pct').replace('.', 'p')}.npy"
        gt_path = os.path.join(args.results_dir, gt_filename)

        if os.path.exists(gt_path) and not args.force:
            # Verify cached file
            cached = np.load(gt_path)
            if cached.shape == (queries.shape[0], args.k):
                print(f"  {label:>5s}  category={cat_id}  "
                      f"count={count:>7d}  [CACHED] {gt_filename}")
                continue
            else:
                print(f"  {label:>5s}  cached file has wrong shape "
                      f"{cached.shape}, recomputing...")

        print(f"  {label:>5s}  category={cat_id}  "
              f"count={count:>7d} ({selectivity*100:.1f}%)  computing...",
              end="", flush=True)

        t0 = time.time()
        gt = compute_ground_truth(
            base_vectors=base,
            query_vectors=queries,
            metadata=metadata,
            category_id=cat_id,
            k=args.k,
            batch_size=100,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        np.save(gt_path, gt)
        print(f"  {elapsed:.1f}s  → {gt_filename}")

    print("=" * 65)
    print(f"Total computation time: {total_time:.1f}s")
    print(f"All ground truth files saved in {os.path.abspath(args.results_dir)}/")

    # Print summary
    print("\nGround truth files:")
    for f in sorted(os.listdir(args.results_dir)):
        if f.startswith("gt_selectivity_"):
            fpath = os.path.join(args.results_dir, f)
            arr = np.load(fpath)
            print(f"  {f}: shape={arr.shape}")


if __name__ == "__main__":
    main()