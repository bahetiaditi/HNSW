# Filter-Aware HNSW for Constrained Vector Search

A from-scratch HNSW (Hierarchical Navigable Small World) implementation in Python, with three metadata filtering strategies benchmarked against each other and FAISS on the SIFT1M dataset.

## Motivation

Vector databases like Qdrant, Pinecone, and Weaviate all support metadata filtering on top of approximate nearest neighbor search. Naive approaches — filtering before search (pre-filter) or after search (post-filter) — each fail in opposite selectivity regimes. This project implements the core insight from [ACORN (Patel et al., 2024)](https://arxiv.org/abs/2403.04871): **predicate-aware graph traversal**, where the search frontier ignores the filter predicate (preserving graph connectivity) but the result set enforces it.

## What's Implemented

- **HNSW from scratch**: multi-layer graph construction, greedy search, RNG-based neighbor selection heuristic (Malkov & Yashunin, 2018), with Numba-accelerated L2 distance kernel.
- **Three filtering strategies**:
  - **Pre-filter**: brute-force exact KNN over the predicate-matching subset.
  - **Post-filter**: standard HNSW search with oversampled ef, then discard non-matching results.
  - **Predicate-aware traversal**: modified beam search where the candidate frontier is unfiltered but the result set only accepts matching nodes.
- **Benchmark study** on SIFT1M (1M vectors, 128 dims) with synthetic categorical metadata, characterizing recall@10 vs. query latency across selectivities from 0.1% to 50%.

## Key Results

*Coming soon — benchmarks in progress.*

## How to Reproduce

```bash
# 1. Clone and install
git clone https://github.com/bahetiaditi/HNSW.git
cd HNSW
pip install -r requirements.txt

# 2. Download SIFT1M dataset (~570 MB)
python data/download_sift1m.py

# 3. Build HNSW index and run unfiltered sanity check
python benchmarks/bench_unfiltered.py

# 4. Compute filtered ground truth (one-time, ~10 min)
python benchmarks/compute_ground_truth.py

# 5. Run filtered benchmark sweep
python benchmarks/bench_filtered.py

# 6. Generate plots
python benchmarks/plot_results.py
```

## Project Structure

```
filter-aware-hnsw/
├── src/
│   ├── hnsw.py              # Core HNSW implementation
│   ├── distance.py           # Numba-accelerated L2 distance kernels
│   ├── filtering.py          # Pre-filter, post-filter, predicate-aware search
│   └── utils.py              # fvecs/ivecs readers, metadata generation
├── benchmarks/
│   ├── bench_unfiltered.py   # Sanity check: our HNSW vs FAISS HNSW
│   ├── bench_filtered.py     # Main benchmark: 3 strategies × 6 selectivities
│   ├── compute_ground_truth.py
│   └── plot_results.py
├── data/
│   └── download_sift1m.py    # Download + verify SIFT1M from HuggingFace
├── results/                  # Generated plots and CSV results
├── writeup/
│   └── writeup.md            # 4-6 page technical report
└── tests/
    ├── test_hnsw.py
    ├── test_filtering.py
    └── test_distance.py
```

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs.* IEEE TPAMI.
- Patel, L., et al. (2024). *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.* SIGMOD.
- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI. (SIFT1M dataset)