"""
Metadata filtering strategies for HNSW.

Implements three approaches to filtered approximate nearest neighbor search:
  - Pre-filter: brute-force exact KNN over predicate-matching subset
  - Post-filter: standard HNSW search with oversampling, then discard
  - Predicate-aware: modified beam search (ACORN-inspired)

TODO: Implement in Commits 7-9.
"""