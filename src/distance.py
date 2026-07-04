"""
Numba-accelerated L2 distance kernels for HNSW.

Implements squared L2 distance (no square root — monotonic,
so ranking is preserved and we save computation).

TODO: Implement in Commit 2.
"""