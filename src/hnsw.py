"""
Core HNSW (Hierarchical Navigable Small World) implementation.

Implements the multi-layer proximity graph from:
  Malkov & Yashunin (2018), "Efficient and Robust Approximate Nearest
  Neighbor using Hierarchical Navigable Small World Graphs", IEEE TPAMI.

Key algorithms:
  - Multi-layer graph construction with geometric layer assignment
  - Greedy beam search (search_layer) at each layer
  - SELECT-NEIGHBORS-HEURISTIC (RNG-based pruning, Algorithm 4)
  - Insert with top-down greedy descent + bottom-up neighbor selection
  - Search with top-down greedy descent + layer-0 beam search

All distances are squared L2 (no sqrt — monotonic, so ranking is preserved).
"""

import heapq
import math
import random
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.distance import l2_distance


class HNSW:
    """Hierarchical Navigable Small World graph for approximate nearest neighbor search.

    Attributes:
        M: Max connections per node per layer (layer 0 uses 2*M).
        ef_construction: Beam width during insertion.
        mL: Layer assignment parameter = 1/ln(M).
        max_layers: Hard cap on number of layers.
        vectors: np.ndarray of shape (n_inserted, dim), stores all vectors.
        graphs: dict[int, dict[int, list[int]]] — adjacency lists per layer.
            graphs[layer][node_id] = [neighbor_id, ...]
        entry_point: Node ID of the global entry point (highest-layer node).
        max_level: Current highest layer in the graph.
        n_nodes: Number of inserted nodes.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        max_layers: int = 16,
        seed: int = 42,
    ):
        """Initialize an empty HNSW index.

        Args:
            M: Max connections per node per layer. Layer 0 allows 2*M.
                Higher M = better recall, more memory, slower search.
                Typical values: 12–48. Default 16.
            ef_construction: Beam width during insertion. Higher = better
                graph quality, slower build. Typical: 100–400. Default 200.
            max_layers: Hard cap on layer count (safety bound). Default 16.
            seed: Random seed for reproducible layer assignment.
        """
        self.M = M
        self.M0 = 2 * M  # max connections at layer 0
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.max_layers = max_layers

        # Vector storage — will be set as a numpy array on first insert
        # or via build(). For incremental insert, we use a list then convert.
        self._vectors_list: list[np.ndarray] = []
        self.vectors: Optional[np.ndarray] = None

        # Graph structure: layer -> {node_id: [neighbor_ids]}
        self.graphs: dict[int, dict[int, list[int]]] = {}

        # Node metadata
        self._node_level: dict[int, int] = {}  # node_id -> max layer

        # Entry point and current max level
        self.entry_point: Optional[int] = None
        self.max_level: int = -1

        # Count
        self.n_nodes: int = 0

        # RNG for layer assignment
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Layer assignment
    # ------------------------------------------------------------------

    def _get_random_level(self) -> int:
        """Sample a layer from the geometric distribution.

        Returns floor(-ln(uniform(0,1)) * mL), clamped to max_layers - 1.
        Most nodes get layer 0. Probability of layer l decreases exponentially.
        """
        r = self._rng.random()  # uniform in (0, 1)
        level = int(-math.log(r) * self.mL)
        return min(level, self.max_layers - 1)

    # ------------------------------------------------------------------
    # Core beam search at a single layer
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Beam search at one layer of the graph.

        Maintains two heaps:
          - candidates (min-heap): frontier of unexplored nodes, closest first
          - results (max-heap): best ef nodes found so far, farthest first

        Stopping condition: the closest remaining candidate is farther than
        the farthest node in results AND results has ef elements.

        Args:
            query: Query vector, shape (dim,).
            entry_points: List of node IDs to start the search from.
            ef: Beam width (max size of results set).
            layer: Which layer to search.

        Returns:
            List of (distance, node_id) sorted by distance ascending,
            containing at most ef elements.
        """
        visited = set(entry_points)

        # candidates: min-heap of (distance, node_id)
        # results: max-heap of (-distance, node_id) — negated for max behavior
        candidates = []
        results = []

        for ep in entry_points:
            dist = l2_distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            # Pop closest candidate
            c_dist, c_id = heapq.heappop(candidates)

            # Stopping condition: closest candidate is farther than farthest result
            # and we already have ef results
            f_dist = -results[0][0]  # farthest distance in results (negate back)
            if c_dist > f_dist and len(results) >= ef:
                break

            # Expand neighbors of c_id at this layer
            neighbors = self.graphs.get(layer, {}).get(c_id, [])
            for n_id in neighbors:
                if n_id in visited:
                    continue
                visited.add(n_id)

                n_dist = l2_distance(query, self.vectors[n_id])
                f_dist = -results[0][0]

                # Add to candidates and results if it could improve results
                if n_dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))

                    # Trim results to ef
                    if len(results) > ef:
                        heapq.heappop(results)  # remove farthest

        # Convert results from max-heap to sorted list (ascending distance)
        output = [(-neg_dist, node_id) for neg_dist, node_id in results]
        output.sort(key=lambda x: x[0])
        return output

    # ------------------------------------------------------------------
    # Neighbor selection heuristic (RNG-based pruning, Algorithm 4)
    # ------------------------------------------------------------------

    def _select_neighbors_heuristic(
        self,
        node_id: int,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[tuple[float, int]]:
        """SELECT-NEIGHBORS-HEURISTIC from Malkov & Yashunin (2018).

        RNG-based pruning: keeps a candidate c only if no already-selected
        neighbor n is closer to c than c is to the insertion node. This
        prunes "redundant" edges where a shorter path exists through an
        already-selected neighbor, approximating a Relative Neighborhood Graph.

        Args:
            node_id: The node we're selecting neighbors for (needed to
                understand the geometric relationship, but distances to
                node_id are already in the candidates list).
            candidates: List of (distance_to_node, candidate_id), sorted
                ascending by distance.
            M: Maximum number of neighbors to select.

        Returns:
            List of (distance, neighbor_id) for the selected neighbors,
            at most M elements.
        """
        # Sort candidates by distance to the node (ascending)
        working = sorted(candidates, key=lambda x: x[0])
        selected: list[tuple[float, int]] = []

        for c_dist, c_id in working:
            if len(selected) >= M:
                break

            # Check: is there any already-selected neighbor n such that
            # dist(c, n) < dist(c, node)?
            # If yes, c is "redundant" — there's a shorter path through n.
            is_good = True
            for _, s_id in selected:
                dist_c_s = l2_distance(self.vectors[c_id], self.vectors[s_id])
                if dist_c_s < c_dist:
                    is_good = False
                    break

            if is_good:
                selected.append((c_dist, c_id))

        return selected

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, vector: np.ndarray, node_id: int) -> None:
        """Insert a single vector into the HNSW index.

        Procedure:
        1. Assign a random max layer l for the new node.
        2. Phase 1 — greedy descent: from the entry point at the top layer,
           greedily walk down to layer l+1 (ef=1), finding the closest
           node at each layer. This locates the entry point for insertion.
        3. Phase 2 — neighbor selection: at each layer from l down to 0,
           beam search with ef=efConstruction, select neighbors via the
           heuristic, add bidirectional edges, prune over-degree nodes.
        4. If l > current max_level, update entry point and max_level.

        Args:
            vector: The vector to insert, shape (dim,).
            node_id: Integer ID for this vector (must equal self.n_nodes).
        """
        if node_id != self.n_nodes:
            raise ValueError(
                f"Expected node_id={self.n_nodes}, got {node_id}. "
                "Insert must be called with sequential IDs."
            )

        # Store vector
        self._vectors_list.append(vector)
        self.vectors = np.array(self._vectors_list, dtype=np.float32)

        # Assign random layer
        node_level = self._get_random_level()
        self._node_level[node_id] = node_level

        # Ensure graph layers exist
        for lyr in range(node_level + 1):
            if lyr not in self.graphs:
                self.graphs[lyr] = {}
            self.graphs[lyr][node_id] = []

        # Handle first node
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = node_level
            self.n_nodes += 1
            return

        # Phase 1: Greedy descent from top to node_level + 1
        # At each layer, do ef=1 search to find the single closest node
        current_ep = self.entry_point
        current_dist = l2_distance(vector, self.vectors[current_ep])

        for lyr in range(self.max_level, node_level, -1):
            # Search this layer with ef=1
            result = self._search_layer(vector, [current_ep], ef=1, layer=lyr)
            if result:
                current_ep = result[0][1]  # closest node
                current_dist = result[0][0]

        # Phase 2: Insert at each layer from min(node_level, max_level) down to 0
        ep_list = [current_ep]

        for lyr in range(min(node_level, self.max_level), -1, -1):
            # Beam search with ef=efConstruction to find candidates
            candidates = self._search_layer(
                vector, ep_list, ef=self.ef_construction, layer=lyr
            )

            # Select neighbors using the heuristic
            M_lyr = self.M0 if lyr == 0 else self.M
            neighbors = self._select_neighbors_heuristic(
                node_id, candidates, M=M_lyr
            )

            # Add bidirectional edges
            for n_dist, n_id in neighbors:
                self.graphs[lyr][node_id].append(n_id)
                if n_id not in self.graphs[lyr]:
                    self.graphs[lyr][n_id] = []
                self.graphs[lyr][n_id].append(node_id)

                # Prune neighbor if it now exceeds degree bound
                max_conn = self.M0 if lyr == 0 else self.M
                if len(self.graphs[lyr][n_id]) > max_conn:
                    self._prune_connections(n_id, max_conn, lyr)

            # Use the closest candidates as entry points for the next layer down
            ep_list = [nid for _, nid in candidates[:self.ef_construction]]

        # Update entry point if new node has a higher layer
        if node_level > self.max_level:
            self.entry_point = node_id
            self.max_level = node_level

        self.n_nodes += 1

    def _prune_connections(self, node_id: int, max_conn: int, layer: int) -> None:
        """Prune a node's neighbor list to max_conn using the heuristic.

        Called when adding a bidirectional edge causes a node to exceed
        its maximum degree. Re-runs the neighbor selection heuristic
        on the existing neighbors to pick the best max_conn to keep.

        Args:
            node_id: The node whose connections need pruning.
            max_conn: Maximum allowed connections (M or M0).
            layer: The layer where pruning is needed.
        """
        neighbors = self.graphs[layer][node_id]
        if len(neighbors) <= max_conn:
            return

        # Build candidate list with distances
        candidates = []
        for n_id in neighbors:
            dist = l2_distance(self.vectors[node_id], self.vectors[n_id])
            candidates.append((dist, n_id))

        # Select best neighbors via heuristic
        selected = self._select_neighbors_heuristic(node_id, candidates, max_conn)

        # Update adjacency list
        self.graphs[layer][node_id] = [n_id for _, n_id in selected]

    # ------------------------------------------------------------------
    # Bulk build (more efficient than repeated insert)
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, show_progress: bool = True) -> None:
        """Build the index from a batch of vectors.

        More memory-efficient than calling insert() one by one because
        it pre-allocates the vector array.

        Args:
            vectors: All vectors to index, shape (n, dim), float32.
            show_progress: Show a tqdm progress bar.
        """
        n = vectors.shape[0]

        # Pre-allocate vector storage
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._vectors_list = []  # not needed when using build()

        # Insert one by one but skip the per-insert array rebuild
        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Building HNSW index", unit="vec")

        for node_id in iterator:
            self._insert_with_preallocated(node_id)

    def _insert_with_preallocated(self, node_id: int) -> None:
        """Insert a node when self.vectors is already pre-allocated.

        Same logic as insert() but skips vector list management.
        """
        vector = self.vectors[node_id]

        # Assign random layer
        node_level = self._get_random_level()
        self._node_level[node_id] = node_level

        # Ensure graph layers exist
        for lyr in range(node_level + 1):
            if lyr not in self.graphs:
                self.graphs[lyr] = {}
            self.graphs[lyr][node_id] = []

        # Handle first node
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = node_level
            self.n_nodes += 1
            return

        # Phase 1: Greedy descent from top to node_level + 1
        current_ep = self.entry_point
        for lyr in range(self.max_level, node_level, -1):
            result = self._search_layer(vector, [current_ep], ef=1, layer=lyr)
            if result:
                current_ep = result[0][1]

        # Phase 2: Neighbor selection at each layer from node_level down to 0
        ep_list = [current_ep]

        for lyr in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(
                vector, ep_list, ef=self.ef_construction, layer=lyr
            )

            M_lyr = self.M0 if lyr == 0 else self.M
            neighbors = self._select_neighbors_heuristic(
                node_id, candidates, M=M_lyr
            )

            for n_dist, n_id in neighbors:
                self.graphs[lyr][node_id].append(n_id)
                if n_id not in self.graphs[lyr]:
                    self.graphs[lyr][n_id] = []
                self.graphs[lyr][n_id].append(node_id)

                max_conn = self.M0 if lyr == 0 else self.M
                if len(self.graphs[lyr][n_id]) > max_conn:
                    self._prune_connections(n_id, max_conn, lyr)

            ep_list = [nid for _, nid in candidates[:self.ef_construction]]

        if node_level > self.max_level:
            self.entry_point = node_id
            self.max_level = node_level

        self.n_nodes += 1

    # ------------------------------------------------------------------
    # Search (query time)
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
    ) -> list[tuple[float, int]]:
        """Find k approximate nearest neighbors of query.

        Procedure:
        1. Start at the entry point at the top layer.
        2. Greedy descent from top layer to layer 1 (ef=1 at each layer).
        3. At layer 0, beam search with ef=ef_search.
        4. Return top k from the ef results.

        Args:
            query: Query vector, shape (dim,).
            k: Number of nearest neighbors to return.
            ef: Beam width at layer 0 (higher = better recall, slower).
                Must be >= k.

        Returns:
            List of (distance, node_id) sorted by distance ascending,
            at most k elements.
        """
        if self.entry_point is None:
            return []

        ef = max(ef, k)  # ef must be at least k

        # Phase 1: Greedy descent from top layer to layer 1
        current_ep = self.entry_point
        for lyr in range(self.max_level, 0, -1):
            result = self._search_layer(query, [current_ep], ef=1, layer=lyr)
            if result:
                current_ep = result[0][1]

        # Phase 2: Beam search at layer 0 with full ef
        results = self._search_layer(query, [current_ep], ef=ef, layer=0)

        # Return top k
        return results[:k]