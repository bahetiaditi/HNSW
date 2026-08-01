# Filter-Aware HNSW: A Complete Technical Study of Metadata Filtering Strategies for Approximate Nearest Neighbor Search

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Background: The Nearest Neighbor Problem](#2-background-the-nearest-neighbor-problem)
3. [Navigable Small World Graphs](#3-navigable-small-world-graphs)
4. [HNSW: Hierarchical Navigable Small Worlds](#4-hnsw-hierarchical-navigable-small-worlds)
5. [The Select-Neighbors Heuristic](#5-the-select-neighbors-heuristic)
6. [HNSW Parameters and Their Effects](#6-hnsw-parameters-and-their-effects)
7. [The Filtered Search Problem](#7-the-filtered-search-problem)
8. [Three Filtering Strategies](#8-three-filtering-strategies)
9. [The ACORN Paper: What It Does and What We Borrowed](#9-the-acorn-paper-what-it-does-and-what-we-borrowed)
10. [Implementation Architecture](#10-implementation-architecture)
11. [Dataset and Experimental Setup](#11-dataset-and-experimental-setup)
12. [Unfiltered Validation Results](#12-unfiltered-validation-results)
13. [Filtered Benchmark Results](#13-filtered-benchmark-results)
14. [Regime Characterization: The Central Finding](#14-regime-characterization-the-central-finding)
15. [FAISS Comparison and What It Tells Us](#15-faiss-comparison-and-what-it-tells-us)
16. [Limitations and Future Work](#16-limitations-and-future-work)
17. [Interview Talking Points](#17-interview-talking-points)
18. [References](#18-references)

---

## 1. Introduction and Motivation

### 1.1 Why This Project Exists

Every production vector database — Qdrant, Pinecone, Weaviate, Milvus, ChromaDB — supports metadata filtering alongside approximate nearest neighbor (ANN) search. A typical query in a RAG pipeline is not "find the 10 most similar documents" but rather "find the 10 most similar documents *where source = 'internal' AND date > 2024-01-01 AND department = 'engineering'*." The combination of high-dimensional similarity search with structured metadata predicates is the default query pattern in production, not an edge case.

Yet the interaction between the graph-based ANN index and the metadata predicate is non-trivial. The naive approaches — filtering before search, or filtering after search — each fail in predictable and opposite ways. Understanding *when* each approach fails, *why* it fails, and *what the alternative is* constitutes a core piece of systems knowledge for anyone building or operating vector search infrastructure.

### 1.2 What This Project Demonstrates

This project implements HNSW (Hierarchical Navigable Small World graphs) from scratch in Python, builds three metadata filtering strategies on top, and produces a benchmark study on the SIFT1M dataset (1,000,000 vectors, 128 dimensions) characterizing when each filtering strategy wins. The key deliverables are:

1. A working, tested HNSW implementation that matches FAISS recall within ±0.005 across all parameter settings.
2. Three filtering strategies: pre-filter, post-filter, and predicate-aware traversal (inspired by the ACORN paper).
3. A regime characterization showing the crossover points between strategies as selectivity varies from 0.1% to 50%.

The goal is not to beat FAISS in raw speed — FAISS is C++ with SIMD intrinsics, our implementation is Python with Numba. The goal is to *understand* the data structure powering every modern RAG pipeline and to characterize the filtering problem rigorously.

### 1.3 Why This Matters for GenAI Roles

Most candidates applying for Generative AI Scientist or AI Engineer roles have used FAISS, Qdrant, or Pinecone as a black box: `index.add(vectors)`, `index.search(query, k)`. They cannot explain why recall degrades when you add a metadata filter, why some queries return zero results even though matching vectors exist, or why the `flatSearchCutoff` parameter in Weaviate exists. This project provides those answers from first principles.

---

## 2. Background: The Nearest Neighbor Problem

### 2.1 The Problem Statement

Given a database of $n$ vectors in $d$-dimensional space and a query vector $q$, find the $k$ vectors closest to $q$ under some distance metric (typically L2/Euclidean or cosine similarity). This is the *k-nearest neighbor* (KNN) problem.

**Exact solutions** compute the distance from $q$ to every vector in the database and return the $k$ smallest. This is $O(n \cdot d)$ per query — linear in the database size. For $n = 1{,}000{,}000$ and $d = 128$, that is 128 million floating-point operations per query. At scale (billions of vectors, thousands of queries per second), exact search is infeasible.

**Approximate solutions** trade a small amount of accuracy (recall) for dramatically lower query latency. The key insight is that we do not need the *exact* $k$ nearest neighbors — we need *most* of them, *most* of the time. If an approximate method returns 9 of the true 10 nearest neighbors in 1ms instead of all 10 in 100ms, that is an excellent trade in production.

### 2.2 Families of ANN Approaches

There are four main families:

**Tree-based methods** (KD-trees, ball trees, Annoy) partition the space recursively. They work well in low dimensions but degrade as dimensionality increases (the "curse of dimensionality" makes all partitions nearly equidistant). Not competitive at $d = 128$.

**Hash-based methods** (LSH — Locality-Sensitive Hashing) project vectors into hash buckets such that nearby vectors hash to the same bucket with high probability. Theoretically elegant, but the practical recall-speed tradeoff is often worse than graph-based methods, and the memory overhead of multiple hash tables is significant.

**Quantization-based methods** (Product Quantization, IVF-PQ) compress vectors into compact codes and search over the compressed representations. Used in FAISS's IVF-PQ index. Very memory-efficient, but the compression introduces quantization error that limits recall.

**Graph-based methods** (NSW, HNSW, NSG, DiskANN) build a proximity graph where each vector is a node and edges connect nearby vectors. Search is a greedy walk on the graph: start at an entry point, move to the neighbor closest to the query, repeat. Graph-based methods currently dominate the recall-speed Pareto frontier for in-memory ANN search. HNSW is the most widely deployed.

### 2.3 Why Graphs Win

The intuition for why graph-based methods work well is *navigability*. Consider a social network: to find someone in a new city, you might ask a friend who knows someone there, who introduces you to someone in the right neighborhood, who directs you to the specific person. At each hop, you get closer to the target. The same principle applies to proximity graphs: if the graph has a mix of long-range connections (for coarse navigation) and short-range connections (for fine-grained convergence), a greedy walk from any starting point can reach the neighborhood of the query in $O(\log n)$ hops. The remaining cost is exploring that neighborhood to find the true $k$ nearest vectors.

### 2.4 Distance Metric: Squared L2

Throughout this project, all distances are **squared L2** (squared Euclidean distance):

$$d(a, b) = \sum_{i=1}^{d} (a_i - b_i)^2$$

We omit the square root because it is a monotonic transformation — the ranking of distances is identical with or without it. Omitting it saves one `sqrt` per distance computation, which adds up over millions of computations. All "distances" in this document are squared L2 unless stated otherwise.

---

## 3. Navigable Small World Graphs

### 3.1 The Precursor to HNSW

HNSW is an improvement over Navigable Small World (NSW) graphs. Understanding NSW first makes the HNSW design choices clear.

An NSW graph is a single-layer proximity graph built by inserting vectors one at a time. When inserting vector $v$:

1. Start from a random entry point in the existing graph.
2. Greedily walk toward $v$: at each step, move to the neighbor of the current node that is closest to $v$. Continue until no neighbor is closer.
3. Connect $v$ to its $M$ nearest nodes found during the walk (or encountered along the way).

**Search** is the same greedy walk: start from a random entry point, move to the neighbor closest to the query, repeat until convergence.

### 3.2 Why NSW Works (Sort Of)

The key insight is that the insertion order creates a natural mix of edge lengths:

- **Early-inserted nodes** arrive when the graph is sparse. Their $M$ nearest neighbors in the current graph may be far away in absolute terms (because there are few nodes to choose from). These become *long-range* edges.
- **Late-inserted nodes** arrive when the graph is dense. Their $M$ nearest neighbors are truly nearby (because the graph now has many nodes). These become *short-range* edges.

This mix gives the graph small-world properties: long-range edges enable fast coarse navigation (big hops toward the target region), and short-range edges enable precise convergence (small hops to the exact nearest neighbors). The result is $O(\text{polylog}(n))$ search complexity.

### 3.3 Why NSW Is Not Enough

The problem is that the long-range/short-range mix is **accidental** and **fragile**. It depends on insertion order. If you rebuild the graph, you get different edges. The long-range connections of early-inserted nodes decay as more nodes are added (their neighbors shift). And there is no way to control the distribution of edge lengths explicitly.

HNSW fixes this by making the hierarchy *explicit*.

---

## 4. HNSW: Hierarchical Navigable Small Worlds

### 4.1 The Core Idea

HNSW separates the long-range and short-range edges into different *layers*. Instead of one graph, it maintains multiple layers:

- **Top layers** (layer $L$, $L-1$, ...): sparse, with few nodes and long-range connections. Used for coarse navigation.
- **Bottom layer** (layer 0): dense, with all nodes and short-range connections. Used for precise convergence.

Each vector is assigned a maximum layer $l$ by sampling from a geometric distribution. Most vectors exist only on layer 0. A few reach layer 1. Fewer reach layer 2. And so on. The entry point of the entire index is the node with the highest assigned layer.

### 4.2 Layer Assignment

The maximum layer $l$ for a new node is:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \times m_L \rfloor$$

where $m_L = 1 / \ln(M)$ and $M$ is the max connections per node per layer.

With $M = 16$, $m_L \approx 0.36$. The resulting distribution for 1,000,000 nodes in our SIFT1M experiment:

| Layer | Node count | Fraction |
|-------|-----------|----------|
| 0     | 1,000,000 | 100%     |
| 1     | 62,355    | 6.24%    |
| 2     | 3,904     | 0.39%    |
| 3     | 219       | 0.022%   |
| 4     | 15        | 0.0015%  |
| 5     | 1         | 0.0001%  |

This is the *actual* distribution from our SIFT1M index build. Layer 0 is the full dataset. Each subsequent layer contains roughly $1/M$ of the layer below. The top layer has a single node — the global entry point.

### 4.3 Construction: How Insertion Works

To insert a new vector $v$ with assigned max layer $l$:

**Phase 1 — Greedy descent (top to $l+1$):** Starting from the global entry point at the top layer, do a greedy walk with beam width $ef = 1$ at each layer from the top down to layer $l+1$. At each layer, find the single closest node to $v$. This locates the "neighborhood" of $v$ before we start connecting edges.

The purpose of this phase is navigation: we use the sparse upper layers to quickly get close to where $v$ belongs, without the expense of searching the dense lower layers from a distant starting point.

**Phase 2 — Neighbor selection ($l$ down to 0):** At each layer from $l$ down to 0:

1. Run a beam search with width $ef_{construction}$ to find the $ef_{construction}$ nearest candidates to $v$ at this layer.
2. From these candidates, select at most $M$ neighbors using the *select-neighbors heuristic* (see Section 5). At layer 0, the limit is $2M$ instead of $M$.
3. Add bidirectional edges between $v$ and each selected neighbor.
4. For each neighbor that now exceeds its degree limit, prune its neighbor list using the same heuristic.

**Entry point update:** If $l$ is higher than the current maximum layer, $v$ becomes the new global entry point.

### 4.4 Search: How Queries Work

To find the $k$ nearest neighbors of query $q$:

**Phase 1 — Greedy descent (top to layer 1):** Starting from the entry point, greedily walk down with $ef = 1$ at each layer. This navigates to the neighborhood of $q$ in the full graph.

**Phase 2 — Beam search at layer 0:** Run a beam search with width $ef_{search}$ at layer 0. Return the top $k$ results.

### 4.5 The Beam Search (search_layer)

This is the most important subroutine in the entire implementation — and the one we modify for predicate-aware search. It deserves detailed explanation.

The beam search maintains two data structures:

- **Candidates $C$** (min-heap by distance): the frontier of unexplored nodes. The closest candidate is popped first.
- **Results $W$** (max-heap by distance): the best $ef$ nodes found so far. The farthest result is on top (for easy comparison and eviction).

**Initialization:** Add all entry points to both $C$ and $W$. Mark them as visited.

**Main loop:**
1. Pop the closest candidate $c$ from $C$.
2. **Stopping condition:** If $c$ is farther than the farthest element in $W$ *and* $W$ already has $ef$ elements, stop. No remaining candidate in $C$ can improve $W$ (because $C$ is a min-heap, so everything else is even farther).
3. For each neighbor $n$ of $c$ in the current layer:
   - If $n$ has been visited, skip.
   - Mark $n$ as visited.
   - Compute $\text{dist}(q, n)$.
   - If $\text{dist}(q, n) < \text{dist}(q, \text{farthest in } W)$ or $|W| < ef$:
     - Add $n$ to $C$ (it might lead to even better nodes).
     - Add $n$ to $W$.
     - If $|W| > ef$, remove the farthest element from $W$ (keep only the best $ef$).

**Output:** Return $W$ sorted by distance (ascending).

The key insight is that $C$ and $W$ serve different purposes: $C$ is the *exploration frontier* (which nodes to visit next), and $W$ is the *result set* (the best nodes found). This separation is what enables predicate-aware search — we can apply different rules to each.

---

## 5. The Select-Neighbors Heuristic

### 5.1 Why Not Just Take the Closest M?

The simplest neighbor selection would be: from the $ef_{construction}$ candidates, take the $M$ closest to the insertion node. This is the "simple" heuristic (Algorithm 3 in the paper).

The problem is that this creates *clusters*: if the $M$ closest nodes are all in the same direction from $v$, the edges all point the same way. The graph has poor connectivity in other directions. A query approaching $v$ from a different direction might not find a path to $v$.

### 5.2 The RNG (Relative Neighborhood Graph) Heuristic

The select-neighbors heuristic (Algorithm 4 in the HNSW paper) approximates a Relative Neighborhood Graph. It works like this:

Given candidate set $C$ sorted by distance to insertion node $v$, and target count $M$:

1. Initialize an empty selected set $S$.
2. For each candidate $c$ in $C$ (closest first):
   - If $|S| = M$, stop.
   - Check if any already-selected node $s \in S$ satisfies $\text{dist}(c, s) < \text{dist}(c, v)$.
   - If **yes**: skip $c$. There is already a shorter path from $c$ to $v$ through $s$, so the edge $c \to v$ is "redundant."
   - If **no**: add $c$ to $S$. The edge $c \to v$ provides connectivity in a direction not already covered.

### 5.3 Geometric Intuition

Imagine $v$ is at the origin. If we have already selected neighbor $s_1$ at position $(1, 0)$, and candidate $c$ is at position $(1.1, 0)$, then $\text{dist}(c, s_1) = 0.01$ which is less than $\text{dist}(c, v) = 1.21$. So $c$ is redundant — we can already reach $c$'s neighborhood via $s_1$. But if candidate $c_2$ is at position $(0, 5)$, then no selected neighbor is closer to $c_2$ than $v$ is, because $c_2$ is in a completely different direction. So $c_2$ is kept.

The heuristic ensures that the selected neighbors *span diverse directions* around $v$. This is precisely the Relative Neighborhood Graph property: two nodes are RNG-neighbors if there is no third node closer to both of them than they are to each other.

### 5.4 Why This Matters for Recall

The heuristic is what gives HNSW its good navigability. Without it, the graph degenerates into clusters with poor inter-cluster connectivity. With it, the graph maintains connections in all directions, enabling the greedy search to converge from any starting direction. In our tests, removing the heuristic (using the simple closest-$M$ selection) reduced recall@10 significantly.

---

## 6. HNSW Parameters and Their Effects

### 6.1 M (Maximum Connections per Layer)

$M$ controls the degree of each node. Higher $M$ means more edges, better connectivity, and higher recall, but also more memory (each edge stores a neighbor ID) and slower search (more neighbors to evaluate per hop).

- Layer 0 uses $2M$ connections (because layer 0 handles most search traffic and denser connectivity helps recall).
- Upper layers use $M$ connections.

**Typical values:** 12–48. We use $M = 16$, the most common default.

**Memory impact:** For 1M vectors with $M = 16$, layer 0 has up to $2 \times 16 = 32$ neighbors per node, requiring $1M \times 32 \times 4$ bytes = ~128 MB just for layer 0 adjacency lists, plus the vectors themselves (~512 MB for 128-dim float32). Total index size is roughly 700 MB.

### 6.2 ef_construction (Build-Time Beam Width)

$ef_{construction}$ controls the quality of the graph during construction. When inserting a node, we search for $ef_{construction}$ candidates, then select $M$ from them using the heuristic. Higher $ef_{construction}$ means we find better candidates, select better neighbors, and build a better graph — but construction is slower.

**Typical values:** 100–400. We use $ef_{construction} = 200$.

**Build time impact:** Our SIFT1M index (1M vectors, $M = 16$, $ef_{construction} = 200$) took 58 minutes to build in Python. FAISS built the same index in 203 seconds (C++). The ~17× difference is the Python vs C++ overhead.

### 6.3 ef_search (Query-Time Beam Width)

$ef_{search}$ is the main tuning knob at query time. It controls the recall–latency tradeoff directly: higher $ef$ means wider beam search, more nodes explored, better recall, but slower queries.

$ef$ must be $\geq k$ (you need to find at least $k$ candidates to return $k$ results).

**Our SIFT1M recall vs ef (unfiltered):**

| ef_search | Recall@10 | p50 Latency (ms) | QPS    |
|-----------|-----------|-------------------|--------|
| 16        | 0.8161    | 0.43              | 2,322  |
| 32        | 0.9150    | 0.69              | 1,483  |
| 64        | 0.9703    | 1.22              | 856    |
| 128       | 0.9919    | 2.13              | 489    |
| 256       | 0.9978    | 3.85              | 269    |

The recall-latency tradeoff is smooth and predictable: doubling ef roughly doubles latency and adds 3–5 percentage points of recall.

### 6.4 mL (Layer Assignment Parameter)

$m_L = 1/\ln(M)$ is derived from $M$, not independently tunable. With $M = 16$:

$$m_L = \frac{1}{\ln(16)} \approx 0.361$$

This means the expected maximum layer for a random node is 0.361 (most nodes stay at layer 0). The probability of a node reaching layer $l$ drops as roughly $(1/M)^l$. For $M = 16$ and $n = 1{,}000{,}000$, the top layer is typically 4–5.

The design ensures that upper layers are geometrically sparser: each layer has roughly $1/M$ of the nodes in the layer below. This gives $O(\log n)$ layers and $O(\log n)$ greedy descent steps.

---

## 7. The Filtered Search Problem

### 7.1 Problem Statement

Given a query vector $q$, a metadata predicate $P$ (e.g., `category == 3`), and a count $k$, find the $k$ nearest neighbors of $q$ **among the vectors satisfying $P$**.

The **selectivity** $s$ is the fraction of vectors matching $P$. At selectivity $s = 0.5$ (50%), half the vectors match. At $s = 0.001$ (0.1%), only one in a thousand matches.

### 7.2 Why This Is Hard on Graphs

The fundamental tension is that the HNSW graph was built to navigate toward the **globally** nearest vectors, not the nearest **matching** vectors. The graph's edges encode proximity in the full unfiltered space. When you traverse the graph looking for matching vectors, most of the neighbors you encounter are non-matching — the graph is leading you toward the globally nearest vectors, which happen to not satisfy the predicate.

Consider a concrete example: query $q$ is looking for `category == 3` at 1% selectivity on 1M vectors. The 10 globally nearest vectors to $q$ are probably not category 3 (only 1% of vectors are). The graph will navigate toward those 10 globally nearest vectors. The beam search converges in their neighborhood. The nearest category-3 vector might be the 100th or 500th nearest vector overall — far outside the beam search's reach at typical ef values.

### 7.3 The Three Regimes

The difficulty of filtered search depends on selectivity:

**High selectivity ($s > 25\%$):** Most vectors match. The globally nearest vectors and the nearest matching vectors overlap substantially. Filtering is easy — you barely lose any candidates.

**Moderate selectivity ($5\% < s < 25\%$):** Some globally nearest vectors match, but many don't. Careful search strategies can still find matching vectors efficiently.

**Low selectivity ($s < 5\%$):** Very few globally nearest vectors match. The graph's navigational structure actively works against you — it leads to non-matching neighborhoods. This is where strategy choice matters enormously.

---

## 8. Three Filtering Strategies

### 8.1 Strategy 1: Pre-Filter

**Mechanism:**
1. Scan the metadata array to find all vector IDs where `metadata[id] == target_category`.
2. Extract those vectors into a submatrix.
3. Compute the squared L2 distance from the query to every matching vector.
4. Return the $k$ smallest.

This is brute-force exact KNN over the matching subset. No graph is involved.

**Recall:** Always 1.0 (exact search).

**Complexity:** $O(s \cdot n \cdot d)$ per query, where $s$ is selectivity, $n$ is total vectors, $d$ is dimensionality.

**When it wins:** At very low selectivity, the matching subset is tiny. At $s = 0.1\%$ on 1M vectors, there are only 1,000 matching vectors. Computing 1,000 × 128 = 128,000 distance operations is trivially fast — sub-millisecond. No graph traversal can beat brute force over 1,000 vectors.

**When it loses:** At high selectivity, the matching subset is huge. At $s = 50\%$, brute force over 500,000 vectors at 128 dimensions is far slower than a graph search that converges in ~100 hops.

**Our SIFT1M results (pre-filter latency by selectivity):**

| Selectivity | Matching vectors | p50 Latency (ms) |
|-------------|-----------------|-------------------|
| 0.1%        | 1,000           | 0.62              |
| 1%          | 10,000          | 2.57              |
| 5%          | 50,000          | 10.62             |
| 10%         | 100,000         | 20.05             |
| 25%         | 250,000         | 41.47             |
| 50%         | 500,000         | 76.52             |

The latency scales linearly with the number of matching vectors, as expected. The crossover point where pre-filter becomes slower than graph methods is around 5–10% selectivity.

**Production analogy:** This is why Weaviate has a `flatSearchCutoff` parameter: when the matching subset is small enough, it switches from graph search to brute force automatically.

### 8.2 Strategy 2: Post-Filter

**Mechanism:**
1. Run standard HNSW search with an inflated beam width: $ef = k \times \text{oversample\_factor}$.
2. From the returned candidates, discard those where `metadata[id] != target_category`.
3. Return the top $k$ from the matching candidates.

This is the simplest strategy — one line of filtering logic on top of standard HNSW search.

**Recall:** Depends entirely on selectivity and oversample factor.

**Complexity:** Same as standard HNSW search at the inflated ef, plus the filtering pass (negligible). $O(ef \cdot M \cdot d)$ approximately.

**When it wins:** At high selectivity, most candidates in the search results match the predicate anyway. With $s = 50\%$ and oversample factor 10, you search with $ef = 100$ and expect ~50 matching results — plenty to find the true $k = 10$ nearest.

**When it fails catastrophically:** At low selectivity, the beam search converges toward the globally nearest vectors, which are overwhelmingly non-matching. Even with large oversample factors, the $k$ nearest matching vectors may never enter the candidate set. The search is looking in the wrong place — the graph doesn't "know" about the filter.

**Our SIFT1M results (post-filter recall by selectivity and oversample factor):**

| Selectivity | osf=5  | osf=10 | osf=20 | osf=50 |
|-------------|--------|--------|--------|--------|
| 0.1%        | 0.0049 | 0.0098 | 0.0199 | 0.0510 |
| 1%          | 0.0449 | 0.0922 | 0.1904 | 0.4890 |
| 5%          | 0.2495 | 0.4891 | 0.8586 | 0.9926 |
| 10%         | 0.4949 | 0.8553 | 0.9791 | 0.9972 |
| 25%         | 0.8763 | 0.9651 | 0.9909 | 0.9993 |
| 50%         | 0.9222 | 0.9747 | 0.9940 | 0.9990 |

**The recall collapse is dramatic.** At 0.1% selectivity, even with oversample factor 50 (searching with $ef = 500$), recall is only 0.051. Out of the 500 candidates HNSW returns, only about 5 are category 0 — and most of those are not among the true 10 nearest category-0 vectors. Doubling the oversample factor only approximately doubles the recall, so reaching useful recall (say 0.9) would require oversample factors in the thousands, which defeats the purpose of graph search.

**Why this happens, mechanically:** The beam search starts at the entry point and greedily navigates toward the globally nearest vectors to the query. At 0.1% selectivity, 99.9% of the vectors it encounters along the way are non-matching. The beam width ef controls how many candidates it keeps, but it cannot control where the graph's edges lead. Even at $ef = 500$, the search converges in a local neighborhood of the globally nearest vectors, and the nearest category-0 vectors may be far away in graph distance (even if they are relatively close in vector space).

### 8.3 Strategy 3: Predicate-Aware Traversal

**Mechanism:** Modify the beam search so that:

- The **candidate frontier $C$** accepts ALL neighbors, regardless of predicate. Non-matching nodes are used as "stepping stones" — the search walks through them to maintain graph connectivity.
- The **result set $W$** accepts ONLY nodes satisfying the predicate. Only matching nodes appear in the output.

Everything else — the initialization, the main loop, the stopping condition — follows the same structure as standard beam search, with one adaptation: the stopping condition checks the farthest *matching* node in $W$, not the farthest node overall.

**The key insight:** Non-matching nodes are not useless — they provide navigational structure. The edge from a non-matching node $n_1$ to another non-matching node $n_2$ to a matching node $m$ is a valid path. By keeping non-matching nodes in the candidate frontier, the search can "hop through" regions of the graph that contain no matching vectors, eventually reaching matching ones on the other side.

This is the core contribution of the ACORN paper: *filter the result set, not the traversal frontier*.

**Where predicate-aware search is applied:** Only at layer 0. At upper layers (1 and above), we use standard greedy descent with $ef = 1$ and no filtering. The upper layers have very few nodes (62K at layer 1, 3.9K at layer 2, etc.), and filtering there would cripple navigation — if the entry point at layer 3 doesn't match the predicate, we would have nowhere to start. So we navigate to the right neighborhood unfiltered, then switch to predicate-aware search at layer 0 where the actual result set is built.

**The stopping condition subtlety:** In standard beam search, we stop when the closest remaining candidate is farther than the farthest result. In predicate-aware search, $W$ contains only matching nodes but $C$ contains all nodes (matching and non-matching). We stop when the closest remaining candidate (matching or not) is farther than the farthest *matching* result in $W$ and $W$ has at least $ef$ matching results. If $W$ never accumulates enough matching nodes — which can happen at extreme selectivity — the search exhausts the reachable portion of the graph and returns what it found.

**Our SIFT1M results (predicate-aware recall and latency by selectivity and ef):**

| Selectivity | ef=16  | ef=32  | ef=64  | ef=128 | ef=256 |
|-------------|--------|--------|--------|--------|--------|
| **Recall@10** | | | | | |
| 0.1%        | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| 1%          | 0.9964 | 0.9994 | 0.9999 | 1.0000 | 1.0000 |
| 5%          | 0.9836 | 0.9962 | 0.9993 | 0.9999 | 0.9999 |
| 10%         | 0.9670 | 0.9935 | 0.9988 | 0.9995 | 0.9997 |
| 25%         | 0.9248 | 0.9770 | 0.9947 | 0.9993 | 0.9996 |
| 50%         | 0.8664 | 0.9465 | 0.9843 | 0.9964 | 0.9992 |
| **p50 Latency (ms)** | | | | | |
| 0.1%        | 112.32 | 206.98 | 395.09 | 762.82 | 1564.81|
| 1%          | 16.78  | 30.88  | 55.81  | 98.90  | 177.61 |
| 5%          | 4.62   | 8.26   | 14.76  | 26.97  | 46.99  |
| 10%         | 2.53   | 4.79   | 8.56   | 15.03  | 27.50  |
| 25%         | 1.21   | 2.18   | 4.00   | 7.14   | 12.69  |
| 50%         | 0.77   | 1.29   | 2.29   | 4.07   | 7.50   |

Two patterns emerge clearly:

1. **Recall is consistently high across all selectivities.** Even at 0.1% selectivity, predicate-aware achieves recall 0.9998 — compared to post-filter's 0.0049-0.0510. The stepping-stone mechanism works.

2. **Latency scales inversely with selectivity.** At 0.1% selectivity, the search must visit thousands of non-matching stepping stones to find 10 matching ones among 1,000 out of 1,000,000. At 50%, matching nodes are everywhere and the search converges quickly.

---

## 9. The ACORN Paper: What It Does and What We Borrowed

### 9.1 Paper Overview

ACORN (*Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*, Patel et al., SIGMOD 2024) addresses filtered ANN search on HNSW-family graphs. The key contribution is a framework for predicate-aware graph traversal with two variants:

**ACORN-$\gamma$ (construction-time modification):** During HNSW construction, expand each node's neighbor list by a factor $\gamma = 1/s_{min}$, where $s_{min}$ is the minimum expected selectivity. If $s_{min} = 0.01$ (1%), each node gets $\gamma = 100 \times$ more neighbors. This ensures that even the *predicate subgraph* (the graph restricted to matching nodes) has enough connectivity for efficient traversal. After expansion, a compression heuristic ($M_\beta$) prunes the neighbor lists to manage memory.

**ACORN-1 ($\gamma = 1$, search-time only):** Use the standard HNSW construction (no neighbor expansion). At search time, compensate by doing 2-hop neighbor expansion: when evaluating a node's neighbors, also look at their neighbors. This effectively doubles the search frontier at each step, increasing the chance of reaching matching nodes through longer paths.

Both variants use the same search-time principle: filter the result set but not the traversal frontier.

### 9.2 What We Took from ACORN

**The core search-time mechanism:** Filtering the result set $W$ while leaving the candidate frontier $C$ unfiltered. This is the fundamental insight that makes predicate-aware search work, and it is conceptually simple: one `if` statement in the beam search loop determines whether a node enters $W$ (only if it matches) or just $C$ (always).

**The conceptual framework:** The idea of comparing pre-filter, post-filter, and predicate-aware as three regimes of filtered search, characterized by selectivity.

### 9.3 What We Simplified (and Why)

**No ACORN-$\gamma$ construction modification.** We build a standard HNSW graph with no neighbor expansion. Reasons:

1. ACORN-$\gamma$ requires choosing $\gamma$ upfront, which means committing to a minimum selectivity at build time. If a query arrives with selectivity below $s_{min}$, the graph may still lack sufficient predicate subgraph connectivity.
2. The memory cost of $\gamma$-expansion is significant: neighbor lists grow by factor $\gamma$, and the compression heuristic only partially recovers this.
3. Our goal is to demonstrate that the search-time insight works even on a standard graph. This is a cleaner experiment: we can isolate the effect of the search-time filtering from the construction-time modification.

**No 2-hop neighbor expansion (ACORN-1).** When evaluating a node's neighbors, we do not also look at their neighbors. This means our predicate-aware search on a standard graph has less effective connectivity than ACORN-1. At extreme selectivities (0.1%), this manifests as high latency: the search must traverse many hops through non-matching stepping stones because the direct neighbor lists are not expanded. We document this as a limitation and reference ACORN-1's 2-hop expansion as the known fix.

**Single equality predicate.** ACORN supports arbitrary predicates (regex, range, contains). We restrict to `category == c` for simplicity. This does not affect the core search mechanism but limits the practical applicability.

**No predicate clustering analysis.** ACORN formally defines *query correlation* — whether matching vectors tend to cluster near or far from queries. Our synthetic metadata is randomly assigned (no spatial correlation), so there is no predicate clustering. In real-world data, predicate clustering can either help (matching vectors are nearby, so the graph naturally leads to them) or hurt (matching vectors are in a distinct cluster, so the graph must navigate farther).

### 9.4 Honest Assessment

Our implementation is **not a reproduction of ACORN**. It is an implementation of the core search-time insight from ACORN, applied to a standard HNSW graph, with a rigorous benchmark study. The value is in demonstrating the insight works and characterizing the regimes where each strategy wins — not in matching ACORN's full performance envelope.

If we wanted ACORN-level performance at 0.1% selectivity, we would need either ACORN-$\gamma$'s construction expansion or ACORN-1's 2-hop search expansion. Our results clearly show this: the predicate-aware recall is near-perfect at 0.1%, but the latency (112ms) is too high for production. The construction-time modification would reduce this by ensuring the predicate subgraph has direct connections between matching nodes, avoiding the need to traverse long chains of stepping stones.

---

## 10. Implementation Architecture

### 10.1 Repository Structure

```
HNSW/
├── src/
│   ├── distance.py     — Numba-JIT L2 distance kernels
│   ├── hnsw.py         — Core HNSW (insert, search, neighbor selection)
│   ├── filtering.py    — Three filtering strategies
│   └── utils.py        — I/O, metadata generation, ground truth, recall
├── benchmarks/
│   ├── bench_unfiltered.py    — Our HNSW vs FAISS (sanity check)
│   ├── bench_filtered.py      — 3 strategies × 6 selectivities
│   ├── compute_ground_truth.py — Brute-force filtered GT (cached)
│   └── plot_results.py        — Pareto curves, regime plots
├── data/
│   └── download_sift1m.py     — HuggingFace download + verify
├── results/                   — CSV results + plots
├── writeup/                   — This document
└── tests/
    ├── test_distance.py       — 9 tests
    ├── test_hnsw.py           — 9 tests
    └── test_filtering.py      — 11 tests
```

### 10.2 Design Decisions

**Numba for the distance kernel, Python for everything else.** The inner loop of HNSW — computing L2 distance between two 128-dim vectors — runs billions of times. This must be fast. We use Numba `@njit(fastmath=True, cache=True)` for the distance functions, which compiles them to native code at first call. Everything else (graph traversal, heap operations, neighbor selection) is pure Python. This is a deliberate tradeoff: Python graph traversal is ~4–10× slower than C++, but the code is readable and modifiable.

**Adjacency lists as nested dicts.** The graph structure is `dict[layer → dict[node_id → list[neighbor_ids]]]`. This is less memory-efficient than a flat array layout (which FAISS uses), but it supports dynamic insertion and makes the code straightforward. For a production implementation, you would use contiguous arrays.

**Separate build() method.** The `insert()` method rebuilds the numpy vector array on every call (appending to a list and converting). The `build()` method pre-allocates the array and uses an internal `_insert_with_preallocated()` that skips this overhead. For 1M vectors, this saves significant time.

**Squared L2 everywhere.** No square roots are ever computed. All distances are squared L2. This is consistent throughout the codebase — distance kernels, heap comparisons, neighbor selection heuristic, recall computation.

### 10.3 Conceptual Flow: How a Predicate-Aware Query Works

Let us trace a predicate-aware search for the 10 nearest category-3 vectors to query $q$ with $ef = 64$:

1. **Start at the entry point** (node 971000 in our SIFT1M index, layer 5). This is a single node at the top layer.

2. **Greedy descent (layers 5 → 1):** At each layer, find the single nearest node to $q$ using standard beam search with $ef = 1$. No filtering. This takes 4 hops total and navigates to the general neighborhood of $q$ in the full graph.

3. **Predicate-aware beam search at layer 0:**
   - Initialize candidates $C$ and results $W$ with the entry node from layer 1.
   - Pop the closest candidate. It's probably not category 3. Add ALL its neighbors (matching or not) to $C$. Add it to $W$ **only if** it is category 3.
   - Continue popping candidates and expanding. Non-matching nodes serve as stepping stones — they stay in $C$ (so we keep exploring through them) but never enter $W$.
   - Category-3 nodes that are encountered are added to $W$. As $W$ fills to 64 matching nodes, the stopping condition tightens: we stop when the closest unprocessed candidate is farther than the farthest matching node in $W$.
   - At 10% selectivity, roughly 1 in 10 neighbors is category 3. The search explores ~640 total nodes to collect 64 matching ones.
   - At 0.1% selectivity, only 1 in 1,000 neighbors matches. The search must explore tens of thousands of nodes, hopping through long chains of non-matching stepping stones.

4. **Return top 10** from $W$, sorted by distance.

### 10.4 Ground Truth Computation

For each selectivity level, we compute brute-force exact filtered KNN:

1. Find all vector IDs matching the target category.
2. For each of the 10,000 queries (or 1,000 in the `--n-queries` variant), compute the squared L2 distance to every matching vector.
3. Return the 10 smallest as ground truth.

This is cached as `.npy` files — one per selectivity level. The computation uses the identity $\|q - v\|^2 = \|q\|^2 + \|v\|^2 - 2q \cdot v$ for efficient matrix computation via numpy broadcasting.

An edge case was discovered during development: when the matching subset has exactly $k$ vectors, `np.argpartition(dists, k)` fails because `kth` must be strictly less than the array length. The fix falls back to `np.argsort` in this case.

---

## 11. Dataset and Experimental Setup

### 11.1 SIFT1M

**Source:** Jégou, Douze & Schmid (2011), originally from INRIA Holidays SIFT descriptors.

**Contents:**
- 1,000,000 base vectors (128-dim, float32) — the index
- 10,000 query vectors (128-dim, float32) — the test queries
- 100 ground-truth nearest neighbors per query (unfiltered)

**File format:** `.fvecs` (binary). Each vector is stored as `[dim: int32] [v1: float32] ... [v_dim: float32]`, so each 128-dim vector occupies $4 + 128 \times 4 = 516$ bytes. The base file is ~516 MB.

**Why SIFT1M:** It is the standard benchmark for ANN search. The vectors have enough structure (they are SIFT visual descriptors, not random) to produce realistic graph topology. It is large enough (1M vectors) that the filtering problem is non-trivial. And it fits in laptop RAM (~700 MB for the index).

### 11.2 Synthetic Metadata

Each base vector is assigned a category from 0 to 19 using a skewed distribution designed to hit specific selectivity targets:

| Category | Count     | Selectivity |
|----------|-----------|-------------|
| 0        | 1,000     | 0.1%        |
| 1        | 10,000    | 1%          |
| 2        | 50,000    | 5%          |
| 3        | 100,000   | 10%         |
| 4        | 250,000   | 25%         |
| 5        | 500,000   | 50%         |
| 6–19     | ~6,429 each | ~0.64% each |

Categories are shuffled randomly across vectors — no spatial clustering by category. This is a deliberate choice: it means the difficulty of filtered search comes purely from selectivity, not from whether matching vectors happen to cluster near the query.

**Random seed = 42** for reproducibility.

### 11.3 Index Configuration

- HNSW: $M = 16$, $ef_{construction} = 200$, seed = 42
- Build time: ~58 minutes (Python, MacBook Air)
- Layer distribution: 1M / 62K / 3.9K / 219 / 15 / 1 across layers 0–5
- Entry point: node 971000 (layer 5)
- FAISS baseline: `IndexHNSWFlat(128, 16)` with $ef_{construction} = 200$

### 11.4 Parameter Sweeps

- **Pre-filter:** No parameters (exact search). One measurement per selectivity.
- **Post-filter:** Oversample factor $\in \{5, 10, 20, 50\}$.
- **Predicate-aware:** $ef \in \{16, 32, 64, 128, 256\}$.
- **FAISS post-filter:** Oversample factor $\in \{5, 10, 20, 50\}$.

### 11.5 Methodology

- 1,000 queries per configuration (from the SIFT1M query set).
- 100 warmup queries discarded before timing.
- Single-threaded execution.
- Metrics: recall@10 (averaged over queries) and p50 query latency (ms).
- Ground truth: brute-force exact filtered KNN, pre-computed and cached.

---

## 12. Unfiltered Validation Results

Before running filtered benchmarks, we validated that our HNSW implementation is correct by comparing against FAISS on unfiltered SIFT1M.

### 12.1 Results

| ef  | Ours Recall@10 | FAISS Recall@10 | Recall Δ  | Ours p50 (ms) | FAISS p50 (ms) |
|-----|---------------|-----------------|-----------|---------------|----------------|
| 16  | 0.8161        | 0.8110          | +0.0051   | 0.43          | 0.10           |
| 32  | 0.9150        | 0.9108          | +0.0042   | 0.69          | 0.17           |
| 64  | 0.9703        | 0.9674          | +0.0028   | 1.22          | 0.31           |
| 128 | 0.9919        | 0.9912          | +0.0006   | 2.13          | 0.60           |
| 256 | 0.9978        | 0.9976          | +0.0002   | 3.85          | 1.10           |

### 12.2 Analysis

**Recall:** Our implementation matches or very slightly exceeds FAISS across all ef values. The positive delta (+0.005 at ef=16, converging to +0.0002 at ef=256) suggests our heuristic produces marginally better graph connectivity — likely due to subtle differences in tie-breaking or random number generation. The key point is that the recalls are within ±0.005, confirming our graph construction is correct.

**Latency:** We are 3.5–4× slower than FAISS. This is entirely expected: FAISS uses C++ with SIMD intrinsics for distance computation and cache-optimized memory layouts for adjacency lists. Our distance kernel is Numba-compiled but our graph traversal is pure Python with dictionary lookups and heapq operations. The latency gap is irrelevant for the filtering study because all three strategies use the same implementation — the relative comparisons are valid.

**Verdict:** PASS. Recall@10 = 0.9978 at ef=256 exceeds the 0.95 threshold. Safe to proceed to filtering strategies.

---

## 13. Filtered Benchmark Results

### 13.1 Complete Results Table

Below are the full results from the SIFT1M benchmark (1,000,000 vectors, 1,000 queries).

#### 13.1.1 Selectivity = 0.1% (1,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) | p95 Latency (ms) |
|---------------------|-----------|-----------|-------------------|-------------------|
| Pre-filter          | —         | 1.0000    | 0.62              | 0.67              |
| Post-filter         | osf=5     | 0.0049    | 1.02              | 1.24              |
| Post-filter         | osf=10    | 0.0098    | 1.79              | 2.09              |
| Post-filter         | osf=20    | 0.0199    | 3.21              | 3.84              |
| Post-filter         | osf=50    | 0.0510    | 7.05              | 8.57              |
| Predicate-aware     | ef=16     | 0.9998    | 112.32            | 163.19            |
| Predicate-aware     | ef=32     | 0.9999    | 206.98            | 274.67            |
| Predicate-aware     | ef=64     | 0.9999    | 395.09            | 474.97            |
| Predicate-aware     | ef=128    | 0.9999    | 762.82            | 889.59            |
| Predicate-aware     | ef=256    | 0.9999    | 1564.81           | 1729.26           |
| FAISS post-filter   | osf=10    | 0.0096    | 0.52              | 0.61              |
| FAISS post-filter   | osf=50    | 0.0514    | 2.41              | 2.88              |

**Verdict:** Pre-filter wins decisively. Brute force over 1,000 vectors: 0.62ms, perfect recall. Post-filter is completely broken (recall < 0.06). Predicate-aware achieves near-perfect recall but at 112–1565ms — it must traverse the entire graph to find 10 matching nodes among 1,000 in 1M.

#### 13.1.2 Selectivity = 1% (10,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) |
|---------------------|-----------|-----------|-------------------|
| Pre-filter          | —         | 1.0000    | 2.57              |
| Post-filter         | osf=5     | 0.0449    | 1.01              |
| Post-filter         | osf=10    | 0.0922    | 1.76              |
| Post-filter         | osf=20    | 0.1904    | 3.23              |
| Post-filter         | osf=50    | 0.4890    | 7.18              |
| Predicate-aware     | ef=16     | 0.9964    | 16.78             |
| Predicate-aware     | ef=32     | 0.9994    | 30.88             |
| Predicate-aware     | ef=64     | 0.9999    | 55.81             |
| FAISS post-filter   | osf=50    | 0.4914    | 2.39              |

**Verdict:** Pre-filter still wins. 2.57ms for perfect recall. Post-filter is still poor (max recall 0.49 at osf=50). Predicate-aware has excellent recall (0.996+) but latency is 17–56ms — slower than pre-filter.

#### 13.1.3 Selectivity = 5% (50,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) |
|---------------------|-----------|-----------|-------------------|
| Pre-filter          | —         | 1.0000    | 10.62             |
| Post-filter         | osf=5     | 0.2495    | 1.05              |
| Post-filter         | osf=10    | 0.4891    | 1.83              |
| Post-filter         | osf=20    | 0.8586    | 3.21              |
| Post-filter         | osf=50    | 0.9926    | 7.30              |
| Predicate-aware     | ef=16     | 0.9836    | 4.62              |
| Predicate-aware     | ef=32     | 0.9962    | 8.26              |
| Predicate-aware     | ef=64     | 0.9993    | 14.76             |
| FAISS post-filter   | osf=50    | 0.9922    | 2.38              |

**Verdict:** The crossover zone. Pre-filter takes 10.62ms — now slower than some graph methods. Predicate-aware at ef=16 achieves recall 0.984 at 4.62ms. Post-filter needs osf=50 (7.30ms) to reach similar recall (0.993). Predicate-aware wins the recall-latency Pareto frontier.

#### 13.1.4 Selectivity = 10% (100,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) |
|---------------------|-----------|-----------|-------------------|
| Pre-filter          | —         | 1.0000    | 20.05             |
| Post-filter         | osf=10    | 0.8553    | 1.84              |
| Post-filter         | osf=20    | 0.9791    | 3.30              |
| Post-filter         | osf=50    | 0.9972    | 7.34              |
| Predicate-aware     | ef=16     | 0.9670    | 2.53              |
| Predicate-aware     | ef=32     | 0.9935    | 4.79              |
| Predicate-aware     | ef=64     | 0.9988    | 8.56              |
| FAISS post-filter   | osf=20    | 0.9768    | 0.96              |

**Verdict:** Pre-filter is now clearly slower (20ms). Post-filter at osf=20 reaches recall 0.979 at 3.30ms. Predicate-aware at ef=32 achieves recall 0.994 at 4.79ms — comparable. Both significantly outperform pre-filter on latency.

#### 13.1.5 Selectivity = 25% (250,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) |
|---------------------|-----------|-----------|-------------------|
| Pre-filter          | —         | 1.0000    | 41.47             |
| Post-filter         | osf=5     | 0.8763    | 1.06              |
| Post-filter         | osf=10    | 0.9651    | 1.78              |
| Post-filter         | osf=20    | 0.9909    | 3.27              |
| Predicate-aware     | ef=16     | 0.9248    | 1.21              |
| Predicate-aware     | ef=32     | 0.9770    | 2.18              |
| Predicate-aware     | ef=64     | 0.9947    | 4.00              |
| FAISS post-filter   | osf=10    | 0.9594    | 0.52              |

**Verdict:** Pre-filter is impractical (41ms). Post-filter and predicate-aware converge in performance. Post-filter at osf=10 (recall 0.965, 1.78ms) is slightly better on latency than predicate-aware at ef=32 (recall 0.977, 2.18ms). Both are excellent.

#### 13.1.6 Selectivity = 50% (500,000 matching vectors)

| Strategy            | Parameter | Recall@10 | p50 Latency (ms) |
|---------------------|-----------|-----------|-------------------|
| Pre-filter          | —         | 1.0000    | 76.52             |
| Post-filter         | osf=5     | 0.9222    | 1.06              |
| Post-filter         | osf=10    | 0.9747    | 1.84              |
| Post-filter         | osf=20    | 0.9940    | 3.35              |
| Predicate-aware     | ef=16     | 0.8664    | 0.77              |
| Predicate-aware     | ef=32     | 0.9465    | 1.29              |
| Predicate-aware     | ef=64     | 0.9843    | 2.29              |
| FAISS post-filter   | osf=10    | 0.9742    | 0.52              |

**Verdict:** Post-filter and predicate-aware are neck and neck. Pre-filter is impractical (77ms). At this selectivity, half the vectors match the predicate, so post-filtering discards very few candidates — the filter barely affects search quality.

---

## 14. Regime Characterization: The Central Finding

### 14.1 The Three Regimes

The results reveal three clear regimes:

**Regime A: Very low selectivity (0.1%–1%).** Pre-filter wins. The matching subset is small enough (1,000–10,000 vectors) that brute force is faster than any graph traversal. Pre-filter achieves 0.62–2.57ms with perfect recall. Predicate-aware achieves near-perfect recall but at 17–112ms (too slow). Post-filter is completely broken (recall < 0.05 at 0.1%).

**Regime B: Moderate selectivity (5%–10%).** Predicate-aware wins the recall-latency tradeoff. At 5% selectivity, predicate-aware at ef=16 delivers recall 0.984 at 4.62ms. Post-filter needs osf=50 (7.30ms) to reach comparable recall (0.993). Pre-filter is getting slow (10–20ms).

**Regime C: High selectivity (25%–50%).** Post-filter becomes competitive. At 25%, post-filter at osf=10 achieves recall 0.965 at 1.78ms. Predicate-aware at ef=32 achieves recall 0.977 at 2.18ms. Both are excellent; the difference is marginal. At 50%, they converge further.

### 14.2 The Crossover Points

- **Pre-filter → Predicate-aware crossover** at ~5% selectivity. Below 5%, pre-filter's O(n·s) cost is lower than graph traversal. Above 5%, graph methods are faster.
- **Post-filter → useful recall crossover** at ~5–10% selectivity. Below 5%, post-filter recall is below 0.5 regardless of oversample factor. Above 10%, post-filter reaches 0.85+ recall at reasonable oversample factors.
- **Predicate-aware → post-filter crossover** at ~25% selectivity. Above 25%, post-filter matches predicate-aware recall at lower latency because most candidates are matching anyway.

### 14.3 The Production Implication

A production vector database should not use a single filtering strategy. The optimal approach is:

1. **Check selectivity** (or estimate it from metadata statistics).
2. If selectivity < ~2%: use pre-filter (brute force over the matching subset).
3. If selectivity is 2%–20%: use predicate-aware traversal.
4. If selectivity > 20%: use post-filter (simpler, nearly as good as predicate-aware).

This is essentially what Weaviate does with its `flatSearchCutoff` parameter (pre-filter below the cutoff, graph search above). ACORN-aware systems add the predicate-aware middle regime.

### 14.4 Why Post-Filter Fails: A Deeper Explanation

The post-filter recall collapse at low selectivity is worth understanding in depth because it reveals a fundamental property of proximity graphs.

When the beam search runs at layer 0, it explores nodes in order of distance from the query. The search converges in the *Voronoi region* of the query — the local neighborhood where $q$'s nearest vectors live. At 0.1% selectivity, 99.9% of nodes in this region are non-matching. The beam width $ef$ controls how many nodes the search keeps, but it cannot control *where those nodes are*. Even at $ef = 500$ (oversample factor 50), the search explores at most 500 nodes, all concentrated in $q$'s Voronoi region. The nearest matching vector might be the 5,000th nearest vector overall — well outside the search radius.

Increasing the oversample factor helps only linearly: doubling osf roughly doubles the number of matching candidates found. To reach recall 0.9 at 0.1% selectivity, you would need osf ≈ 900, meaning $ef = 9{,}000$. At that point, you are doing nearly as much work as brute force but with graph traversal overhead on top.

This is why predicate-aware traversal is necessary: by keeping non-matching nodes in the candidate frontier (but not the result set), the search can "escape" $q$'s immediate Voronoi region and navigate to distant parts of the graph where matching vectors live. The non-matching nodes serve as navigational bridges.

---

## 15. FAISS Comparison and What It Tells Us

### 15.1 Recall Match

Our post-filter and FAISS's post-filter produce nearly identical recall at every configuration:

| Selectivity | Our osf=50 | FAISS osf=50 | Δ       |
|-------------|-----------|--------------|---------|
| 0.1%        | 0.0510    | 0.0514       | -0.0004 |
| 1%          | 0.4890    | 0.4914       | -0.0024 |
| 5%          | 0.9926    | 0.9922       | +0.0004 |
| 10%         | 0.9972    | 0.9972       | 0.0000  |
| 25%         | 0.9993    | 0.9984       | +0.0009 |
| 50%         | 0.9990    | 0.9991       | -0.0001 |

### 15.2 What This Confirms

The recall match within ±0.003 confirms two things:

1. **Our HNSW implementation is correct.** The graph structure produces the same search quality as FAISS's battle-tested C++ implementation.

2. **The recall collapse is inherent to post-filtering, not our implementation.** FAISS's post-filter also achieves only 0.05 recall at 0.1% selectivity. The problem is the strategy itself, not a bug.

### 15.3 Latency Difference

FAISS is ~3–4× faster in query latency (C++ vs Python). This is expected and irrelevant for the filtering study. The relative comparisons between strategies within our implementation are valid. If we reimplemented in C++, all strategies would speed up proportionally.

---

## 16. Limitations and Future Work

### 16.1 No Construction-Time Modification

Our HNSW graph is built without predicate awareness. ACORN-$\gamma$ expands neighbor lists during construction by factor $\gamma = 1/s_{min}$ to ensure the predicate subgraph has sufficient connectivity. This would dramatically reduce predicate-aware latency at extreme selectivities (from 112ms to perhaps 5–10ms at 0.1% selectivity).

**Future work:** Implement ACORN-$\gamma$ construction and measure the latency improvement vs memory overhead tradeoff.

### 16.2 No 2-Hop Neighbor Expansion

ACORN-1 examines neighbors-of-neighbors during search to compensate for unexpanded neighbor lists. This would reduce latency at low selectivity without modifying construction — a cheaper alternative to ACORN-$\gamma$.

**Future work:** Add 2-hop expansion as an option and benchmark its impact on latency and recall.

### 16.3 Single Equality Predicate

Production systems support boolean combinations (`AND`, `OR`), range predicates (`price < 100`), and set membership (`category IN ('A', 'B', 'C')`). Our implementation handles only `category == c`.

**Future work:** Extend to multi-attribute filters and range predicates. The search-time mechanism generalizes naturally (the predicate check is a callback), but the selectivity estimation becomes more complex.

### 16.4 No Predicate Clustering

Our synthetic metadata is randomly assigned. In real-world data, metadata often correlates with vector proximity:

- Documents about "finance" may cluster in embedding space.
- Products in "electronics" may cluster near each other.
- Images of "cats" form a distinct region.

Predicate clustering can either help (matching vectors are nearby, fewer hops needed) or hurt (matching vectors are in a distant cluster, more stepping stones needed). Our results represent the "no clustering" baseline.

**Future work:** Generate synthetic metadata with controllable spatial clustering and measure the impact on all three strategies.

### 16.5 Python Speed Gap

Our implementation is 3–10× slower than FAISS in absolute latency. All latency numbers should be interpreted as relative comparisons between strategies, not as production performance benchmarks. The recall results, which depend on graph structure rather than language, are directly comparable.

### 16.6 Fixed Metadata Distribution

Our metadata distribution is fixed at build time: category 0 always has exactly 1,000 vectors, category 1 has 10,000, and so on. In production, metadata distributions change over time and queries may target arbitrary predicates with arbitrary selectivities.

---

## 17. Talking Points


### 17.1 "Walk me through HNSW."

HNSW builds a multi-layer proximity graph. Each vector is assigned a random layer from a geometric distribution — most vectors live only on layer 0, a few reach higher layers. Upper layers are sparse with long-range connections for coarse navigation; layer 0 is dense with short-range connections for precise convergence. Search starts at the top layer and greedily descends: at each upper layer, find the single closest node; at layer 0, run a beam search with width $ef$ to find the $k$ nearest. The neighbor selection uses an RNG-based heuristic that prunes redundant edges — it keeps a candidate only if no already-selected neighbor provides a shorter path, ensuring edges span diverse directions. Key parameters: $M$ controls degree (memory/recall tradeoff), $ef_{construction}$ controls graph quality, $ef_{search}$ controls query-time recall/latency.

### 17.2 "Why does post-filtering fail at low selectivity?"

The graph was built to navigate toward the globally nearest vectors, not the nearest matching vectors. At low selectivity, 99.9% of the globally nearest vectors don't match the predicate. The beam search converges in a neighborhood where matches are sparse. Even with a 50× oversample factor, the search stays within a local region that simply doesn't contain the target vectors. The graph's edges lead to the wrong place — they are optimized for unfiltered proximity, not predicate-filtered proximity.

### 17.3 "What's the ACORN insight?"

Don't filter the traversal frontier — filter the result set. During beam search, add ALL neighbors to the candidate heap (matching or not), but only add matching neighbors to the result heap. Non-matching nodes are stepping stones: the search walks through them to maintain graph connectivity, reaching matching nodes that might be far away in graph distance. This preserves the graph's navigational structure while enforcing the predicate on the output.

### 17.4 "Why not full ACORN?"

Full ACORN-$\gamma$ modifies construction to expand neighbor lists by factor $\gamma$, ensuring predicate subgraph connectivity. This requires choosing $\gamma$ upfront (committing to a minimum selectivity at build time), adds memory overhead, and makes construction slower. My implementation shows the search-time insight alone yields dramatic improvements (recall 0.9999 vs 0.05 at 0.1% selectivity). The tradeoff is latency at extreme selectivities: without construction-time expansion, the search must traverse many stepping stones. If I needed production-grade performance at 0.1% selectivity, the next step would be ACORN-$\gamma$ or ACORN-1's 2-hop expansion.

### 17.5 "When does pre-filter win?"

When selectivity is below ~2–5%. At 0.1% selectivity on 1M vectors, there are only 1,000 matching vectors. Brute force over 1,000 128-dim vectors takes 0.62ms — no graph traversal can beat that. This is why Weaviate implements a `flatSearchCutoff` parameter: below the cutoff, switch to brute force automatically.

### 17.6 "How would you build a production filtered vector search system?"

Three-tier strategy based on selectivity estimation:
- Below ~2%: pre-filter (brute force over the matching subset).
- 2%–20%: predicate-aware traversal (ACORN-style filtered beam search).
- Above 20%: post-filter (standard search + discard non-matching, simplest and fast enough).

Plus: maintain metadata statistics (category cardinalities, selectivity estimates), use them to route queries to the right strategy, and set a flat-search cutoff that adapts as the data distribution changes.

### 17.7 "What failure modes did you observe?"

Three key failure modes:
1. **Post-filter recall collapse:** At 0.1% selectivity, recall is 0.05 even with 50× oversampling. The graph cannot navigate to matching vectors because its edges are optimized for unfiltered proximity.
2. **Predicate-aware latency explosion:** At 0.1% selectivity, predicate-aware takes 112–1565ms per query. The search must explore tens of thousands of non-matching stepping stones. Without ACORN-$\gamma$'s construction-time neighbor expansion, the predicate subgraph has poor connectivity.
3. **Pre-filter latency scaling:** At 50% selectivity, pre-filter takes 77ms — brute force over 500,000 vectors is impractical even though recall is perfect.

### 17.8 "How did you validate correctness?"

Four levels:
1. **Unit tests:** 29 tests covering distance kernels (9), HNSW core (9), and filtering strategies (11).
2. **Unfiltered recall validation:** Our HNSW matches FAISS recall within ±0.005 across all ef values on SIFT1M. If recall were broken, the graph structure would be wrong.
3. **Pre-filter as recall ceiling:** Pre-filter recall is 1.0 by construction (brute force). It serves as ground truth and validates the ground truth computation itself.
4. **FAISS post-filter cross-check:** Our post-filter and FAISS's post-filter produce identical recall (within ±0.003), confirming the strategy implementation and that recall collapse is inherent to post-filtering.

---

## 18. References

1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836.

2. Patel, L., Kraft, P., Guestrin, C., & Zaharia, M. (2024). ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data. *Proceedings of the ACM on Management of Data (SIGMOD)*.

3. Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1), 117–128.

4. Fu, C., Xiang, C., Wang, C., & Cai, D. (2019). Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph. *Proceedings of the VLDB Endowment*, 12(5), 461–474.

5. Subramanya, S. J., Devvrit, Kadekodi, R., Krishaswamy, R., & Simhadri, H. V. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node. *Advances in Neural Information Processing Systems (NeurIPS)*.

---

*Repository: [github.com/bahetiaditi/HNSW](https://github.com/bahetiaditi/HNSW)*
