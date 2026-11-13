# Spatial-Hashed DBSCAN: From $O(n^2)$ to $O(n)$ From Scratch

A zero-dependency implementation of **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** engineered from scratch to reduce clustering time complexity from **$O(n^2)$** to **$O(n)$ expected time** in low-dimensional space—with **100% exact label equivalence** to standard brute-force DBSCAN.

### How It Works
Standard textbook implementations of DBSCAN rely on nested loops over arrays to compute pairwise distances and recursive BFS/DFS queues to expand clusters, creating an $O(n^2)$ bottleneck. This project replaces brute-force scanning with two custom-built data structures:

1. **Custom Spatial Hash Table (Separate Chaining):** Partitions 2D space into an $\epsilon \times \epsilon$ uniform grid using prime-number polynomial hashing and singly linked lists. Neighbor queries inspect at most **9 grid cells** ($3 \times 3$ neighborhood) instead of all $n$ points, achieving $O(1)$ expected neighborhood discovery.
2. **Disjoint Set Union (Union-Find):** Replaces recursive graph traversal with **Path Compression** and **Union by Rank** to merge Core Points in $O(\alpha(n)) \approx O(1)$ amortized time while isolating Border Points to prevent false cluster bridging.

### Key Highlights
* **Zero External ML/DS Libraries:** Spatial hash buckets, linked lists, Union-Find trees, and Euclidean distance metrics are implemented entirely from scratch.
* **Verified Correctness:** Automated test suite proving 100% identical cluster assignments and noise detection between the $O(n^2)$ baseline and the $O(n)$ engine.
* **Empirical Benchmarks:** Runtime and distance-calculation comparisons scaling from $N = 500$ to $N = 50,000+$ points, alongside formal best/average/worst-case and $O(3^d \cdot n)$ dimensional complexity analysis.
