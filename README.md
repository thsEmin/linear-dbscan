# Spatially Indexed DBSCAN: Reducing Neighborhood Search Cost

This dependency-free project compares a brute-force DBSCAN baseline with an exact queue-based DBSCAN that uses a custom spatial hash grid. It also retains a DSU-based spatial variant.

## Where the quadratic work occurs

The brute-force `region_query` scans all `n` points for each query. Up to `n` such queries gives `O(n^2)` distance comparisons. The custom grid limits candidate checks to nearby cells when local density stays bounded.

## Data structures and grid queries

`SpatialHashTable` uses a prime-sized bucket array, the Teschner hash, and custom singly linked chaining. Cell coordinates are `gx=floor(x/eps)` and `gy=floor(y/eps)`, including for negative coordinates. With cell side length `eps`, every true epsilon-neighbor is in the same cell or one of its eight adjacent cells. Each candidate is checked by exact squared Euclidean distance: `(x1-x2)^2 + (y1-y2)^2 <= eps^2`. Hash collisions are filtered against the full grid coordinates.

The queue implementation sorts returned neighbors by original input index to preserve the baseline's deterministic expansion and ambiguous-border assignment behavior. `min_samples` counts the point itself. Duplicates are distinct points and count toward density. A large epsilon can make nearly every point a neighbor and remove the grid's advantage.

## Pseudocode

```text
insert all points into the custom epsilon grid
for each input point p:
    if p is already visited: continue
    neighbors = exact radius query over p's 3x3 cells
    if neighbor count < min_samples: mark p noise; continue
    start a cluster; enqueue neighbors in input order
    while queue is not empty:
        q = dequeue
        if q was noise: assign it to this cluster
        if q was already visited: continue
        mark q visited
        q_neighbors = exact radius query over q's 3x3 cells
        if q is core: enqueue not-yet-queued q_neighbors
```

## Complexity

For fixed 2D dimension, spatial hashing plus queue expansion is expected near `O(n)` when expected hash-chain length and the number of local candidates/neighbors remain bounded. More generally, work is proportional to grid candidates examined and actual neighbors processed, plus the linear insertion and labeling work. In a dense worst case or with very large epsilon, a point can have `Theta(n)` neighbors, so exact DBSCAN can still take `Theta(n^2)` time. This implementation makes no unconditional linear-time claim.

## Run

```powershell
python -m unittest discover -s tests -v
python benchmarks/run_benchmarks.py
```

The benchmark compares brute force and custom-grid queue DBSCAN on the same generated sparse, clustered, and dense datasets. It asserts exact label equality for each run.

## Benchmark results

Python 3.14.6 on this machine. Timings vary by machine; comparison counts are deterministic for the fixed seeds. Sparse data expands its bounding square with `sqrt(N)`. Clustered data scales spread and center spacing with `sqrt(N)`. Dense data keeps all points in a compact cloud with `eps=1`.

| Data | N | Brute (s) | Grid + queue (s) | Speedup | Brute checks | Grid checks |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 500 | 0.037324 | 0.004444 | 8.40x | 250,000 | 534 |
| Sparse | 2,000 | 0.553938 | 0.017306 | 32.01x | 4,000,000 | 2,130 |
| Sparse | 5,000 | 3.889708 | 0.061094 | 63.67x | 25,000,000 | 5,268 |
| Clustered | 500 | 0.049209 | 0.015151 | 3.25x | 250,000 | 24,202 |
| Clustered | 2,000 | 0.709654 | 0.068863 | 10.31x | 4,000,000 | 120,754 |
| Clustered | 5,000 | 3.881876 | 0.156785 | 24.76x | 25,000,000 | 322,616 |
| Dense | 500 | 0.050087 | 0.104041 | 0.48x | 250,000 | 250,000 |
| Dense | 2,000 | 0.991390 | 1.786673 | 0.55x | 4,000,000 | 4,000,000 |
| Dense | 5,000 | 4.800607 | 9.100666 | 0.53x | 25,000,000 | 25,000,000 |

## Integrated website

Run `python serve.py` from this folder, then open http://localhost:8000.
The website in `web/` calls the local server, which directly imports the algorithms
from `src/dbscan_project`. No duplicated Python package or browser runtime is used.
Keep the terminal running. Ctrl+C stops the website. Use `--port 8001` to change ports.
Timings are medians of three runs after warm-up, with alternating algorithm order.
The older sibling `site/` folder is a hosting prototype, not used by this website.
