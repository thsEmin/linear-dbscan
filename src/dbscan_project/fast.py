"""Spatially hashed, exact 2D DBSCAN using custom grid and DSU structures."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .brute_force import DBSCANStats
from .structures import DisjointSetUnion, SpatialHashTable

Point = Tuple[float, float]
NOISE = -1


def fast_dbscan(points: Sequence[Point], eps: float, min_samples: int):
    """Run exact 2D DBSCAN using a custom spatial grid and core-only DSU.

    Cluster and ambiguous border ownership follows the baseline's input-order
    first-discovery policy. `distance_checks` counts exact candidate distance
    evaluations (grid candidates for this function).

    Expected time is O(n) for fixed dimension when expected hash-chain length
    and local epsilon-neighborhood size are bounded. Worst-case time is O(n^2)
    because a point can have Theta(n) neighbors and the exact neighbor graph
    can contain Theta(n^2) edges. This is exact DBSCAN; it makes no density or
    approximation shortcut.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    normalized: List[Point] = []
    for i, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {i} must have exactly 2 coordinates")
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {i} coordinates must be finite")
        normalized.append((float(x), float(y)))

    n = len(normalized)
    grid = SpatialHashTable(eps, n)
    for i, (x, y) in enumerate(normalized):
        grid.insert(i, x, y)

    eps_squared = eps * eps
    neighbors: List[List[int]] = [[] for _ in range(n)]
    checks = 0
    for i, (x, y) in enumerate(normalized):
        gx, gy = grid.cell_coordinates(x, y)
        for j in grid.query_cells(gx, gy):
            checks += 1
            dx = x - normalized[j][0]
            dy = y - normalized[j][1]
            if dx * dx + dy * dy <= eps_squared:
                neighbors[i].append(j)

    core = [len(neighbors[i]) >= min_samples for i in range(n)]
    dsu = DisjointSetUnion(n)
    # Only process each core-core edge once. The sorted traversal and canonical
    # component key below preserve the baseline's earliest core-point ordering.
    for i in range(n):
        if core[i]:
            for j in neighbors[i]:
                if j > i and core[j]:
                    dsu.union(i, j)

    component_min = {}
    for i in range(n):
        if core[i]:
            root = dsu.find(i)
            if root not in component_min or i < component_min[root]:
                component_min[root] = i
    ordered_components = sorted(component_min.items(), key=lambda item: item[1])
    root_labels = {root: label for label, (root, _) in enumerate(ordered_components)}
    labels = [NOISE] * n
    for i in range(n):
        if core[i]:
            labels[i] = root_labels[dsu.find(i)]

    # A border adjacent to multiple core components belongs to the component
    # whose first core seed is earliest, matching baseline expansion order.
    for i in range(n):
        if core[i]:
            continue
        chosen_label = -1
        for j in neighbors[i]:
            if core[j]:
                candidate_label = root_labels[dsu.find(j)]
                if chosen_label < 0 or candidate_label < chosen_label:
                    chosen_label = candidate_label
        if chosen_label >= 0:
            labels[i] = chosen_label

    return labels, DBSCANStats(checks, n)
