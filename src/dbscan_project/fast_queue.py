"""Queue-based DBSCAN using the project's custom spatial hash grid."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .brute_force import DBSCANStats
from .fast import Point
from .structures import SpatialHashTable

UNVISITED = -2
NOISE = -1


def dbscan_spatial_queue(points: Sequence[Point], eps: float, min_samples: int):
    """Exact DBSCAN with indexed epsilon queries and iterative queue expansion.

    Sorted neighbor indices preserve the brute-force implementation's input
    order and therefore its deterministic ambiguous-border assignment.
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
    distance_checks = 0
    region_queries = 0

    def region_query(i: int) -> List[int]:
        nonlocal distance_checks, region_queries
        region_queries += 1
        x, y = normalized[i]
        gx, gy = grid.cell_coordinates(x, y)
        result = []
        for j in grid.query_cells(gx, gy):
            distance_checks += 1
            dx = x - normalized[j][0]
            dy = y - normalized[j][1]
            if dx * dx + dy * dy <= eps_squared:
                result.append(j)
        result.sort()
        return result

    labels = [UNVISITED] * n
    enqueued_stamp = [-1] * n
    cluster_id = 0
    for point_index in range(n):
        if labels[point_index] != UNVISITED:
            continue
        neighbors = region_query(point_index)
        if len(neighbors) < min_samples:
            labels[point_index] = NOISE
            continue

        labels[point_index] = cluster_id
        queue = list(neighbors)

        for neighbor in queue:
            enqueued_stamp[neighbor] = cluster_id
        head = 0
        while head < len(queue):
            candidate = queue[head]
            head += 1
            if labels[candidate] == NOISE:
                labels[candidate] = cluster_id
            if labels[candidate] != UNVISITED:
                continue
            labels[candidate] = cluster_id
            candidate_neighbors = region_query(candidate)
            if len(candidate_neighbors) >= min_samples:
                for neighbor in candidate_neighbors:
                    if enqueued_stamp[neighbor] != cluster_id:
                        enqueued_stamp[neighbor] = cluster_id
                        queue.append(neighbor)
        cluster_id += 1

    return labels, DBSCANStats(distance_checks, region_queries)

