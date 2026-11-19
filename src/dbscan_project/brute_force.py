"""Readable O(n^2) DBSCAN reference implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
UNVISITED = -2
NOISE = -1


@dataclass(frozen=True)
class DBSCANStats:
    distance_checks: int
    region_queries: int


def dbscan_brute_force(
    points: Sequence[Point], eps: float, min_samples: int
) -> Tuple[List[int], DBSCANStats]:
    """Cluster 2D points with classic scan-based DBSCAN.

    `min_samples` counts the query point itself. Expansion is iterative and
    deterministic: input order determines cluster IDs and border assignment.
    The method performs O(n^2) distance checks in the general case.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    normalized: List[Point] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {index} must have exactly 2 coordinates")
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {index} coordinates must be finite")
        normalized.append((float(x), float(y)))

    n = len(normalized)
    eps_squared = eps * eps
    labels = [UNVISITED] * n
    distance_checks = 0
    region_queries = 0

    def region_query(point_index: int) -> List[int]:
        nonlocal distance_checks, region_queries
        region_queries += 1
        px, py = normalized[point_index]
        neighbors: List[int] = []
        for candidate_index, (qx, qy) in enumerate(normalized):
            distance_checks += 1
            dx = px - qx
            dy = py - qy
            if dx * dx + dy * dy <= eps_squared:
                neighbors.append(candidate_index)
        return neighbors

    cluster_id = 0
    for point_index in range(n):
        if labels[point_index] != UNVISITED:
            continue

        neighbors = region_query(point_index)
        if len(neighbors) < min_samples:
            labels[point_index] = NOISE
            continue

        labels[point_index] = cluster_id
        # This queue-like list is the standard iterative DBSCAN expansion.
        # A flag array avoids repeatedly inserting the same point.
        seeds = list(neighbors)
        queued = [False] * n
        for neighbor_index in seeds:
            queued[neighbor_index] = True

        cursor = 0
        while cursor < len(seeds):
            candidate_index = seeds[cursor]
            cursor += 1

            if labels[candidate_index] == NOISE:
                # Previously discovered noise can be a border point.
                labels[candidate_index] = cluster_id
            if labels[candidate_index] != UNVISITED:
                continue

            labels[candidate_index] = cluster_id
            candidate_neighbors = region_query(candidate_index)
            if len(candidate_neighbors) >= min_samples:
                for neighbor_index in candidate_neighbors:
                    if not queued[neighbor_index]:
                        queued[neighbor_index] = True
                        seeds.append(neighbor_index)

        cluster_id += 1

    return labels, DBSCANStats(distance_checks, region_queries)
