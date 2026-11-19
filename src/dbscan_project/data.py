"""Deterministic synthetic 2D datasets for DBSCAN experiments."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def make_dataset(
    n: int,
    *,
    seed: int = 0,
    cluster_count: int = 4,
    noise_fraction: float = 0.08,
    spread: float = 0.55,
    separation: float = 8.0,
) -> Tuple[List[Point], Optional[List[int]]]:
    """Return reproducible clustered points and their source labels.

    Source labels are for evaluating the generator only; DBSCAN labels need not
    match them because density connectivity, not generator membership, defines
    the algorithm's result. Noise points have source label ``-1``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between 0 and 1")
    if spread < 0 or separation <= 0:
        raise ValueError("spread must be non-negative and separation positive")

    rng = random.Random(seed)
    noise_n = int(n * noise_fraction)
    clustered_n = n - noise_n
    counts = [clustered_n // cluster_count] * cluster_count
    for i in range(clustered_n % cluster_count):
        counts[i] += 1

    centers = [
        (separation * (i % 2), separation * (i // 2))
        for i in range(cluster_count)
    ]
    points: List[Point] = []
    labels: List[int] = []
    for cluster_id, count in enumerate(counts):
        cx, cy = centers[cluster_id]
        for _ in range(count):
            points.append((rng.gauss(cx, spread), rng.gauss(cy, spread)))
            labels.append(cluster_id)

    # Keep noise in a wider bounding region so it is usually isolated, while
    # retaining a small chance of overlap that makes the generator realistic.
    extent = separation * (max(2, cluster_count) + 1)
    for _ in range(noise_n):
        points.append((rng.uniform(-extent, extent), rng.uniform(-extent, extent)))
        labels.append(-1)

    order = list(range(n))
    rng.shuffle(order)
    return [points[i] for i in order], [labels[i] for i in order]
