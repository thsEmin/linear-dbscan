"""Compare brute-force and custom-grid queue DBSCAN on three distributions."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dbscan_project.brute_force import dbscan_brute_force
from src.dbscan_project.data import make_dataset
from src.dbscan_project.fast_queue import dbscan_spatial_queue


def dataset(kind, n, seed):
    rng = random.Random(seed)
    if kind == "sparse":
        # Fixed epsilon, expanding square, low expected local occupancy.
        width = max(30.0, 3.0 * n ** 0.5)
        return ([(rng.uniform(-width, width), rng.uniform(-width, width))
                 for _ in range(n)], 0.5, 5)
    if kind == "clustered":
        points, _ = make_dataset(n, seed=seed, cluster_count=4,
                                 noise_fraction=0.05,
                                 spread=0.5 * (n / 500) ** 0.5,
                                 separation=max(8.0, n ** 0.5 * 1.5))
        return points, 0.45, 5
    # One compact cloud with a large radius: quadratic dense-neighborhood case.
    points = [(rng.gauss(0.0, 0.1), rng.gauss(0.0, 0.1)) for _ in range(n)]
    return points, 1.0, 5


def run(sizes, seed=2026):
    print("case,N,brute_seconds,grid_queue_seconds,speedup,"
          "brute_distance_checks,grid_distance_checks")
    rows = []
    for kind in ("sparse", "clustered", "dense"):
        for n in sizes:
            points, eps, min_samples = dataset(kind, n, seed + n)
            start = time.perf_counter()
            expected, brute_stats = dbscan_brute_force(points, eps, min_samples)
            brute_time = time.perf_counter() - start
            start = time.perf_counter()
            actual, grid_stats = dbscan_spatial_queue(points, eps, min_samples)
            grid_time = time.perf_counter() - start
            assert expected == actual, f"label mismatch: {kind}, N={n}"
            speedup = brute_time / grid_time if grid_time else float("inf")
            row = (kind, n, brute_time, grid_time, speedup,
                   brute_stats.distance_checks, grid_stats.distance_checks)
            rows.append(row)
            print(f"{kind},{n},{brute_time:.6f},{grid_time:.6f},{speedup:.2f}x,"
                  f"{brute_stats.distance_checks},{grid_stats.distance_checks}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000, 5000])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args.sizes, args.seed)
