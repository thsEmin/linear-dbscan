import random
import unittest

from src.dbscan_project.brute_force import dbscan_brute_force
from src.dbscan_project.fast import fast_dbscan
from src.dbscan_project.fast_queue import dbscan_spatial_queue
from src.dbscan_project.structures import DisjointSetUnion, SpatialHashTable


def same_partition(left, right):
    if len(left) != len(right):
        return False
    for i in range(len(left)):
        if (left[i] == -1) != (right[i] == -1):
            return False
        for j in range(len(left)):
            if (left[i] == left[j]) != (right[i] == right[j]):
                return False
    return True


class SpatialHashTests(unittest.TestCase):
    def test_collisions_keep_coordinates_separate(self):
        grid = SpatialHashTable(1.0, 3)
        first = (0, 0)
        second = (1, 1)
        while grid.hash_coordinates(*first, grid.bucket_count) != grid.hash_coordinates(
            *second, grid.bucket_count
        ):
            second = (second[0] + 1, second[1])
        grid.insert(10, 0.1, 0.1)
        grid.insert(20, second[0] + 0.1, second[1] + 0.1)
        self.assertEqual(list(grid.query_cells(*first)), [10])
        self.assertLessEqual(3 / grid.bucket_count, 0.5)

    def test_three_by_three_boundary_lookup(self):
        grid = SpatialHashTable(1.0, 9)
        cells = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                 (0, 1), (1, -1), (1, 0), (1, 1), (2, 0)]
        for i, (gx, gy) in enumerate(cells):
            grid.insert(i, gx + 0.2, gy + 0.2)
        self.assertEqual(set(grid.query_cells(0, 0)), set(range(9)))


class DSUTests(unittest.TestCase):
    def test_union_by_rank_and_path_compression(self):
        dsu = DisjointSetUnion(8)
        dsu.union(0, 1)
        dsu.union(2, 3)
        dsu.union(0, 2)
        dsu.union(4, 5)
        dsu.union(6, 7)
        dsu.union(4, 6)
        dsu.union(0, 4)
        root = dsu.find(7)
        self.assertEqual(dsu.find(5), root)
        self.assertEqual(dsu.parent[7], root)
        self.assertEqual(dsu.rank[root], 3)


class EquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, points, eps, min_samples):
        expected, _ = dbscan_brute_force(points, eps, min_samples)
        actual, _ = fast_dbscan(points, eps, min_samples)
        self.assertTrue(same_partition(expected, actual), (expected, actual))

    def test_many_seeds(self):
        for seed in range(20):
            rng = random.Random(seed)
            points = []
            for center_x, center_y in ((0.0, 0.0), (5.0, 0.0), (0.0, 5.0)):
                for _ in range(18):
                    points.append((rng.gauss(center_x, 0.35), rng.gauss(center_y, 0.35)))
            points.extend((rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(8))
            rng.shuffle(points)
            self.assert_equivalent(points, 0.8, 5)

    def test_empty_duplicates_and_noise(self):
        self.assert_equivalent([], 1.0, 2)
        self.assert_equivalent([(0.0, 0.0)] * 12, 0.01, 5)
        self.assert_equivalent([(float(i * 4), 0.0) for i in range(20)], 1.0, 2)

    def test_border_point_cannot_bridge_core_components(self):
        # Two dense core islands are 2*eps apart, with a shared non-core point
        # within eps of each island. The shared point gets one deterministic
        # border owner and never merges the island components.
        eps = 1.0
        points = [(-1.0, 0.0), (-1.0, 0.05), (-1.0, -0.05),
                  (1.0, 0.0), (1.0, 0.05), (1.0, -0.05), (0.0, 0.0)]
        self.assert_equivalent(points, eps, 4)

    def test_dense_single_cell_fallback_correctness(self):
        points = [(0.01 * (i % 5), 0.01 * (i // 5)) for i in range(60)]
        self.assert_equivalent(points, 1.0, 4)

    def test_exact_labels_match_baseline_policy(self):
        for seed in range(10):
            rng = random.Random(seed)
            points = [(rng.gauss(0, 0.4), rng.gauss(0, 0.4)) for _ in range(25)]
            points += [(5 + rng.gauss(0, 0.4), rng.gauss(0, 0.4)) for _ in range(25)]
            rng.shuffle(points)
            self.assertEqual(dbscan_brute_force(points, 0.9, 5)[0],
                             fast_dbscan(points, 0.9, 5)[0])

    def test_queue_grid_exact_labels_match_baseline(self):
        cases = [([], 1.0, 2), ([(0.0, 0.0)] * 12, 0.01, 5),
                 ([(float(i * 4), -float(i % 3)) for i in range(30)], 1.0, 3)]
        for seed in range(12):
            rng = random.Random(seed)
            points = ([(rng.gauss(0, 0.4), rng.gauss(0, 0.4)) for _ in range(20)] +
                      [(5 + rng.gauss(0, 0.4), rng.gauss(0, 0.4)) for _ in range(20)])
            points.extend([(2.5, 0.0)])
            rng.shuffle(points)
            cases.append((points, 0.9, 5))
        for points, eps, min_samples in cases:
            baseline = dbscan_brute_force(points, eps, min_samples)[0]
            indexed = dbscan_spatial_queue(points, eps, min_samples)[0]
            self.assertEqual(baseline, indexed)


if __name__ == "__main__":
    unittest.main()
