"""Custom spatial hash and disjoint-set structures used by fast DBSCAN."""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple


class LinkedListNode:
    __slots__ = ("gx", "gy", "point_index", "next")

    def __init__(self, gx: int, gy: int, point_index: int,
                 next_node: Optional["LinkedListNode"] = None) -> None:
        self.gx = gx
        self.gy = gy
        self.point_index = point_index
        self.next = next_node


class SinglyLinkedList:
    """Minimal custom singly linked list for hash bucket chains."""

    def __init__(self) -> None:
        self.head: Optional[LinkedListNode] = None

    def prepend(self, gx: int, gy: int, point_index: int) -> None:
        self.head = LinkedListNode(gx, gy, point_index, self.head)

    def __iter__(self) -> Iterator[LinkedListNode]:
        node = self.head
        while node is not None:
            yield node
            node = node.next


def _next_prime(value: int) -> int:
    candidate = max(2, value)
    while True:
        is_prime = True
        divisor = 2
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            return candidate
        candidate += 1


class SpatialHashTable:
    """Uniform epsilon grid stored in a prime-sized, chained hash table."""

    def __init__(self, eps: float, point_count: int) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if point_count < 0:
            raise ValueError("point_count must be non-negative")
        # M >= 2n ensures n/M <= 0.5 (for n=0 choose the smallest prime).
        self.bucket_count = _next_prime(max(2, 2 * point_count))
        self.buckets = [SinglyLinkedList() for _ in range(self.bucket_count)]
        self.eps = eps
        self.point_count = 0

    @staticmethod
    def hash_coordinates(gx: int, gy: int, bucket_count: int) -> int:
        mixed = (gx * 73856093) ^ (gy * 19349663)
        return abs(mixed) % bucket_count

    def cell_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        import math
        return math.floor(x / self.eps), math.floor(y / self.eps)

    def insert(self, point_index: int, x: float, y: float) -> None:
        gx, gy = self.cell_coordinates(x, y)
        bucket = self.hash_coordinates(gx, gy, self.bucket_count)
        self.buckets[bucket].prepend(gx, gy, point_index)
        self.point_count += 1

    def query_cells(self, gx: int, gy: int) -> Iterator[int]:
        """Yield indices in the 3x3 cells; hash collisions are filtered by key."""
        for target_x in range(gx - 1, gx + 2):
            for target_y in range(gy - 1, gy + 2):
                bucket = self.buckets[self.hash_coordinates(
                    target_x, target_y, self.bucket_count)]
                for node in bucket:
                    if node.gx == target_x and node.gy == target_y:
                        yield node.point_index


class DisjointSetUnion:
    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if item < 0 or item >= len(self.parent):
            raise IndexError("DSU index out of range")
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root
