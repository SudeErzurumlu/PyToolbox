import heapq
import math
import sys

# Priority Queue (Min-Heap)
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        heapq.heappush(self.heap, val)

    def extract_min(self):
        return heapq.heappop(self.heap)

    def get_min(self):
        return self.heap[0] if self.heap else None

    def is_empty(self):
        return len(self.heap) == 0

    def __str__(self):
        return str(self.heap)


# Graph Algorithms

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj = {i: [] for i in range(vertices)}

    def add_edge(self, u, v, weight):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

    def dijkstra(self, start):
        dist = [math.inf] * self.V
        dist[start] = 0
        pq = MinHeap()
        pq.insert((0, start))
        while not pq.is_empty():
            d, u = pq.extract_min()
            for v, weight in self.adj[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    pq.insert((dist[v], v))
        return dist


# Sorting Algorithms

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result


# Dynamic Programming Algorithms

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][W]


# LRU Cache (Least Recently Used)

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if len(self.cache) >= self.capacity:
            lru = self.order.pop(0)
            del self.cache[lru]
        self.cache[key] = value
        self.order.append(key)


# Test Example Usage

if __name__ == "__main__":
    # Graph Example
    g = Graph(5)
    g.add_edge(0, 1, 2)
    g.add_edge(0, 4, 3)
    g.add_edge(1, 2, 4)
    g.add_edge(1, 3, 5)
    g.add_edge(3, 4, 1)

    print("Dijkstra's Algorithm: Shortest path from vertex 0:")
    print(g.dijkstra(0))

    # Sorting Example
    arr = [5, 2, 9, 1, 5, 6]
    print("\nQuick Sort:", quick_sort(arr))
    print("Merge Sort:", merge_sort(arr))

    # Dynamic Programming Example
    X = "AGGTAB"
    Y = "GXTXAYB"
    print("\nLongest Common Subsequence (LCS) length:", lcs(X, Y))

    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    W = 5
    print("\nKnapsack Problem solution:", knapsack(weights, values, W))

    # LRU Cache Example
    cache = LRUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    print("\nLRU Cache:")
    print(cache.get(1)) # Returns 1
    cache.put(4, 4)  # Evicts key 2
    print(cache.get(2)) # Returns -1 (not found)
    cache.put(5, 5)  # Evicts key 3
    print(cache.get(3)) # Returns -1 (not found)
    print(cache.get(4)) # Returns 4
    print(cache.get(5)) # Returns 5
