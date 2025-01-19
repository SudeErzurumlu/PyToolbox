import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class GeometryEngine:
    def __init__(self):
        pass

    # Convex Hull Algorithm (Graham's Scan)
    def convex_hull(self, points):
        """Compute the convex hull of a set of 2D points."""
        points = sorted(points, key=lambda x: (x[0], x[1]))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))

        return lower[:-1] + upper[:-1]

    # Closest Pair of Points Algorithm
    def closest_pair(self, points):
        """Find the closest pair of points using divide-and-conquer."""
        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def closest_pair_recursive(points_sorted_x, points_sorted_y):
            if len(points_sorted_x) <= 3:
                return min(
                    ((p1, p2) for i, p1 in enumerate(points_sorted_x) for p2 in points_sorted_x[i + 1:]),
                    key=lambda pair: dist(pair[0], pair[1]),
                )

            mid = len(points_sorted_x) // 2
            left_x = points_sorted_x[:mid]
            right_x = points_sorted_x[mid:]
            mid_x = points_sorted_x[mid][0]

            left_y = list(filter(lambda p: p[0] <= mid_x, points_sorted_y))
            right_y = list(filter(lambda p: p[0] > mid_x, points_sorted_y))

            (p1_left, p2_left) = closest_pair_recursive(left_x, left_y)
            (p1_right, p2_right) = closest_pair_recursive(right_x, right_y)

            d = min(dist(p1_left, p2_left), dist(p1_right, p2_right))
            closest = (p1_left, p2_left) if dist(p1_left, p2_left) < dist(p1_right, p2_right) else (p1_right, p2_right)

            strip = [p for p in points_sorted_y if abs(p[0] - mid_x) < d]
            for i, p1 in enumerate(strip):
                for p2 in strip[i + 1 : i + 8]:
                    if dist(p1, p2) < d:
                        d = dist(p1, p2)
                        closest = (p1, p2)

            return closest

        points_sorted_x = sorted(points, key=lambda x: x[0])
        points_sorted_y = sorted(points, key=lambda x: x[1])

        return closest_pair_recursive(points_sorted_x, points_sorted_y)

    # Delaunay Triangulation
    def delaunay_triangulation(self, points):
        """Compute the Delaunay triangulation of a set of 2D points."""
        delaunay = Delaunay(points)
        return delaunay.simplices

    # Visualization
    def plot_geometry(self, points, hull=None, delaunay_simplices=None):
        """Plot points and optional geometric structures (hull, Delaunay)."""
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], label="Points", color="blue")

        if hull:
            hull = np.array(hull)
            plt.plot(
                np.append(hull[:, 0], hull[0, 0]),
                np.append(hull[:, 1], hull[0, 1]),
                label="Convex Hull",
                color="green",
            )

        if delaunay_simplices is not None:
            for simplex in delaunay_simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], "k-")

        plt.legend()
        plt.show()


# Example Usage
if __name__ == "__main__":
    engine = GeometryEngine()

    # Generate random points
    points = np.random.rand(30, 2) * 100

    # Convex Hull
    hull = engine.convex_hull(points)
    print("Convex Hull:", hull)

    # Closest Pair of Points
    closest = engine.closest_pair(points)
    print("Closest Pair:", closest)

    # Delaunay Triangulation
    delaunay = engine.delaunay_triangulation(points)
    print("Delaunay Triangles:", delaunay)

    # Plot results
    engine.plot_geometry(points, hull=hull, delaunay_simplices=delaunay)
