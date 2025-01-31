import numpy as np
import scipy.spatial
import gudhi
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph

class ComputationalGeometryToolkit:
    def __init__(self):
        pass

    # Compute Delaunay triangulation
    def delaunay_triangulation(self, points):
        """Computes Delaunay triangulation of a given set of points."""
        return scipy.spatial.Delaunay(points)

    # Compute Voronoi diagram
    def voronoi_diagram(self, points):
        """Computes Voronoi diagram for a given set of points."""
        return scipy.spatial.Voronoi(points)

    # Compute Convex Hull
    def convex_hull(self, points):
        """Computes Convex Hull of a given set of points."""
        return scipy.spatial.ConvexHull(points)

    # Compute Simplicial Complex for Topological Data Analysis
    def compute_simplicial_complex(self, points):
        """Constructs a Vietoris-Rips simplicial complex from points."""
        rips_complex = gudhi.RipsComplex(points=points, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        return simplex_tree

    # Compute Betti numbers from a simplicial complex
    def compute_betti_numbers(self, simplex_tree):
        """Computes Betti numbers for a given simplicial complex."""
        return simplex_tree.betti_numbers()

    # Compute Laplacian Eigenvalues (used in Spectral Graph Theory & FEM)
    def compute_laplacian_eigenvalues(self, adjacency_matrix, k=6):
        """Computes the first k eigenvalues of the Laplacian matrix."""
        laplacian = csgraph.laplacian(adjacency_matrix, normed=True)
        eigenvalues, _ = eigs(laplacian, k=k, which="SM")
        return np.real(eigenvalues)

# Example Usage
if __name__ == "__main__":
    toolkit = ComputationalGeometryToolkit()

    # Example 1: Compute Delaunay Triangulation
    points = np.random.rand(10, 2)
    delaunay = toolkit.delaunay_triangulation(points)
    print("Delaunay Triangulation Simplices:\n", delaunay.simplices)

    # Example 2: Compute Voronoi Diagram
    voronoi = toolkit.voronoi_diagram(points)
    print("Voronoi Vertices:\n", voronoi.vertices)

    # Example 3: Compute Convex Hull
    hull = toolkit.convex_hull(points)
    print("Convex Hull Vertices:\n", hull.vertices)

    # Example 4: Compute Topological Features (Simplicial Complex)
    simplex_tree = toolkit.compute_simplicial_complex(points)
    betti_numbers = toolkit.compute_betti_numbers(simplex_tree)
    print("Computed Betti Numbers:", betti_numbers)

    # Example 5: Compute Laplacian Eigenvalues
    adjacency_matrix = np.random.rand(10, 10)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Symmetric matrix
    eigenvalues = toolkit.compute_laplacian_eigenvalues(adjacency_matrix)
    print("Laplacian Eigenvalues:\n", eigenvalues)
