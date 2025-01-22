import numpy as np
import sympy as sp
from scipy.linalg import svd
from numpy.linalg import eig

class TensorAlgebraEngine:
    def __init__(self):
        pass

    # Tensor Addition
    def add_tensors(self, tensor_a, tensor_b):
        """Add two tensors."""
        return np.add(tensor_a, tensor_b)

    # Tensor Contraction
    def contract_tensors(self, tensor, axes):
        """
        Perform contraction over specified axes.
        E.g., Contracting a 3D tensor to 2D along axes (1, 2).
        """
        return np.tensordot(tensor, tensor, axes=axes)

    # Tensor Dot Product
    def tensor_dot(self, tensor_a, tensor_b):
        """Compute the dot product of two tensors."""
        return np.tensordot(tensor_a, tensor_b, axes=1)

    # Symbolic Tensor Gradient
    def symbolic_gradient(self, tensor_expr, variables):
        """Compute symbolic gradients of a tensor expression."""
        gradients = [sp.diff(tensor_expr, var) for var in variables]
        return gradients

    # Tensor Eigendecomposition
    def tensor_eigen(self, tensor):
        """Compute the eigenvalues and eigenvectors of a 2D tensor."""
        if tensor.ndim != 2:
            raise ValueError("Eigendecomposition requires a 2D tensor.")
        return eig(tensor)

    # Tensor Singular Value Decomposition (SVD)
    def tensor_svd(self, tensor):
        """Perform singular value decomposition on a 2D tensor."""
        if tensor.ndim != 2:
            raise ValueError("SVD requires a 2D tensor.")
        U, S, V = svd(tensor)
        return U, S, V

    # Christoffel Symbols
    def christoffel_symbols(self, metric_tensor, variables):
        """Compute Christoffel symbols of the first kind for a metric tensor."""
        dim = len(variables)
        christoffel = [[[None for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    term = sp.diff(metric_tensor[i, j], variables[k])
                    term += sp.diff(metric_tensor[i, k], variables[j])
                    term -= sp.diff(metric_tensor[j, k], variables[i])
                    christoffel[i][j][k] = sp.simplify(term / 2)
        return christoffel

    # Visualization (Tensor Heatmap)
    def visualize_tensor(self, tensor):
        """Visualize a 2D tensor as a heatmap."""
        if tensor.ndim != 2:
            raise ValueError("Visualization supports 2D tensors only.")
        import matplotlib.pyplot as plt
        plt.imshow(tensor, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("Tensor Heatmap")
        plt.show()


# Example Usage
if __name__ == "__main__":
    engine = TensorAlgebraEngine()

    # Example 1: Tensor Addition
    tensor_a = np.array([[1, 2], [3, 4]])
    tensor_b = np.array([[5, 6], [7, 8]])
    result_add = engine.add_tensors(tensor_a, tensor_b)
    print("Tensor Addition Result:\n", result_add)

    # Example 2: Tensor Contraction
    tensor = np.random.rand(3, 3, 3)
    contracted = engine.contract_tensors(tensor, axes=((1, 2), (1, 2)))
    print("Tensor Contraction Result:\n", contracted)

    # Example 3: Symbolic Gradient
    x, y = sp.symbols("x y")
    tensor_expr = x**2 + y**2
    gradients = engine.symbolic_gradient(tensor_expr, [x, y])
    print("Symbolic Gradients:", gradients)

    # Example 4: Eigendecomposition
    eigenvalues, eigenvectors = engine.tensor_eigen(tensor_a)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Example 5: Singular Value Decomposition
    U, S, V = engine.tensor_svd(tensor_a)
    print("SVD - U:\n", U)
    print("SVD - S:\n", S)
    print("SVD - V:\n", V)

    # Example 6: Christoffel Symbols
    metric = sp.Matrix([[1, 0], [0, sp.sin(x)**2]])
    christoffel = engine.christoffel_symbols(metric, [x, y])
    print("Christoffel Symbols (sample):", christoffel[0][0][0])

    # Example 7: Visualization
    engine.visualize_tensor(tensor_a)
