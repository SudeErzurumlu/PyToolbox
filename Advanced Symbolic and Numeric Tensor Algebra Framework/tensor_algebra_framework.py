import numpy as np
from sympy import symbols, Matrix, simplify, diff
from scipy.sparse import coo_matrix
from scipy.linalg import svd
import matplotlib.pyplot as plt

class TensorAlgebraFramework:
    def __init__(self):
        pass

    # Symbolic tensor differentiation
    def symbolic_derivative(self, expression, variables):
        """Compute symbolic derivative of a tensor expression."""
        if not isinstance(variables, list):
            variables = [variables]
        return [diff(expression, var) for var in variables]

    # Efficient tensor contraction using Einstein summation
    def tensor_contraction(self, tensor_a, tensor_b, indices):
        """
        Perform tensor contraction using Einstein summation.
        Example: indices="ij,jk->ik"
        """
        return np.einsum(indices, tensor_a, tensor_b)

    # Sparse tensor generation
    def generate_sparse_tensor(self, shape, density=0.1):
        """Generate a random sparse tensor."""
        size = np.prod(shape)
        num_nonzero = int(size * density)
        indices = np.random.choice(range(size), num_nonzero, replace=False)
        data = np.random.rand(num_nonzero)
        sparse_tensor = coo_matrix((data, (indices // shape[1], indices % shape[1])), shape=shape)
        return sparse_tensor

    # Tensor decomposition (SVD)
    def tensor_svd(self, tensor):
        """Perform Singular Value Decomposition (SVD) on a tensor."""
        U, S, Vt = svd(tensor, full_matrices=False)
        return U, S, Vt

    # Tucker decomposition
    def tucker_decomposition(self, tensor, ranks):
        """Perform Tucker decomposition on a tensor."""
        core = tensor
        factors = []
        for mode, rank in enumerate(ranks):
            unfold = core.reshape(core.shape[mode], -1)
            U, _, _ = svd(unfold, full_matrices=False)
            factors.append(U[:, :rank])
            core = np.tensordot(core, U[:, :rank].T, axes=(mode, 0))
        return core, factors

    # Tensor visualization
    def visualize_tensor(self, tensor, title="Tensor Visualization"):
        """Visualize a tensor as a heatmap."""
        if tensor.ndim > 2:
            raise ValueError("Visualization supports only 2D tensors.")
        plt.imshow(tensor, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

# Example Usage
if __name__ == "__main__":
    framework = TensorAlgebraFramework()

    # Example 1: Symbolic tensor differentiation
    x, y = symbols("x y")
    expression = x**2 + y**2 + x * y
    derivatives = framework.symbolic_derivative(expression, [x, y])
    print("Symbolic Derivatives:", derivatives)

    # Example 2: Tensor contraction using Einstein summation
    tensor_a = np.random.rand(3, 4)
    tensor_b = np.random.rand(4, 5)
    result = framework.tensor_contraction(tensor_a, tensor_b, "ij,jk->ik")
    print("Tensor Contraction Result Shape:", result.shape)

    # Example 3: Sparse tensor generation
    sparse_tensor = framework.generate_sparse_tensor(shape=(100, 100), density=0.05)
    print("Sparse Tensor Shape:", sparse_tensor.shape)
    print("Non-zero Entries:", sparse_tensor.nnz)

    # Example 4: Tensor SVD
    tensor = np.random.rand(10, 10)
    U, S, Vt = framework.tensor_svd(tensor)
    print("SVD Components Shapes: U={}, S={}, Vt={}".format(U.shape, S.shape, Vt.shape))

    # Example 5: Tucker decomposition
    tensor = np.random.rand(6, 6, 6)
    core, factors = framework.tucker_decomposition(tensor, ranks=[3, 3, 3])
    print("Core Tensor Shape:", core.shape)
    for i, factor in enumerate(factors):
        print(f"Factor Matrix {i+1} Shape:", factor.shape)

    # Example 6: Visualize tensor
    framework.visualize_tensor(result, title="Contraction Result")
