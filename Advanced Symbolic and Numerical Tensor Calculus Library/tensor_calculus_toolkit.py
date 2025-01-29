import numpy as np
import sympy as sp
import jax.numpy as jnp
from jax import grad, jit
from scipy.linalg import svd

class TensorCalculusToolkit:
    def __init__(self):
        pass

    # Define a symbolic tensor
    def symbolic_tensor(self, name, shape):
        """Create a symbolic tensor with given shape."""
        indices = [sp.symbols(f"{name}_{i}{j}") for i in range(shape[0]) for j in range(shape[1])]
        return sp.Matrix(shape, indices)

    # Tensor contraction using Einstein Summation Notation
    def tensor_contraction(self, A, B, equation):
        """Perform tensor contraction using Einstein summation."""
        return np.einsum(equation, A, B)

    # Automatic Differentiation (Gradient of a function)
    def tensor_gradient(self, func):
        """Compute gradient using JAX for automatic differentiation."""
        return jit(grad(func))

    # Tensor Decomposition (Singular Value Decomposition)
    def tensor_decomposition(self, tensor):
        """Perform Singular Value Decomposition (SVD) on a tensor."""
        U, S, Vt = svd(tensor, full_matrices=False)
        return U, S, Vt

    # Tensor Laplacian (Second Order Derivatives)
    def tensor_laplacian(self, f, vars):
        """Compute the Laplacian of a function in given variables."""
        return sum(sp.diff(f, v, v) for v in vars)

    # Parallelized Einstein Summation
    def parallel_einsum(self, equation, *tensors):
        """Perform high-speed Einstein summation using JAX."""
        return jnp.einsum(equation, *tensors)

# Example Usage
if __name__ == "__main__":
    toolkit = TensorCalculusToolkit()

    # Example 1: Symbolic tensor creation
    tensor = toolkit.symbolic_tensor("T", (2, 2))
    print("Symbolic Tensor:\n", tensor)

    # Example 2: Tensor contraction (Einstein Summation)
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = toolkit.tensor_contraction(A, B, "ij,jk->ik")
    print("Tensor Contraction Result:\n", result)

    # Example 3: Gradient computation using JAX
    def sample_function(x):
        return jnp.sin(x[0]) * jnp.cos(x[1])
    
    gradient_fn = toolkit.tensor_gradient(sample_function)
    print("Tensor Gradient at [π/4, π/4]:", gradient_fn(jnp.array([np.pi / 4, np.pi / 4])))

    # Example 4: SVD Decomposition of a tensor
    U, S, Vt = toolkit.tensor_decomposition(A)
    print("SVD Decomposition:\nU:\n", U, "\nS:\n", S, "\nVt:\n", Vt)

    # Example 5: Compute Laplacian of a symbolic function
    x, y = sp.symbols("x y")
    f = sp.exp(-x**2 - y**2)
    laplacian_result = toolkit.tensor_laplacian(f, [x, y])
    print("Laplacian of Function:", laplacian_result)

    # Example 6: Parallelized Einstein Summation
    result_parallel = toolkit.parallel_einsum("ij,jk->ik", jnp.array(A), jnp.array(B))
    print("Parallelized Tensor Contraction:\n", result_parallel)
