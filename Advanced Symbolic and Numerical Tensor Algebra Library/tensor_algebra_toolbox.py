import numpy as np
import sympy as sp
from numpy.linalg import svd, eig
from scipy.linalg import khatri_rao

# Symbolic Tensor Algebra
def symbolic_tensor(rank, dims, name):
    """
    Creates a symbolic tensor of given rank and dimensions.
    Args:
        rank (int): Rank of the tensor.
        dims (tuple): Dimensions of the tensor.
        name (str): Base name for tensor elements.
    Returns:
        sympy.MutableDenseNDimArray: Symbolic tensor.
    """
    indices = tuple([sp.symbols(f"{name}_{i}_{j}") for i in range(dims[0]) for j in range(dims[1])])
    return sp.MutableDenseNDimArray(indices, dims)

# Example: 3x3x3 Symbolic Tensor
dims = (3, 3, 3)
T = symbolic_tensor(3, dims, 'T')
print("Symbolic Tensor:")
print(T)

# Tensor Contraction
def tensor_contract(tensor, axes):
    """
    Contracts a tensor along the given axes.
    Args:
        tensor (np.ndarray): Input tensor.
        axes (tuple): Axes to contract.
    Returns:
        np.ndarray: Resulting tensor after contraction.
    """
    return np.tensordot(tensor, tensor, axes=axes)

# Numerical Tensor Operations
# Create Random 3D Tensor
tensor = np.random.rand(3, 3, 3)

# Tensor Contraction Example
contracted = tensor_contract(tensor, axes=([0], [1]))
print("Tensor Contraction Result:")
print(contracted)

# Eigenvalue Decomposition
def tensor_eigen(tensor):
    """
    Computes the eigenvalues and eigenvectors of a square tensor.
    Args:
        tensor (np.ndarray): Square tensor.
    Returns:
        tuple: Eigenvalues and eigenvectors.
    """
    reshaped_tensor = tensor.reshape(tensor.shape[0], -1)
    values, vectors = eig(reshaped_tensor)
    return values, vectors

eigenvalues, eigenvectors = tensor_eigen(tensor)
print("Tensor Eigenvalues:")
print(eigenvalues)

# Kronecker Product
def tensor_kronecker(tensor1, tensor2):
    """
    Computes the Kronecker product of two tensors.
    Args:
        tensor1 (np.ndarray): First tensor.
        tensor2 (np.ndarray): Second tensor.
    Returns:
        np.ndarray: Kronecker product.
    """
    return np.kron(tensor1, tensor2)

tensor2 = np.random.rand(3, 3)
kronecker_result = tensor_kronecker(tensor, tensor2)
print("Kronecker Product Result:")
print(kronecker_result)

# Tensor Visualization
def visualize_tensor(tensor):
    """
    Visualizes a tensor as a heatmap.
    Args:
        tensor (np.ndarray): Tensor to visualize.
    """
    import matplotlib.pyplot as plt
    if tensor.ndim == 2:
        plt.imshow(tensor, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("Tensor Heatmap")
        plt.show()
    else:
        raise ValueError("Visualization is only supported for 2D tensors.")

visualize_tensor(tensor[:, :, 0])

# Symbolic Gradients
x, y = sp.symbols('x y')
F = sp.Matrix([[x**2 + y**2, x*y], [sp.sin(x), sp.cos(y)]])  # Tensor-Valued Function
grad_F = F.diff(x)
print("Symbolic Gradient of Tensor Function:")
sp.pprint(grad_F)

# Advanced Tensor Decomposition (CP Decomposition)
def cp_decomposition(tensor, rank):
    """
    Performs CP (CANDECOMP/PARAFAC) decomposition on a tensor.
    Args:
        tensor (np.ndarray): Input tensor.
        rank (int): Desired rank for decomposition.
    Returns:
        list: Factor matrices.
    """
    from tensorly.decomposition import parafac
    import tensorly as tl
    tl_tensor = tl.tensor(tensor)
    factors = parafac(tl_tensor, rank=rank)
    return factors

from tensorly.random import random_cp
decomposed_factors = cp_decomposition(tensor, rank=2)
print("CP Decomposition Factors:")
for i, factor in enumerate(decomposed_factors):
    print(f"Factor {i}:")
    print(factor)
