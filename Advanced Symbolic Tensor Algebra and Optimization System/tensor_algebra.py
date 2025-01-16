import sympy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd
import matplotlib.pyplot as plt

class TensorAlgebra:
    def __init__(self):
        self.tensors = {}

    def define_tensor(self, name, shape):
        """Define a symbolic tensor with the given name and shape."""
        tensor = sp.MatrixSymbol(name, *shape)
        self.tensors[name] = tensor
        return tensor

    def tensor_contract(self, tensor_a, tensor_b, axes):
        """Perform tensor contraction along specified axes."""
        contracted = sp.tensorcontraction(sp.tensorproduct(tensor_a, tensor_b), axes)
        return contracted

    def tensor_decomposition(self, tensor):
        """Perform SVD decomposition of a tensor."""
        u, s, vh = svd(tensor)
        return u, s, vh

    def symbolic_gradient(self, cost_function, tensor):
        """Compute the symbolic gradient of a cost function with respect to a tensor."""
        return sp.derive_by_array(cost_function, tensor)

    def optimize_tensor(self, cost_function, initial_value, constraints=None):
        """Optimize a tensor-based cost function numerically."""
        def objective(x):
            return cost_function(*x)

        result = minimize(objective, initial_value, constraints=constraints)
        return result

    def visualize_tensor(self, tensor_data):
        """Visualize a tensor (matrix or 3D structure)."""
        if tensor_data.ndim == 2:
            plt.imshow(tensor_data, cmap='viridis')
            plt.colorbar()
            plt.title("Tensor Heatmap")
        elif tensor_data.ndim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = np.where(tensor_data > 0)
            ax.scatter(x, y, z, c=tensor_data[x, y, z], cmap='viridis')
            ax.set_title("3D Tensor Visualization")
        else:
            raise ValueError("Unsupported tensor dimensionality for visualization.")
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Create an instance of TensorAlgebra
    tensor_system = TensorAlgebra()

    # Define two tensors
    tensor_a = tensor_system.define_tensor("A", (3, 3))
    tensor_b = tensor_system.define_tensor("B", (3, 3))

    # Define a symbolic cost function (e.g., Frobenius norm)
    cost_function = sp.sqrt(sp.Sum(tensor_a[i, j]**2 for i in range(3) for j in range(3)))

    # Compute the gradient of the cost function with respect to tensor_a
    gradient = tensor_system.symbolic_gradient(cost_function, tensor_a)
    print("Symbolic Gradient of Cost Function:", gradient)

    # Perform tensor contraction
    contracted_tensor = tensor_system.tensor_contract(tensor_a, tensor_b, (0, 1))
    print("Contracted Tensor:", contracted_tensor)

    # Simulate a numerical tensor (example data)
    numerical_tensor = np.random.rand(3, 3)

    # Perform tensor decomposition (SVD)
    u, s, vh = tensor_system.tensor_decomposition(numerical_tensor)
    print("SVD Decomposition Results:")
    print("U:", u)
    print("Singular Values:", s)
    print("Vh:", vh)

    # Optimize the cost function numerically
    initial_guess = np.random.rand(3, 3).flatten()
    optimized_result = tensor_system.optimize_tensor(cost_function, initial_guess)
    print("Optimized Tensor:", optimized_result)

    # Visualize the tensor
    tensor_system.visualize_tensor(numerical_tensor)
