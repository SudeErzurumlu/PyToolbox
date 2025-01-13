import numpy as np
from scipy.linalg import svd
from numpy.linalg import norm

class TensorDecomposition:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def unfold_tensor(self, mode):
        """Unfolds a tensor along the specified mode."""
        return np.reshape(np.moveaxis(self.tensor, mode, 0), (self.tensor.shape[mode], -1))

    def fold_tensor(self, unfolded, mode):
        """Folds an unfolded tensor back to its original shape."""
        full_shape = list(self.shape)
        full_shape[mode], full_shape[0] = full_shape[0], full_shape[mode]
        return np.moveaxis(np.reshape(unfolded, full_shape), 0, mode)

    def hosvd(self):
        """Performs Higher-Order Singular Value Decomposition (HOSVD)."""
        core_tensor = self.tensor
        factors = []

        for mode in range(len(self.shape)):
            unfolded = self.unfold_tensor(mode)
            u, _, _ = svd(unfolded, full_matrices=False)
            factors.append(u)
            core_tensor = np.tensordot(core_tensor, u.T, axes=(0, 1))

        return core_tensor, factors

    def cp_decomposition(self, rank, max_iter=100, tol=1e-6):
        """Performs CANDECOMP/PARAFAC (CP) decomposition using ALS."""
        dimensions = self.shape
        factors = [np.random.rand(dim, rank) for dim in dimensions]

        for iteration in range(max_iter):
            for mode in range(len(dimensions)):
                v = np.ones((rank, rank))
                for i, factor in enumerate(factors):
                    if i != mode:
                        v *= factor.T @ factor

                unfolded = self.unfold_tensor(mode)
                factors[mode] = unfolded @ np.linalg.pinv(np.kron(*[f for i, f in enumerate(factors) if i != mode]))
                factors[mode] /= norm(factors[mode], axis=0)

            # Check convergence (reconstruction error)
            reconstructed = self.reconstruct_tensor(factors)
            error = norm(self.tensor - reconstructed) / norm(self.tensor)
            if error < tol:
                break

        return factors

    def reconstruct_tensor(self, factors):
        """Reconstructs the tensor from its factors."""
        rank = factors[0].shape[1]
        reconstructed = np.zeros(self.tensor.shape)

        for r in range(rank):
            outer_product = np.outer(factors[0][:, r], factors[1][:, r])
            for f in factors[2:]:
                outer_product = np.outer(outer_product.flatten(), f[:, r]).reshape(self.shape)
            reconstructed += outer_product

        return reconstructed

    def tucker_decomposition(self, rank):
        """Performs Tucker decomposition."""
        core_tensor = self.tensor
        factors = []

        for mode in range(len(self.shape)):
            unfolded = self.unfold_tensor(mode)
            u, _, _ = svd(unfolded, full_matrices=False)
            factors.append(u[:, :rank])
            core_tensor = np.tensordot(core_tensor, u[:, :rank].T, axes=(0, 1))

        return core_tensor, factors

    def visualize_factors(self, factors):
        """Visualizes the factor matrices."""
        for i, factor in enumerate(factors):
            plt.figure(figsize=(8, 6))
            plt.imshow(factor, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Factor Matrix for Mode {i + 1}")
            plt.xlabel("Components")
            plt.ylabel("Original Dimensions")
            plt.show()

# Example Usage
if __name__ == "__main__":
    # Create a random 3D tensor
    tensor = np.random.rand(10, 10, 10)
    decomposition = TensorDecomposition(tensor)

    # Perform HOSVD
    core_tensor, factors_hosvd = decomposition.hosvd()
    print("HOSVD Core Tensor Shape:", core_tensor.shape)

    # Perform CP Decomposition
    factors_cp = decomposition.cp_decomposition(rank=3)
    print("CP Decomposition Factors:")
    for i, factor in enumerate(factors_cp):
        print(f"Mode-{i + 1} Factor Shape:", factor.shape)

    # Perform Tucker Decomposition
    core_tucker, factors_tucker = decomposition.tucker_decomposition(rank=5)
    print("Tucker Core Tensor Shape:", core_tucker.shape)

    # Visualize the first factor matrix
    decomposition.visualize_factors(factors_cp)
