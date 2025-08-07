import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    A class-based implementation of the K-Means Clustering algorithm written from scratch.
    """
    
    def __init__(self, k=3, max_iters=100, tolerance=1e-4, random_state=None):
        """
        Initializes the KMeans clustering object.

        Parameters:
        - k: Number of clusters
        - max_iters: Maximum number of iterations to run the algorithm
        - tolerance: Threshold for convergence
        - random_state: Seed for reproducibility
        """
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fits the KMeans algorithm to the dataset X.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Step 1: Initialize centroids randomly from the data points
        random_indices = np.random.permutation(X.shape[0])
        self.centroids = X[random_indices[:self.k]]

        for i in range(self.max_iters):
            # Step 2: Assign each point to the nearest centroid
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)

            # Step 3: Compute new centroids from the clusters
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            # Step 4: Check convergence (if centroids do not change significantly)
            if np.linalg.norm(self.centroids - new_centroids) < self.tolerance:
                break

            self.centroids = new_centroids
            self.labels = new_labels

    def predict(self, X):
        """
        Assigns a cluster label to each point in X.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        
        Returns:
        - labels: Cluster index for each sample
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        """
        Computes the Euclidean distance between each point and the centroids.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - distances: numpy array of shape (n_samples, k)
        """
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def plot_clusters(self, X):
        """
        Visualizes the clustered data (only for 2D data).

        Parameters:
        - X: numpy array of shape (n_samples, 2)
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot 2D data.")
        
        plt.figure(figsize=(8, 6))
        for i in range(self.k):
            plt.scatter(X[self.labels == i][:, 0], X[self.labels == i][:, 1], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
        plt.title("K-Means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()
