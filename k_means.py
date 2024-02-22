import numpy as np
import pandas as pd


class KMeans:
    def __init__(self, n_clusters=6, n_iterations=10):
        self.K = n_clusters
        self.n_iterations = n_iterations
        self.centroids = None

    @staticmethod
    def fit_input(X):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, pd.DataFrame):
            return X.values
        else:
            raise TypeError('Unsupported input type')

    def _find_closest_centroids(self, X):
        m, n = X.shape
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            distances = []
            for j in range(self.K):
                # Euclidean distance
                norm_ij = np.linalg.norm(X[i] - self.centroids[j], ord=2)
                distances.append(norm_ij)
            idx[i] = np.argmin(distances)

        return idx

    def _compute_centroids(self, X, idx):
        m, n = X.shape
        for j in range(self.K):
            points = X[idx == j]
            if len(points) > 0:
                self.centroids[j] = np.mean(points, axis=0)

    def _init_centroids(self, X):
        m, n = X.shape
        random_idx = np.random.permutation(m)
        self.centroids = X[random_idx[:self.K]]

    def fit(self, X):
        X = self.fit_input(X)
        # init centroids
        self._init_centroids(X)
        # get centroids
        for i in range(self.n_iterations):
            idx = self._find_closest_centroids(X)
            self._compute_centroids(X, idx)
            print(f"K-Means iteration {i+1}/{self.n_iterations}")

    def predict(self, X):
        X = self.fit_input(X)
        idx = self._find_closest_centroids(X)
        return idx


