"""
Spectral Clustering

Graph-based clustering. Builds a similarity graph between points, then
uses the eigenvectors of the graph Laplacian to project data into a
space where clusters become linearly separable (even if they weren't
in the original space) — good for non-convex / curved cluster shapes
where K-means fails.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import SpectralClustering, KMeans


def generate_data():
    # Two interleaving crescent moons - NOT linearly separable
    X, y_true = make_moons(n_samples=300, noise=0.07, random_state=42)
    return X, y_true


def run_spectral(X, n_clusters=2, affinity="nearest_neighbors", n_neighbors=10):
    """
    affinity: how pairwise similarity is computed.
      - "nearest_neighbors" : good for curved/manifold-like clusters
      - "rbf"               : good for blob-like clusters, sensitive to gamma
    n_neighbors: only used when affinity="nearest_neighbors".
    """
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=42,
    )
    labels = model.fit_predict(X)
    return labels


def compare_with_kmeans(X):
    kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
    spectral_labels = run_spectral(X, n_clusters=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="coolwarm", s=20)
    axes[0].set_title("K-Means (fails on curved clusters)")

    axes[1].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap="coolwarm", s=20)
    axes[1].set_title("Spectral Clustering (handles curved clusters)")

    plt.tight_layout()
    plt.savefig("spectral_vs_kmeans.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X, y_true = generate_data()
    compare_with_kmeans(X)
