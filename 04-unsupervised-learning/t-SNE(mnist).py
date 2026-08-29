"""
t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for visualizing high-dimensional data
in 2D/3D. Unlike PCA, t-SNE preserves local structure (neighbors stay
close) rather than global variance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    return X, y


def run_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    perplexity: roughly the number of effective nearest neighbors.
    Typical range 5-50. Small perplexity -> tight local clusters.
    Large perplexity -> more global structure preserved.
    """
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    X_embedded = tsne.fit_transform(X_scaled)
    return X_embedded


def plot_embedding(X_embedded, y, title="t-SNE on Digits Dataset"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1],
        c=y, cmap="tab10", s=15, alpha=0.8
    )
    plt.legend(*scatter.legend_elements(), title="Digit", loc="best")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig("tsne_digits.png", dpi=150)
    plt.show()


def compare_perplexities(X, y, perplexities=(5, 30, 50, 100)):
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))
    for ax, p in zip(axes, perplexities):
        X_embedded = run_tsne(X, perplexity=p)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=10)
        ax.set_title(f"Perplexity = {p}")
    plt.tight_layout()
    plt.savefig("tsne_perplexity_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X, y = load_data()

    X_embedded = run_tsne(X, perplexity=30)
    plot_embedding(X_embedded, y)
    
    compare_perplexities(X, y)
