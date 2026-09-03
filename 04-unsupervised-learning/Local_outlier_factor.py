"""
Local Outlier Factor (LOF)

Density-based anomaly detection. Flags a point as an outlier if its
local density is significantly lower than its neighbors' density.
Unlike global methods, LOF adapts to regions of varying density.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs


def generate_data():
    # Two dense clusters + a handful of scattered outlier points
    X_inliers, _ = make_blobs(
        n_samples=200, centers=[[2, 2], [-2, -2]],
        cluster_std=0.5, random_state=42
    )
    rng = np.random.RandomState(42)
    X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))

    X = np.vstack([X_inliers, X_outliers])
    return X


def run_lof(X, n_neighbors=20, contamination=0.1):
    """
    n_neighbors: how many nearby points define "local" density.
    contamination: expected proportion of outliers in the dataset
                    (used to set the decision threshold).
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)          # 1 = inlier, -1 = outlier
    scores = -lof.negative_outlier_factor_  # higher = more outlier-like

    return y_pred, scores


def plot_results(X, y_pred, scores):
    plt.figure(figsize=(8, 6))

    # Size of each point scaled by outlier score
    radius = (scores - scores.min()) / (scores.max() - scores.min())
    colors = np.where(y_pred == -1, "red", "steelblue")

    plt.scatter(X[:, 0], X[:, 1], c=colors, s=1000 * radius + 20,
                edgecolors="k", alpha=0.6, label="Point size = outlier score")
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=15, edgecolors="k")

    plt.title("Local Outlier Factor (red = flagged outlier)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("lof_outliers.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X = generate_data()
    y_pred, scores = run_lof(X, n_neighbors=20, contamination=0.1)

    n_outliers = np.sum(y_pred == -1)
    print(f"Detected {n_outliers} outliers out of {len(X)} points")

    plot_results(X, y_pred, scores)
