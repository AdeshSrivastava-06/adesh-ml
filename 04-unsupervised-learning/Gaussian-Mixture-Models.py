# Gaussian Mixture Model (GMM) Clustering
# GMM does soft clustering - each point gets a probability of belonging 
# to each cluster (unlike K-means where assignment is hard/fixed)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Step 1: Create fake data for testing (3 clusters)
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.80, random_state=42)

# Step 2: Fit the GMM model
# n_components = number of clusters we want (same as k in K-means)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Step 3: Get predicted cluster labels
labels = gmm.predict(X)

# Step 4: Get probability of each point belonging to each cluster
# This is the special part of GMM - not just one label, but probabilities
probs = gmm.predict_proba(X)
print("Cluster probabilities for first 5 points:")
print(probs[:5].round(3))

# Step 5: Plot the clustering result
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)

# Also plot the cluster centers (means)
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')

plt.title("GMM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Step 6: Check model quality using AIC/BIC score
print("AIC Score:", gmm.aic(X))
print("BIC Score:", gmm.bic(X))
