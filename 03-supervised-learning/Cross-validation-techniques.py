"""
Cross Validation Techniques

Quick comparison of KFold vs Stratified KFold cross-validation.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Load data
X, y = load_breast_cancer(return_X_y=True)

model = LogisticRegression(max_iter=5000)

# KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X, y, cv=kfold)

# Stratified KFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skfold_scores = cross_val_score(model, X, y, cv=skfold)

print("KFold scores:", np.round(kfold_scores, 4))
print("KFold mean accuracy:", kfold_scores.mean())

print("\nStratified KFold scores:", np.round(skfold_scores, 4))
print("Stratified KFold mean accuracy:", skfold_scores.mean())
