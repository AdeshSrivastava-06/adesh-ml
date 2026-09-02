"""
Blending

Blending is similar to stacking, but instead of using k-fold
out-of-fold predictions to train the meta-model, it uses a simple
holdout set: base models are trained on one chunk of the training
data, then their predictions on a separate holdout chunk are used
as features to train the meta-model.

Simpler and faster than stacking (no k-fold looping), but uses less
data per base model and the meta-model sees fewer examples.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and split data
X, y = load_breast_cancer(return_X_y=True)

# Split: train (for base models) -> holdout (to generate meta-features) -> test (final eval)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_holdout, X_test, y_holdout, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} | Holdout: {X_holdout.shape[0]} | Test: {X_test.shape[0]}")

# 2. Define base models
base_models = {
    "log_reg": LogisticRegression(max_iter=5000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 3. Train base models on X_train, predict probabilities on holdout
holdout_meta_features = np.zeros((X_holdout.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    model.fit(X_train, y_train)

    holdout_preds = model.predict_proba(X_holdout)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    holdout_meta_features[:, i] = holdout_preds
    test_meta_features[:, i] = test_preds

    acc = accuracy_score(y_holdout, model.predict(X_holdout))
    print(f"  Base model '{name}' holdout accuracy: {acc:.4f}")


# 4. Train meta-model on holdout predictions

meta_model = LogisticRegression(max_iter=5000)
meta_model.fit(holdout_meta_features, y_holdout)


# 5. Final evaluation on test set

blend_preds = meta_model.predict(test_meta_features)
blend_acc = accuracy_score(y_test, blend_preds)

print(f"\nBlended ensemble accuracy on test set: {blend_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, blend_preds))


# 6. Compare against best individual base model on the same test set

print("Individual base model accuracy on test set (for comparison):")
for name, model in base_models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  {name}: {acc:.4f}")
