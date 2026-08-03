import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve

# ── Load Data 
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Train models at different depths 
depths       = list(range(1, 21))
train_errors = []
test_errors  = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_errors.append(1 - clf.score(X_train, y_train))
    test_errors.append( 1 - clf.score(X_test,  y_test))

best_depth = depths[np.argmin(test_errors)]

# ── Learning Curve at best depth 
clf_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
train_sizes, train_sc, val_sc = learning_curve(
    clf_best, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

# ── Plot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Bias-Variance Tradeoff — Breast Cancer Dataset",
             fontsize=14, fontweight="bold")

# Left: Train vs Test Error
ax1.plot(depths, train_errors, "o-", color="#2196F3", label="Train Error")
ax1.plot(depths, test_errors,  "s-", color="#F44336", label="Test Error")
ax1.axvline(best_depth, color="green", linestyle="--",
            label=f"Best Depth = {best_depth}")
ax1.set_xlabel("Tree Depth  (complexity →)")
ax1.set_ylabel("Error Rate")
ax1.set_title("Train vs Test Error")
ax1.legend()
ax1.grid(alpha=0.3)

ax1.text(2,   max(test_errors)*0.85,  "Underfitting\n(High Bias)",
         color="#2196F3", fontsize=9, ha="center")
ax1.text(17,  max(test_errors)*0.85, "Overfitting\n(High Variance)",
         color="#F44336", fontsize=9, ha="center")

# Right: Learning Curve
ax2.plot(train_sizes, train_sc.mean(axis=1), "o-", color="#2196F3", label="Train Accuracy")
ax2.plot(train_sizes, val_sc.mean(axis=1),   "s-", color="#F44336", label="Validation Accuracy")
ax2.fill_between(train_sizes,
                 val_sc.mean(axis=1) - val_sc.std(axis=1),
                 val_sc.mean(axis=1) + val_sc.std(axis=1),
                 alpha=0.15, color="#F44336")
ax2.set_xlabel("Training Samples")
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Learning Curve (depth = {best_depth})")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bias_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Best depth: {best_depth}")
print(f"Train Error: {train_errors[best_depth-1]:.4f}")
print(f"Test  Error: {test_errors[best_depth-1]:.4f}")
