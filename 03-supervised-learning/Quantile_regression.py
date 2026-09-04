"""
Quantile Regression]

Normal Linear Regression only tells you the average value of y for
a given X. But sometimes you want more than just the average — you
want to know the range y could fall in.
 
Quantile Regression lets you predict any percentile of y, not just
the mean. For example:
    - 10th percentile line -> "only 10% of points are expected below this"
    - 50th percentile line -> the median (middle) prediction
    - 90th percentile line -> "90% of points are expected below this"
 
Together, the 10th and 90th percentile lines form a band that most
of your data (80% of it) should fall inside. This is useful when
the spread of your data isn't constant everywhere (e.g. predictions
get less certain as X grows), which a single mean line can't show.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor

# Generate simple data
rng = np.random.RandomState(42)
X = rng.uniform(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 5 + rng.normal(0, 2, 100)

# Fit quantile regressors
quantiles = [0.1, 0.5, 0.9]
colors = ["blue", "green", "red"]

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="lightgray", label="Data")

X_line = np.linspace(0, 10, 100).reshape(-1, 1)

for q, c in zip(quantiles, colors):
    model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
    model.fit(X, y)
    plt.plot(X_line, model.predict(X_line), color=c, linewidth=2, label=f"Quantile {q}")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Quantile Regression")
plt.legend()
plt.tight_layout()
plt.savefig("quantile_regression_simple.png", dpi=120)
plt.show()
