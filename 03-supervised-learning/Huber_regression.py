"""
Huber Regression 
    
Robust regression that behaves like OLS for small residuals and like
Mean Absolute Error (L1) for large residuals (outliers), controlled by
the epsilon parameter. Compared here against plain Linear Regression
on data contaminated with outliers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

# Generate simple linear data
rng = np.random.RandomState(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 3 * X.ravel() + 5 + rng.normal(0, 1, 50)

# Add a few outliers
y[5] += 50
y[25] += 60
y[40] -= 55

# Fit both models
linear_model = LinearRegression().fit(X, y)
huber_model = HuberRegressor().fit(X, y)

# Compare
print("Linear Regression -> slope:", linear_model.coef_[0], "intercept:", linear_model.intercept_)
print("Huber Regression   -> slope:", huber_model.coef_[0], "intercept:", huber_model.intercept_)

print("\nLinear Regression MSE:", mean_squared_error(y, linear_model.predict(X)))
print("Huber Regression MSE:  ", mean_squared_error(y, huber_model.predict(X)))

# Plot comparison
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="steelblue", label="Data (with outliers)")
plt.plot(X, linear_model.predict(X), color="orange", linewidth=2, label="Linear Regression")
plt.plot(X, huber_model.predict(X), color="green", linewidth=2, label="Huber Regression")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Huber Regression vs Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("huber_vs_linear_simple.png", dpi=120)
plt.show()
