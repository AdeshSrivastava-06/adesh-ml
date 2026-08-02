import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score

print("MSE-Linear Regression:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score-Linear Regression:", r2_score(y_test, y_pred))


alphas = [0.01,1,5,10,30]
plt.figure(figsize=(12,6))
plt.scatter(X,y)
for i in alphas:
    L = Lasso(alpha=i) #alpha is the regularization parameter that controls the strength of the regularization. A higher alpha value means more regularization, which can lead to a simpler model with fewer features. In Lasso regression, increasing alpha can also lead to some coefficients being exactly zero, effectively performing feature selection.
    L.fit(X_train,y_train)
    plt.plot(X_test,L.predict(X_test),label='alpha={}'.format(i))
    y_pred_lasso = L.predict(X_test)
    print("MSE-Lasso Regression with alpha {}: {}".format(i, np.sqrt(mean_squared_error(y_test, y_pred_lasso))))
    print("R2 Score-Lasso Regression with alpha {}: {}".format(i, r2_score(y_test, y_pred_lasso)))
    print('alpha: {}, coef: {}, intercept: {}'.format(i, L.coef_, L.intercept_))
plt.legend()
plt.show()

"""here we can see as the alpha increases the line becomes more flat and at alpha=30 it is almost a straight line. This is because as we increase the alpha, the regularization term becomes more significant and the model is forced to fit a simpler line to the data, which can lead to underfitting."""
