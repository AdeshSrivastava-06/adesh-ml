import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from sklearn.pipeline import Pipeline

X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

plt.plot(X, y,'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.show()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
print("R2-score for linear regression--",r2_score(y_test,y_pred))

plt.plot(x_train,lr.predict(x_train),color='r')
plt.plot(X,y,'b.')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

"""applying ploy lr"""

poly = PolynomialFeatures(degree=2)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

print(x_train[0])
print(x_test_trans[0])

"""include _bias parameter=False in the model here we will not have to add X0 term in the polynomial features"""

lr = LinearRegression()
lr.fit(x_train_trans,y_train)

y_pred = lr.predict(x_test_trans)
print("R2-score for polynomial regression--",r2_score(y_test,y_pred))

print(lr.coef_,lr.intercept_) 
"""here we can see the coefficient of x^2 is 0.8 and coefficient of x is 0.9 and intercept is 2 which is close to the original equation"""

x_new = np.linspace(-3,3,200).reshape(200,1)
x_new_poly = poly.transform(x_new)
y_new = lr.predict(x_new_poly)

plt.plot(x_new,y_new,"r-",linewidth=2,label="Predictions")
plt.plot(x_train,y_train,'b.',label="Training data")
plt.plot(x_test,y_test,'g.',label="Testing data")
plt.xlabel("X")
plt.ylabel("Y") 
plt.legend()
plt.show()

"""if we increase the degree of the polynomial then we can get better fit but it may lead to overfitting it will relieve more on the training data and may not generalize well on the test data"""
