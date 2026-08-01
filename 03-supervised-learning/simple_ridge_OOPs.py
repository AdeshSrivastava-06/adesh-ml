from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)


from sklearn.linear_model import Ridge

rr = Ridge(alpha=10)
rr.fit(X,y)
print(rr.coef_)
print(rr.intercept_)

rr1 = Ridge(alpha=100)
rr1.fit(X,y)
print(rr1.coef_)
print(rr1.intercept_)

plt.plot(X,y,'b.')
plt.plot(X,lr.predict(X),color='red',label='alpha=0')
plt.plot(X,rr.predict(X),color='green',label='alpha=10')
plt.plot(X,rr1.predict(X),color='orange',label='alpha=100')
plt.legend()
plt.show()

class CustomRidge:
    
    def __init__(self,alpha=1.0):
        self.alpha = alpha
        self.m = None
        self.b = None
    
    def fit(self,X_train,y_train):
        
        dem = 0
        num = 0
        
        for i in range(X_train.shape[0]):
            num = num + (y_train[i] - y_train.mean()) * (X_train[i] - X_train.mean())
            
            dem = dem + (X_train[i] - X_train.mean())**2
            
        self.m = num/(dem+self.alpha)
            
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m,self.b)
        
    def predict(self,X_test):
        pass
    
reg = CustomRidge(alpha=10)
reg.fit(X,y)
