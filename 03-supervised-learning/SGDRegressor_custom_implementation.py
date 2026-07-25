from sklearn.datasets import load_diabetes

import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time
X,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg = LinearRegression()
reg.fit(x_train,y_train)

print("Coefficients:--",reg.coef_)
print("Intercept:--",reg.intercept_)

y_pred = reg.predict(x_test)
print("R2 Score for Linear Regression using sklearn:", r2_score(y_test,y_pred))

class SGDegressor:
    
    def __init__(self,learning_rate = 0.01,epochs=100):
        self.coef_=None
        self.intercept_=None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,x_train_scaled,y_train):
        #init your coefficients
        self.intercept_ = 0
        self.coef_=np.ones(x_train_scaled.shape[1])
        
        for i in range(self.epochs):
            for j in range(x_train_scaled.shape[0]):
                idx = np.random.randint(0,x_train_scaled.shape[0]) 

                y_hat = np.dot(x_train_scaled[idx],self.coef_) + self.intercept_ #now it will be a single number
                
                intercept_der = -2 * (y_train[idx] - y_hat)
                
                self.intercept_ = self.intercept_ - (self.lr*intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx]-y_hat),x_train_scaled[idx])
                
                self.coef_ = self.coef_ - (self.lr * coef_der)
        print(self.intercept_,self.coef_)
        
        pass
    
    def predict(self,x_test_scaled):
        return np.dot(x_test_scaled,self.coef_) + self.intercept_
    
sgd = SGDegressor(learning_rate=0.001,epochs=500)
start = time.time()
sgd.fit(x_train_scaled,y_train)
print("Time taken is:-",time.time() - start)

y_pred = sgd.predict(x_test_scaled)

print("R2 Score for SGDRegressor using custom implementation:", r2_score(y_test,y_pred))

"""using sklearn"""
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(
    loss='squared_error',   # same as MSE
    learning_rate='invscaling',
    eta0=0.001,              # learning rate
    max_iter=500,            # epochs
    random_state=2
)

# Training
start = time.time()
sgd.fit(x_train_scaled, y_train)
print("Time taken is:-", time.time() - start)

# Coefficients
print("Coefficients:", sgd.coef_)
print("Intercept:", sgd.intercept_)

# Prediction
y_pred = sgd.predict(x_test_scaled)

# R2 score
print("R2 Score for SGDRegressor using sklearn:", r2_score(y_test, y_pred))
