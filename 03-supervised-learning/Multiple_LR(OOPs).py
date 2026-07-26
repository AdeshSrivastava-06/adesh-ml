import numpy as np
from sklearn.datasets import load_diabetes

X,y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)

"""making our own class"""

class Multiple_LR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self,x_train,y_train):
        x_train = np.insert(x_train,0,1,axis=1) 

        betas = np.linalg.inv(np.dot(x_train.T,x_train)).dot(x_train.T).dot(y_train)
        print(betas)
        
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
        
    def predict(self,x_test):
        y_pred = np.dot(x_test,self.coef_) +  self.intercept_
        return y_pred
    
lr = Multiple_LR()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
print(r2_score(y_test,y_pred))

print(lr.coef_)
print(lr.intercept_)
