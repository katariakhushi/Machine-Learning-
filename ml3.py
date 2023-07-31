import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split

data = datasets.load_diabetes()
X = data["data"]
Y = data["target"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)