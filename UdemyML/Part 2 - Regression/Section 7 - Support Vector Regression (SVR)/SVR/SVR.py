#Support Vector Regression

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #here we used 1:2 so that X will be treated as a metrix rather than vector
y = dataset.iloc[:, 2].values

#Since SVR doesn't do feature scaling we must do it manually
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1)) #reshape used since StandardScaler.fit_transform needs an array of 2D as input

#Importing the SVR 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

#fitting regressor to dataset
regressor.fit(X, y)

#Visualisations
plt.scatter(X, y, color = 'blue')
plt.plot(X, regressor.predict(X), color = 'red')