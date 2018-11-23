#Polynomial Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynimail Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree = 2)
X_poly = ploy_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualisation (Linear Regression)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import r2_score
r2_score(y, lin_reg.predict(X))

#The red line showing the relation between y (Salary) and X (position) which seems not to be linear but rather polynomial
#The blue line gives the relation between the predicted output of X using linear regression (strait line)
#Using only Linear Regression does not give good results

#Visualisation (Polynomial Regression)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#The result presents a curve which is much better and closer to y (Salary)

from sklearn.metrics import r2_score
r2_score(y, lin_reg2.predict(X_poly))
#r2_score has remarquebly increased and improved compared to normal Linear Regression