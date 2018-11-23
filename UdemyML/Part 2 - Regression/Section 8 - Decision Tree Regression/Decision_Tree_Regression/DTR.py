#Support Vector Regression

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #here we used 1:2 so that X will be treated as a metrix rather than vector
y = dataset.iloc[:, 2].values

#Fitting Decision Tree to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)

#Visualising Decision tree regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
#this not appropriate for Decision Tree because the model is not a continuous one

#Visualising with Higher Resolution
X_grid = np.arange(min(X), max(X), 0.05)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

#the decision tree algo calculates the average
#so for a 1D it must give a constant which is the average in each interval and that is why we had the second output plot