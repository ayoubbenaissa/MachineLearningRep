# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data : LabelEncoder
#Dummy variables : OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableencoder_X = LabelEncoder()
X[:, 3] = lableencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoid Dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#MLR library does feature scaling => no need to do it manually

#Fitting MLR to raining set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#In countinuous case it is recommended to use r2_score instead of accuracy_score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Backward Elimination :
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#here we need at add a column of one at the beginning of the metrix X (this column will be used for b0 in Y = b0 + b1*X1+...+bn*Xn)
#we putted X as values so the column will be added at the beginning
#we must transform no.one to int by : astype(int)

X_opt = X[:, [0 ,1, 2 ,3 ,4 ,5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
#search for feature with highest P-value and see if P-value > SL (SL = 0.005)
#in this case feature = x2 => we drop it since Px2 = 0.9 > SL = 0.005

X_opt = X[:, [0, 1, 3 ,4 ,5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
#const = column of ones
X_opt = X[:, [0, 3 ,4 ,5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

