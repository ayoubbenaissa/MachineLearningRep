#Data Prerocessing

#Importing libraries
import numpy as numpy #used for mathematical operations
import matplotlib.pyplot as plt #used for plotting (charts...)
import pandas as pd #used for importing and managing data sets

#Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
#sklearn -> many machine learning models
# Imputer => take care of missing data
from sklearn.preprocessing import Imputer
#Crtl + i = Inspect element to get informations about it 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #axis = 0 calculate the mean on the columns
#replacing vars having "nan" values with the "mean" vlaue in Y-axis (axis=0)
imputer= imputer.fit(X[:, 1:3])
#indexing strats from 0, lower bound is included but upper one is not!
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Importing categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#LabelEncoder = transforms text into numerical values
#OneHotEncoder = used to make dummy variables
#dummy vars are used to prevent ml algo from thinking that a cathegorical data is greater than another
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features= [0]) #deals with the first column
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#feature scaling = giving features the same importance + algo will converge must faster
#exp if models using euclidean distance, the result will hugely depend on the feature having the bigger value (in our exp the Salary)
#feature scaling : Standarisation, Normalisation

#Feature scaling :
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
