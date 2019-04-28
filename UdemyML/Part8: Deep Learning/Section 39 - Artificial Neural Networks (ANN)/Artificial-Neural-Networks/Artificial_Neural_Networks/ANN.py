#Artificial Neural Networks:

#importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset:
dataset = pd.read_csv('Churn_Modelling.csv')

#vector of independent variables (features)
X = dataset.iloc[:, 3:13].values

#vector of dependent variable :
y = dataset.iloc[:, 13]

#Encoding categorical variables:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encode the geography feature (France, Germany, Spain...)
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])
#encode gender feature (Male, Female)
labelencoder2 = LabelEncoder()
X[:, 2] = labelencoder2.fit_transform(X[:, 2])

#dummy variables (only with geography since the Gender has 2 values)
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
#drop the first dummy variable to avoid dummy variable trap! !
X = X[:, 1:]

#Splitting dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#For ANN feature scaling is a "must"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Creating the NN:
import keras
#sequential => initialize the neural network
from keras.models import Sequential
#Dense => build layers of the neural network
from keras.layers import Dense

#initialize the ANN
classifier = Sequential()

#add input layer and first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#add second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#output layer:
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to data
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Predictions:
y_pred = classifier.predict(X_test) #this contains probabilities
y_pred = (y_pred > 0.5) #here we transformed it into a binary vector (if prob>0.5 -> 1, else 0)

#Metrics :
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
