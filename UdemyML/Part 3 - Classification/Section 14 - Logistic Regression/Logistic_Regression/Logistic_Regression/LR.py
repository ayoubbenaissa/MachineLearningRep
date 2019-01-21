#Logistic Regression

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
#we are going to use the age and estimated Salary to predict whether to buy or not
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

#Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Logistic Regression requires using feature scaling :
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test) #of course sc_X is fitted to X_train which is scaled at the same basis so no need to refit it for X_test

plt.scatter(X_train[:, 0], y_train, color = 'blue')
plt.show() #we can easely see that this is a problem best handeled with LogisticRegression

plt.scatter(X_train[:, 1], y_train, color = 'red')
plt.show()

#Logistic Regression Library :
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0)

clf.fit(X_train, y_train)

#Prediction :
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
accuracy_score(y_test, y_pred)
r2_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




