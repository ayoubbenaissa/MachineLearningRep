#importing lybraries :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataste:
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

#using elbow method to determine optimal number of clusters:
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #append the value of the wcss for number of clusters equal to 'i'
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clsuters')
plt.ylabel('WCSS')
plt.show()
#By analysing the graph, I see that picking 5 clusters is the best choice xD

#Applying KMeans algo and use 5 clusters:
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#Visualizing clusters:
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, color='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, color='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color='yellow', label='centroids')
plt.title('Cluster of clients')
plt.xlabel('income')
plt.ylabel('spending')
plt.legend()
plt.show()