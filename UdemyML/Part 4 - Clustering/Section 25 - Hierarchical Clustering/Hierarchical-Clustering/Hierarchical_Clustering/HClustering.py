#importing lybraries :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataste:
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

#using dendrogram to find best nb of clusters :
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('dinstance')
plt.show()

#import the library:
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing clusters:
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, color='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, color='magenta', label='Cluster 5')
plt.title('Cluster of clients')
plt.xlabel('income')
plt.ylabel('spending')
plt.legend()
plt.show()