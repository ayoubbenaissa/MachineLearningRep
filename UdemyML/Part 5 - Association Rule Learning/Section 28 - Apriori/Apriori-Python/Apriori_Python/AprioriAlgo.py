import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#preparing the input (list of lists):
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#import the apriori algorithm
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#for min_support : we are only interested in products bought 3 at least times a day
# min_support = 3*7/7500 = 0.003

#high values of confidence is not very good 

results = list(rules)
