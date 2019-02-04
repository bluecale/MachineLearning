"""
GaussianNB: continuos data
MultinomialNB: binary data
GaussianNB: count data
"""

import numpy as np



# BernulliNp count how often every feature of each class is not 0
X = np.array([[0, 1, 0, 1],
			  [1, 0, 1, 1],
			  [0, 0, 0, 1],
			  [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
	# iterate over each class
	# count(sum) antries of 1 per feature
	counts[label] = X[y == label].sum(axis=0)
	print("Features counts:\n{}".format(counts))
	