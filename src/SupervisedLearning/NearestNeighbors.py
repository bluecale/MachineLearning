"""
Testing accuracy for k-NearestNeaighbors and visualizing the complexity
with different n_neighbors numbers
"""

import mglearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 3 neighbors accuracy
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5)
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

print("Test set prediction: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10,3))

# visualizing different n_neighbors areas
for n_neighbors, ax in zip([1, 3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title("{} neighbor(s)".format(n_neighbors))
	ax.set_xlabel("feature 0")
	ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

# training and test accuracy for different n_neighbors number, using cancer dataset
from sklearn import load_breast_cancer

cancer = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split(
	cancer.data, cancer.target, startify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# n_neighbors tested
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
	# build the model
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)
	# record training set accuracy
	training_accuracy.append(clf.score(X_train, y_train))
	# record generalization accuracy
	test_accuracy.append(clf.score(X_test, y_test))
	
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n-Neighbors")
plt.legend()
plt.show()
	

