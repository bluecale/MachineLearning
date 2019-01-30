from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# setup dataset
iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_train)

# predict new entry
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Prediction target name: {}".format(iris_dataset['target_names'][prediction]))

# evaluate accuracy
y_pred = knn.predict(X_test)
print("Predictions: {}".format(y_pred))
print("Test score: {:.2f}".format(np.mean(y_pred==Y_test)))