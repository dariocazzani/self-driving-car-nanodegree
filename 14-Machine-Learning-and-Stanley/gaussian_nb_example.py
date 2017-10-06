import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # Features
Y = np.array([1, 1, 1, 2, 2, 2]) # Labels
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
new_point = [[-0.8, -1]]
print(clf.predict(new_point))
