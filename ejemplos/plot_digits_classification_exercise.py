"""
================================
Digits Classification Exercise
================================

A tutorial exercise regarding the use of classification techniques on
the Digits dataset.

This exercise is used in the :ref:`clf_tut` part of the
:ref:`supervised_learning_tut` section of the
:ref:`stat_learn_tut_index`.
"""
print(__doc__)

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

#Algoritmo K-Nearest neighbors
knn = neighbors.KNeighborsClassifier()
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))

#Algoritmo REGRESION LOGISTICA
logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
