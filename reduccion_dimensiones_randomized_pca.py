import numpy as np
from sklearn.decomposition import RandomizedPCA

#http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.decomposition.RandomizedPCA.html
print "---- Reduccion de dimensiones - RandomizedPCA"
print "Permite fit() y transform(). En nuestro caso lo queremos para reducir el numero de features de la matriz de entrada, extrayendo solo las mas relevantes"
X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])
pca = RandomizedPCA(n_components=2)

print "Parametros:"
print(pca.get_params())

print "Fit y Transform (apply dimensionality reduction on input matrix)..."
pca.fit_transform(X)

print "Components with maximum variance:"
print(pca.components_)

print "Percentage of variance explained by each of the selected components. If k is not set then all components are stored and the sum of explained variances is equal to 1.0 :"
print(pca.explained_variance_ratio_)

print "Per-feature empirical mean, estimated from the training set:"
print(pca.mean_)

print "FIN"