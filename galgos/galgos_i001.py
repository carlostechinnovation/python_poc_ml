from __future__ import print_function
import numpy as np
from sklearn.preprocessing import Imputer
import pymysql
from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib

print("GALGOS - Informe 001")
print("------------------")
print("Tipo de modelo: clasificacion")
print("Objetivo: dado un galgo en una carrera, quiero clasificarle en 2 grupos: 1o2 o 3a6")
print("Entrada: features de la carrera, features del galgo analizado, features de una galgo agregado (galgos competidores agrupados) y target para train/test.")
print("Categorias (classes): 1o2 (valor 1) y 3a6 (valor 0)")
print("------------------")


def leerDatasetDatosDesdeBaseDatos():
    print("Conectando a Base de datos para descargar DATOS...")
    con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
    c = con.cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_data_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: "+str(len(alist)))
    #print("Primera fila de datos: "+str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


def leerDatasetTargetsDesdeBaseDatos():
    print("Conectando a Base de datos para descargar TARGETs...")
    con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
    c = con.cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_target_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: " + str(len(alist)))
    #print("Primera fila de target: " + str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


print("INICIO")
X=leerDatasetDatosDesdeBaseDatos()

print("Missing values: cambiamos los NULL por valor medio...")
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_conpadding=imp.fit(X).transform(X)
#print("Primera fila CON PADDING: "+str(X_conpadding[0]))

print("Datasets: train y test...")
n_sample = len(X_conpadding)
X_train = X_conpadding[:int(.9 * n_sample)]
X_test = X_conpadding[int(.9 * n_sample):]

Y=leerDatasetTargetsDesdeBaseDatos()
y_train = Y[:int(.9 * n_sample)]
y_test = Y[int(.9 * n_sample):]


########### Modelos de CLASIFICACION #########################################
print("Modelos de CLASIFICACION...")

print("X_train: "+str(X_train.shape))
print("y_train: "+str(y_train.shape))

y_train=y_train.ravel()
print("y_train (reshaped): "+str(y_train.shape))

#Algoritmo K-Nearest neighbors
knn = neighbors.KNeighborsClassifier()
knn_score=knn.fit(X_train, y_train).score(X_test, y_test)
print('KNN score: %f' % knn_score)

#Algoritmo REGRESION LOGISTICA
logistic = linear_model.LogisticRegression()
logistic_score=logistic.fit(X_train, y_train).score(X_test, y_test)
print('LogisticRegression score: %f' % logistic_score)


print("Guardando modelos...")
modeloGuardado = joblib.dump(knn, '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_knn.pkl')
modeloGuardado = joblib.dump(logistic, '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/i001_logistic.pkl')

print("Guardando modelo GANADOR...")
if knn_score>=logistic_score:
   print("Gana modelo KNN")
   modeloGuardado = joblib.dump(knn,'/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')
else:
   print("Gana modelo LOGISTIC")
   modeloGuardado = joblib.dump(logistic,'/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')


print("FIN")

