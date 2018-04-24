from __future__ import print_function

import numpy as np
import pymysql
from sklearn import neighbors, linear_model
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

print("GALGOS - Informe 001")
print("------------------")
print("Tipo de modelo: clasificacion")
print("Objetivo: dado un galgo en una carrera, quiero clasificarle en 2 grupos: 1o2 o 3a6")
print(
    "Entrada: features de la carrera, features del galgo analizado, features de un galgo agregado (galgos competidores agrupados) y target para train/test.")
print("Categorias (classes): 1o2 (valor 1) y 3a6 (valor 0)")
print("------------------")


def leerDatasetDatosDesdeBaseDatos():
    print("Conectando a Base de datos para descargar DATOS...")
    c = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa').cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_data_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: "+str(len(alist)))
    print("Primera fila de datos: "+str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


def leerDatasetTargetsDesdeBaseDatos():
    print("Conectando a Base de datos para descargar TARGETs...")
    c = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa').cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_target_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: " + str(len(alist)))
    print("Primera fila de target: " + str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


print("INICIO")
X=leerDatasetDatosDesdeBaseDatos()
print("Shape de la matriz X =" + str(X.shape[0])+ "x"+ str(X.shape[1]))
print("Primera fila de X: "+str(X[0]))

############################################
print("Missing values: cambiamos los NULL por otro valor...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
X_conpadding=imp.fit(X).transform(X)
print("Shape de la matriz X_conpadding =" + str(X_conpadding.shape[0])+ "x"+ str(X_conpadding.shape[1]))
print("Primera fila de X_conpadding: "+str(X_conpadding[0]))


#####################################################
print("Datasets: train y test...")
n_sample = len(X_conpadding)
X_train = X_conpadding[:int(.9 * n_sample)]
X_test = X_conpadding[int(.9 * n_sample):]

Y=leerDatasetTargetsDesdeBaseDatos()
y_train = Y[:int(.9 * n_sample)]
y_test = Y[int(.9 * n_sample):]

print("Entrada (TEST) ...")
fichero_entrada_test = open('/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/i001_test_entrada.txt', 'w')
for item in X_test:
    fichero_entrada_test.write("%s\n" % item)


########### Modelos de CLASIFICACION #########################################
print("Modelos de CLASIFICACION...")

print("X_train: "+str(X_train.shape))
print("y_train: "+str(y_train.shape))

y_train=y_train.ravel()
print("y_train (reshaped): "+str(y_train.shape))

print('\nAlgoritmo K-Nearest Neighbors')
knn = neighbors.KNeighborsClassifier()
knn_score = knn.fit(X_train, y_train).score(X_test, y_test) * 100.0
print('KNN score: %f' % knn_score)

print("Salida (TEST) de KNN...")
targets_predichos_knn = knn.predict(X_test)
fichero_resultados_test_knn = open('/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/i001_knn_test_targets_predichos.txt',
                                   'w')
for item in targets_predichos_knn:
    fichero_resultados_test_knn.write("%s\n" % item)

print('\nAlgoritmo REGRESION LOGISTICA')
logistic = linear_model.LogisticRegression()
logistic_score = logistic.fit(X_train, y_train).score(X_test, y_test) * 100.0
print('LogisticRegression score: %f' % logistic_score)

print("Salida (TEST) de REG LOG...")
targets_predichos_reglog = logistic.predict(X_test)
fichero_resultados_test_logistic = open(
    '/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/i001_reglog_test_targets_predichos.txt', 'w')
for item in targets_predichos_reglog:
    fichero_resultados_test_logistic.write("%s\n" % item)


print('\n\n')

#############################################

############################################

print("Guardando modelos...")
modeloGuardado = joblib.dump(knn, '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_i001_knn.pkl')
modeloGuardado = joblib.dump(logistic, '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/i001_logistic.pkl')

print("Guardando modelo GANADOR...")
if (knn_score >= logistic_score):
    print("Gana modelo K-Nearest Neighbors")
    modeloGuardado = joblib.dump(knn,
                                 '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')
else:
    print("Gana modelo REGRESION LOGISTICA")
    modeloGuardado = joblib.dump(logistic,
                                 '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')


print("FIN")

