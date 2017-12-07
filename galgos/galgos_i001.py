from __future__ import print_function

import numpy as np
import pymysql
from sklearn import neighbors, linear_model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import Imputer

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

    c.execute('SELECT * from datos_desa.tb_galgos_data_pre;')
    alist = c.fetchall()
    print("Numero de filas leidas: "+str(len(alist)))
    # print("Primera fila de datos: "+str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


def leerDatasetTargetsDesdeBaseDatos():
    print("Conectando a Base de datos para descargar TARGETs...")
    con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
    c = con.cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_target_pre;')
    alist = c.fetchall()
    print("Numero de filas leidas: " + str(len(alist)))
    #print("Primera fila de target: " + str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


print("INICIO")
X=leerDatasetDatosDesdeBaseDatos()
print("Shape de la matriz X =" + str(X.shape[0])+ "x"+ str(X.shape[1]))
#print("Primera fila de X: "+str(X[0]))

############################################
print("Missing values: cambiamos los NULL por otro valor...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
X_conpadding=imp.fit(X).transform(X)
print("Shape de la matriz X_conpadding =" + str(X_conpadding.shape[0])+ "x"+ str(X_conpadding.shape[1]))
#print("Primera fila de X_conpadding: "+str(X_conpadding[0]))


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

##################
print('\n ------ Algoritmo K-Nearest Neighbors ------ ')
knn = neighbors.KNeighborsClassifier()
knn_score = knn.fit(X_train, y_train).score(X_test, y_test) * 100.0
print('Score: %f' % knn_score)
# Obtención de matriz de confusión
knn_y_train_pred = knn.predict(X_train)
knn_y_test_pred = knn.predict(X_test)
knn_confusion_matrix_train = confusion_matrix(y_train, knn_y_train_pred)
knn_confusion_matrix_test = confusion_matrix(y_test, knn_y_test_pred)
# print('Matriz de confusión para train es:')
# print(knn_confusion_matrix_train / sum(knn_confusion_matrix_train))
# print('Matriz de confusión para test es:')
# print(knn_confusion_matrix_test / sum(knn_confusion_matrix_test))
print('Precisión:', accuracy_score(y_test, knn_y_test_pred))
print('Exactitud:', precision_score(y_test, knn_y_test_pred))
print('Exhaustividad:', recall_score(y_test, knn_y_test_pred))

false_positive_rate, recall, thresholds = roc_curve(y_test, knn_y_test_pred)
knn_roc_auc = auc(false_positive_rate, recall)
print('AUC (area bajo curva ROC) = %0.2f' % knn_roc_auc)

knn_modeloGuardado = joblib.dump(knn,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_knn.pkl')

########################
print('\n\n ------ Algoritmo REGRESION LOGISTICA ------ ')
logistic = linear_model.LogisticRegression()
logistic_score = logistic.fit(X_train, y_train).score(X_test, y_test) * 100.0
print('LogisticRegression coeficientes (pesos) de cada feature: ', logistic.coef_)
print('LogisticRegression score: %f' % logistic_score)
# Obtención de matriz de confusión
logistic_y_train_pred = logistic.predict(X_train)
logistic_y_test_pred = logistic.predict(X_test)
logistic_confusion_matrix_train = confusion_matrix(y_train, logistic_y_train_pred)
logistic_confusion_matrix_test = confusion_matrix(y_test, logistic_y_test_pred)
# print('REG_LOG Matriz de confusión para train es:')
# print(logistic_confusion_matrix_train / sum(logistic_confusion_matrix_train))
# print('REG_LOG Matriz de confusión para test es:')
# print(logistic_confusion_matrix_test / sum(logistic_confusion_matrix_test))
print('Precisión:', accuracy_score(y_test, logistic_y_test_pred))
print('Exactitud:', precision_score(y_test, logistic_y_test_pred))
print('Exhaustividad:', recall_score(y_test, logistic_y_test_pred))

false_positive_rate, recall, thresholds = roc_curve(y_test, logistic_y_test_pred)
logistic_roc_auc = auc(false_positive_rate, recall)
print('AUC (area bajo curva ROC) = %0.2f' % logistic_roc_auc)

logistic_modeloGuardado = joblib.dump(logistic,
                                      '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/i001_logistic.pkl')

#############################################
############################################

print("\n\n -------- Guardando modelo GANADOR...  --------")
if (knn_score >= logistic_score):
    print("Gana modelo K-Nearest Neighbors con score=%f" % knn_score)
    modeloGuardado = joblib.dump(knn,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')
else:
    print("Gana modelo REGRESION LOGISTICA con score=%f" % logistic_score)
    modeloGuardado = joblib.dump(logistic,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')


print("FIN")

