from __future__ import print_function

import sys

import numpy as np
import pymysql
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

print("**** GALGOS ****")
print("Tipo de modelo: REGRESION")
print("Objetivo: dado un galgo en una carrera, predecir su velocidad")
print("Entrada: FUTURO --> features")
print("****************")


################# FUNCIONES #################

def leerDataset(camposRelleno, sufijoTiempo, sufijoGrupoColumnas, sufijoEtiqueta):
    c = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa').cursor()
    consulta = "SELECT *" + camposRelleno + " FROM datos_desa.tb_ds" + sufijoTiempo + sufijoGrupoColumnas + sufijoEtiqueta + ";"
    c.execute(consulta)
    alist = c.fetchall()
    dataOut = np.array(alist)
    c.close()
    print(
        "\ndataset" + sufijoTiempo + sufijoGrupoColumnas + sufijoEtiqueta + "-Consulta --> " + consulta)
    print(
        "dataset" + sufijoTiempo + sufijoGrupoColumnas + sufijoEtiqueta + "-Filas = " + str(len(alist)))
    print("dataset" + sufijoTiempo + sufijoGrupoColumnas + sufijoEtiqueta + "-Shape = " + str(
        dataOut.shape[0]) + "x" + str(dataOut.shape[1]))
    print("dataset" + sufijoTiempo + sufijoGrupoColumnas + sufijoEtiqueta + "-Ejemplos:\n" + str(
        alist[0]) + "\n" + str(alist[1]))
    return dataOut


###########################################################

print("\nINICIO")

print("Numero de parametros de entrada:", len(sys.argv))
print("Parametros de entrada --> ", str(sys.argv))
sufijoEtiqueta = str(sys.argv[1])

######################################
X = leerDataset(", 0 AS relleno", "_futuro", "_features", sufijoEtiqueta)
print("Shape de la matriz X =" + str(X.shape[0]) + "x" + str(X.shape[1]))
print("Primera fila de X: " + str(X[0]))

#########################################
print("Missing values: cambiamos los NULL por otro valor...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
X_conpadding = imp.fit(X).transform(X)
print("Shape de la matriz X_conpadding =" + str(X_conpadding.shape[0]) + "x" + str(X_conpadding.shape[1]))
print("Primera fila CON PADDING: " + str(X_conpadding[0]))

######################## PREDICCION ##############
print("Prediciendo con matriz de entrada X_conpadding...")
path_modelo = '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl'
print("Path del modelo usado = " + path_modelo)
mejor_modelo = joblib.load(path_modelo)
targets_predichos = mejor_modelo.predict(X_conpadding)

print(targets_predichos)
print("Longitud de salida targets_predichos =" + str(len(targets_predichos)))

##############################################
path_futuro_targets = "/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/datos_desa.tb_ds_futuro_targets" + sufijoEtiqueta + ".txt"
print("Guardando resultado futuro-targets en = " + path_futuro_targets)
fichero_resultados = open(path_futuro_targets, 'w')
for item in targets_predichos:
    fichero_resultados.write("%s\n" % item)

print("FIN")
