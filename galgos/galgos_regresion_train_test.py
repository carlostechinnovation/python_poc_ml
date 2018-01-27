from __future__ import print_function

import sys

import numpy as np
import pymysql
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer

print("**** GALGOS ****")
print("Tipo de modelo: REGRESION")
print("Objetivo: dado un galgo en una carrera, predecir su velocidad")
print("Entrada: PASADO --> features y targets, de train y test.")
print("****************")


def leerDataset(camposRelleno, sufijoTiempo, sufijoTipoDs, sufijoGrupoColumnas, sufijoEtiqueta):
    c = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa').cursor()
    consulta = "SELECT *" + camposRelleno + " FROM datos_desa.tb_ds" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + ";"
    c.execute(consulta)
    alist = c.fetchall()
    dataOut = np.array(alist)
    c.close()
    print(
        "\ndataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Consulta --> " + consulta)
    print(
        "dataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Filas = " + str(len(alist)))
    print("dataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Shape = " + str(
        dataOut.shape[0]) + "x" + str(dataOut.shape[1]))
    print("dataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Ejemplo --> " + str(
        alist[0]))
    return dataOut


###########################################################

print("\nINICIO")

print("Numero de parametros de entrada:", len(sys.argv))
print("Parametros de entrada --> ", str(sys.argv))
sufijoEtiqueta = str(sys.argv[1])

X_train_features = leerDataset(", 0 AS relleno", "_pasado", "_train", "_features", sufijoEtiqueta)
X_train_targets = leerDataset("", "_pasado", "_train", "_targets", sufijoEtiqueta)
X_test_features = leerDataset(", 0 AS relleno", "_pasado", "_test", "_features", sufijoEtiqueta)
X_test_targets = leerDataset("", "_pasado", "_test", "_targets", sufijoEtiqueta)

############################################
print(
    "\n\n*** Missing values ***\nCambiamos los NULL por otro valor (solo afecta a los FEATURES, pero no a los targets)...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

# Numero de filas (no columnas) de cada dataset
num_train = X_train_features.shape[0]
num_test = X_test_features.shape[0]

# Juntar TRAIN+TEST
X_trainYtest_features = np.concatenate((X_train_features, X_test_features), axis=0)
print("\nShape (train+test SIN padding, pero con columna de RELLENO que la va a quitar el Imputer) = " + str(
    X_trainYtest_features.shape[0]) + "x" + str(X_trainYtest_features.shape[1]))

# Rellenar los NULL con el valor de la MEDIANA de cada columna
X_trainYtest_features_sinnulos = imp.fit(X_trainYtest_features).transform(X_trainYtest_features)
print("\nShape (train+test CON padding) = " + str(X_trainYtest_features_sinnulos.shape[0]) + "x" + str(
    X_trainYtest_features_sinnulos.shape[1]))
print("Primera fila de X_conpadding: " + str(X_trainYtest_features_sinnulos[0]))

# Separar otra vez las FEATURES de TRAIN y TEST, pero ya tienen hecho el relleno
X_train_features_sinnulos = X_trainYtest_features_sinnulos[0:int(num_train)]  # el ultimo indice es excluyente
print("\nShape (train CON padding) = " + str(X_train_features_sinnulos.shape[0]) + "x" + str(
    X_train_features_sinnulos.shape[1]))
print("Primera fila de X_train_features_sinnulos: " + str(X_train_features_sinnulos[0]))

X_test_features_sinnulos = X_trainYtest_features_sinnulos[
                           int(num_train):(num_train + num_test)]  # el ultimo indice es excluyente
print("\nShape (test CON padding) = " + str(X_test_features_sinnulos.shape[0]) + "x" + str(
    X_test_features_sinnulos.shape[1]))
print("Primera fila de X_test_features_sinnulos: " + str(X_test_features_sinnulos[0]))

##################### Datasets preparados para usar ###########
# X_train_features_sinnulos
# X_train_targets
# X_test_features_sinnulos
# X_test_targets

################### Comprobacion de: Input contains NaN, infinity or a value too large for dtype('float64'). ########
print("checkInvalidos_train_features_sinnulos --> ")
checkInvalidos_train_features_sinnulos = np.isnan(X_train_features_sinnulos)
# print("checkInvalidos_X_train_targets --> ")
# checkInvalidos_X_train_targets=np.isnan(X_train_targets)
print("checkInvalidos_test_features_sinnulos --> ")
checkInvalidos_test_features_sinnulos = np.isnan(X_test_features_sinnulos)
# print("checkInvalidos_X_test_targets --> ")
# checkInvalidos_X_test_targets=np.isnan(X_test_targets)

#####################################
print("\n\nMACHINE LEARNING: REGRESIONES")
#####################################

########### Lasso ##################################################
from sklearn.linear_model import Lasso

modeloLasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                    normalize=False, positive=False, precompute=False, random_state=None,
                    selection='cyclic', tol=0.0001, warm_start=False)
print(modeloLasso)

y_pred_lasso = modeloLasso.fit(X_train_features_sinnulos, X_train_targets).predict(X_test_features_sinnulos)
r2_score_lasso = r2_score(X_test_targets, y_pred_lasso)

print("\n\nREGRESION - Lasso --> R^2 (sobre DS-test) = %f" % r2_score_lasso)
print('\n\n')

# ####### ElasticNet #################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

modeloElasticNet = ElasticNet(alpha=0.1, l1_ratio=0.7)

y_pred_enet = modeloElasticNet.fit(X_train_features_sinnulos, X_train_targets).predict(X_test_features_sinnulos)
r2_score_elasticnet = r2_score(X_test_targets, y_pred_enet)
print(modeloElasticNet)
print("\n\nREGRESION - ElasticNet --> R^2 (sobre DS-test) = %f" % r2_score_elasticnet)
print('\n\n')

#############################################

############################################

print("Guardando modelos...")
modeloGuardado = joblib.dump(modeloLasso,
                             '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_modeloLasso.pkl')
modeloGuardado = joblib.dump(modeloElasticNet,
                             '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_elasticnet.pkl')

print("Guardando modelo GANADOR...")
if (r2_score_lasso >= r2_score_elasticnet):
    print("Gana modelo Lasso")
    modeloGuardado = joblib.dump(modeloLasso,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
else:
    print("Gana modelo ElasticNet")
    modeloGuardado = joblib.dump(modeloElasticNet,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')

print("FIN")
