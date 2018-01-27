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


################# FUNCIONES #################

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
    print("dataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Ejemplo:\n" + str(
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
X_validation_features = leerDataset(", 0 AS relleno", "_pasado", "_validation", "_features", sufijoEtiqueta)
X_validation_targets = leerDataset("", "_pasado", "_validation", "_targets", sufijoEtiqueta)

############################################
print(
    "\n\n*** Missing values ***\nCambiamos los NULL por otro valor (solo afecta a los FEATURES, pero no a los targets)...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

# Numero de filas (no columnas) de cada dataset
num_train = X_train_features.shape[0]
num_test = X_test_features.shape[0]
num_validation = X_validation_features.shape[0]

# Juntar TRAIN+TEST+VALIDATION
X_juntos_features_aux = np.concatenate((X_train_features, X_test_features), axis=0)
X_juntos_features = np.concatenate((X_juntos_features_aux, X_validation_features), axis=0)
print("\nShape (JUNTOS SIN padding, pero con columna de RELLENO que la va a quitar el Imputer) = "
      + str(X_juntos_features.shape[0]) + "x" + str(X_juntos_features.shape[1]))

# Rellenar los NULL con el valor de la MEDIANA de cada columna
X_juntos_features_sinnulos = imp.fit(X_juntos_features).transform(X_juntos_features)
print("\nShape (juntos CON padding) = " + str(X_juntos_features_sinnulos.shape[0]) + "x" + str(
    X_juntos_features_sinnulos.shape[1]))

# Separar otra vez las FEATURES de TRAIN y TEST, pero ya tienen hecho el relleno
X_train_features_sinnulos = X_juntos_features_sinnulos[0:int(num_train)]  # el ultimo indice es excluyente
print("\nShape (train CON padding) = " + str(X_train_features_sinnulos.shape[0]) + "x" + str(
    X_train_features_sinnulos.shape[1]))
print("Primera fila de X_train_features_sinnulos: " + str(X_train_features_sinnulos[0]))

X_test_features_sinnulos = X_juntos_features_sinnulos[
                           int(num_train):(num_train + num_test)]  # el ultimo indice es excluyente
print("\nShape (test CON padding) = " + str(X_test_features_sinnulos.shape[0]) + "x" + str(
    X_test_features_sinnulos.shape[1]))
print("Primera fila de X_test_features_sinnulos: " + str(X_test_features_sinnulos[0]))

X_validation_features_sinnulos = X_juntos_features_sinnulos[
                                 int(num_train + num_test):(
                                     num_train + num_test + num_validation)]  # el ultimo indice es excluyente
print("\nShape (X_validation_features_sinnulos) = " + str(X_validation_features_sinnulos.shape[0]) + "x" + str(
    X_validation_features_sinnulos.shape[1]))
print("Primera fila de X_test_features_sinnulos: " + str(X_validation_features_sinnulos))


##################### Datasets preparados para usar ###########
# X_train_features_sinnulos
# X_train_targets
# X_test_features_sinnulos
# X_test_targets
# X_validation_features_sinnulos
# X_validation_targets

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
print("\n*********************\nMACHINE LEARNING: REGRESIONES\n*********************\n")
#####################################

########### Lasso ##################################################
from sklearn.linear_model import Lasso

modeloLasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                    normalize=False, positive=False, precompute=False, random_state=None,
                    selection='cyclic', tol=0.0001, warm_start=False)
print(modeloLasso)

modeloLassoEntrenado = modeloLasso.fit(X_train_features_sinnulos, X_train_targets)
y_pred_lasso = modeloLassoEntrenado.predict(X_test_features_sinnulos)
r2_score_lasso = r2_score(X_test_targets, y_pred_lasso)

print("\n\nREGRESION - Lasso --> R^2 (sobre DS-test) = %f" % r2_score_lasso)
print('\n\n')

# ####### ElasticNet #################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

modeloElasticNet = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.7,
                              max_iter=1000, normalize=False, positive=False, precompute=False,
                              random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

print(modeloElasticNet)

modeloElasticNetEntrenado = modeloElasticNet.fit(X_train_features_sinnulos, X_train_targets)
y_pred_enet = modeloElasticNetEntrenado.predict(X_test_features_sinnulos)
r2_score_elasticnet = r2_score(X_test_targets, y_pred_enet)

print("\n\nREGRESION - ElasticNet --> R^2 (sobre DS-test) = %f" % r2_score_elasticnet)
print('\n\n')


#######################################################
# MODELO + Prediccion sobre DATASET-PASADO-VALIDATION
#######################################################
# X_validation_features_sinnulos
# X_validation_targets

def predecirValidationTargets(modelo, matriz_features, sufijoEtiqueta):
    # PREDICCION de los targets sobre DS-pasado-validation
    print("Prediciendo targets de validation...")
    validation_targets_path = "/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/pasado_validation_targets_predichos" + sufijoEtiqueta + ".txt"
    print("path --> " + validation_targets_path)
    pasado_validation_targets_predichos = modelo.predict(matriz_features)
    print("pasado_validation_targets_predichos:\n")
    print(pasado_validation_targets_predichos)
    print("Longitud de salida targets_validation (PREDICHOS) =" + str(len(pasado_validation_targets_predichos)))
    print("Primera fila de pasado_validation_targets_predichos: " + str(pasado_validation_targets_predichos[0]))

    fichero_resultados = open(validation_targets_path, 'w')
    for item in pasado_validation_targets_predichos:
        fichero_resultados.write("%s\n" % item)

    return pasado_validation_targets_predichos


##############################################
############################################

print("Guardando modelos...")
joblib.dump(modeloLassoEntrenado,
                             '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_modeloLasso.pkl')
joblib.dump(modeloElasticNetEntrenado,
                             '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_elasticnet.pkl')


print("Guardando modelo GANADOR...")
if (r2_score_lasso >= r2_score_elasticnet):
    print("Gana modelo Lasso")
    joblib.dump(modeloLassoEntrenado,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
    predecirValidationTargets(modeloLassoEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
else:
    print("Gana modelo ElasticNet")
    joblib.dump(modeloElasticNetEntrenado,
                                 '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
    predecirValidationTargets(modeloElasticNetEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)



print("FIN")
