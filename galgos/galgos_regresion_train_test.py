from __future__ import print_function

import sys

import numpy as np

import pymysql
from sklearn.externals import joblib
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
    print("dataset" + sufijoTiempo + sufijoTipoDs + sufijoGrupoColumnas + sufijoEtiqueta + "-Ejemplos:\n" + str(
        alist[0]) + "\n" + str(alist[1]))
    return dataOut


###########################################################

print("\n\nTRAIN y TEST: INICIO")

print("Numero de parametros de entrada:", len(sys.argv))
print("Parametros de entrada --> ", str(sys.argv))
sufijoEtiqueta = str(sys.argv[1])

X_train_features = leerDataset(", 0 AS relleno", "_pasado", "_train", "_features", sufijoEtiqueta)
Y_train_targets = leerDataset("", "_pasado", "_train", "_targets", sufijoEtiqueta)
X_test_features = leerDataset(", 0 AS relleno", "_pasado", "_test", "_features", sufijoEtiqueta)
Y_test_targets = leerDataset("", "_pasado", "_test", "_targets", sufijoEtiqueta)
X_validation_features = leerDataset(", 0 AS relleno", "_pasado", "_validation", "_features", sufijoEtiqueta)
Y_validation_targets = leerDataset("", "_pasado", "_validation", "_targets", sufijoEtiqueta)

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
print("Primeras filas de X_train_features_sinnulos: \n" + str(X_train_features_sinnulos[0]) + "\n" + str(
    X_train_features_sinnulos[1]))

X_test_features_sinnulos = X_juntos_features_sinnulos[
                           int(num_train):(num_train + num_test)]  # el ultimo indice es excluyente
print("\nShape (test CON padding) = " + str(X_test_features_sinnulos.shape[0]) + "x" + str(
    X_test_features_sinnulos.shape[1]))
print("Primeras filas de X_test_features_sinnulos: \n" + str(X_test_features_sinnulos[0]) + "\n" + str(
    X_test_features_sinnulos[1]))

X_validation_features_sinnulos = X_juntos_features_sinnulos[
                                 int(num_train + num_test):(
                                     num_train + num_test + num_validation)]  # el ultimo indice es excluyente
print("\nShape (X_validation_features_sinnulos) = " + str(X_validation_features_sinnulos.shape[0]) + "x" + str(
    X_validation_features_sinnulos.shape[1]))
print("Primeras filas de X_validation_features_sinnulos: \n" + str(X_validation_features_sinnulos[0]) + "\n" + str(
    X_validation_features_sinnulos[1]))


##################### Datasets preparados para usar ###########
# X_train_features_sinnulos
# Y_train_targets
# X_test_features_sinnulos
# Y_test_targets
# X_validation_features_sinnulos
# Y_validation_targets

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

############# Linear regression ##############################################
print("\n\n****** REGRESION - Linear ******")
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

modeloLinear = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(modeloLinear)

# Train the model using the training sets
modeloLinearEntrenado = modeloLinear.fit(X_train_features_sinnulos, Y_train_targets)
y_pred_linear = modeloLinearEntrenado.predict(X_test_features_sinnulos)
r2_score_linear = r2_score(Y_test_targets, y_pred_linear)
print(modeloLinearEntrenado)

print('Linear --> Coeficientes: \n')
print(*modeloLinearEntrenado.coef_, sep=', ')
print("Linear --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_linear))
print('Linear -->R^2 (Variance score) = %f' % r2_score_linear)  # si prediccion fuera ideal tomaria valor 1

############# Ridge ##############################################
print("\n\n****** REGRESION - RidgeCV ******")
# He probado las combinaciones y el mejor alpha es 13
modeloRidgeCV = linear_model.RidgeCV(alphas=(13.0, 14.0),
                                     fit_intercept=True, normalize=True, scoring=None, cv=None, gcv_mode='auto',
                                     store_cv_values=False)
modeloRidgeCVEntrenado = modeloRidgeCV.fit(X_train_features_sinnulos, Y_train_targets)
y_pred_ridgecv = modeloLinearEntrenado.predict(X_test_features_sinnulos)
r2_score_ridgecv = r2_score(Y_test_targets, y_pred_ridgecv)
print(modeloRidgeCVEntrenado)

# print('RidgeCV --> Cross-validation values para cada alpha: \n', modeloRidgeCVEntrenado.cv_values_)
print('RidgeCV --> Coeficientes: \n')
print(*modeloRidgeCVEntrenado.coef_, sep=', ')
print("RidgeCV --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_ridgecv))
print('RidgeCV --> R^2 (Variance score) = %f' % r2_score_ridgecv)  # si prediccion fuera ideal tomaria valor 1
print("RidgeCV --> Alfa: ", modeloRidgeCVEntrenado.alpha_)
print("RidgeCV --> Intercept: ", modeloRidgeCVEntrenado.intercept_)


########### Lasso ##################################################
print("\n\n****** REGRESION - Lasso ******")
from sklearn.linear_model import Lasso

modeloLasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False,
                    precompute=False, random_state=None, selection='cyclic', tol=0.001, warm_start=False)
print(modeloLasso)

modeloLassoEntrenado = modeloLasso.fit(X_train_features_sinnulos, Y_train_targets)
y_pred_lasso = modeloLassoEntrenado.predict(X_test_features_sinnulos)
r2_score_lasso = r2_score(Y_test_targets, y_pred_lasso)

print('Lasso --> Coeficientes: \n')
print(*modeloLassoEntrenado.coef_, sep=', ')
print("Lasso --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_lasso))
print("Lasso --> R^2 (sobre DS-test) = %f" % r2_score_lasso)

########### LassoLarsCV ##################################################
print("\n\n****** REGRESION - LassoLarsCV ******")
from sklearn.linear_model import LassoLarsCV

modeloLassoLarsCV = LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto',
                                cv=None, max_n_alphas=1000, n_jobs=-1, eps=2.2204460492503131e-16, copy_X=True,
                                positive=False)
print(modeloLassoLarsCV)

modeloLassoLarsCVEntrenado = modeloLassoLarsCV.fit(X_train_features_sinnulos, Y_train_targets)
y_pred_lassoLarsCV = modeloLassoEntrenado.predict(X_test_features_sinnulos)
r2_score_lassoLarsCV = r2_score(Y_test_targets, y_pred_lassoLarsCV)

print('LassoLarsCV --> Coeficientes: \n')
print(*modeloLassoLarsCVEntrenado.coef_, sep=', ')
print('LassoLarsCV --> Alpha: \n', modeloLassoLarsCVEntrenado.alpha_)
print("LassoLarsCV --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_lassoLarsCV))
print("LassoLarsCV --> R^2 (sobre DS-test) = %f" % r2_score_lassoLarsCV)

################## ElasticNet #################################################
print("\n\n****** REGRESION - ElasticNet ******")
from sklearn.linear_model import ElasticNet

modeloElasticNet = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.7, max_iter=1000, normalize=False,
                              positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.001,
                              warm_start=False)
print(modeloElasticNet)

modeloElasticNetEntrenado = modeloElasticNet.fit(X_train_features_sinnulos, Y_train_targets)
y_pred_enet = modeloElasticNetEntrenado.predict(X_test_features_sinnulos)
r2_score_elasticnet = r2_score(Y_test_targets, y_pred_enet)

print('ElasticNet --> Coeficientes: \n')
print(*modeloElasticNetEntrenado.coef_, sep=', ')
print("ElasticNet --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_enet))
print("ElasticNet --> R^2 (sobre DS-test) = %f" % r2_score_elasticnet)

######## RANSAC ######################
modeloRansac = linear_model.RANSACRegressor()
# modeloRansacEntrenado = modeloRansac.fit(X_train_features_sinnulos, Y_train_targets)
# inlier_mask = modeloRansacEntrenado.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
# y_pred_ransac = modeloRansacEntrenado.predict(X_test_features_sinnulos)
# r2_score_ransac = r2_score(Y_test_targets, y_pred_ransac)

# print('RANSAC --> Coeficientes: \n')
# print(*modeloRansacEntrenado.coef_, sep=', ')
# print("RANSAC --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_ransac))
# print("RANSAC --> R^2 (sobre DS-test) = %f" % r2_score_ransac)


################## LogisticRegression #################################################
# print("\n\n****** REGRESION - LogisticRegression ******")

# modeloLogReg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
# print(modeloLogReg)

# modeloLogRegEntrenado = modeloLogReg.fit(X_train_features_sinnulos, Y_train_targets.ravel())
# y_pred_logreg = modeloLogRegEntrenado.predict(X_test_features_sinnulos)
# r2_score_logreg = r2_score(Y_test_targets, y_pred_enet)

# print('LogisticRegression --> Coeficientes: \n')
# print(*modeloLogRegEntrenado.coef_, sep=', ')
# print("LogisticRegression --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_logreg))
# print("LogisticRegression --> R^2 (sobre DS-test) = %f" % r2_score_logreg)


################## SVR #################################################
# print("\n\n****** REGRESION - SVR ******")
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)


# svr_rbf_entrenado=svr_rbf.fit(X_train_features_sinnulos, Y_train_targets.ravel())
# svr_lin_entrenado=svr_lin.fit(X_train_features_sinnulos, Y_train_targets.ravel())
# svr_poly_entrenado=svr_poly.fit(X_train_features_sinnulos, Y_train_targets.ravel())

# print("SVR-RBF --> " + svr_rbf_entrenado)
# print("SVR-LIN --> " + svr_lin_entrenado)
# print("SVR-POLY --> " + svr_poly_entrenado)

# y_pred_rbf = svr_rbf_entrenado.predict(X_test_features_sinnulos)
# y_pred_lin = svr_lin_entrenado.predict(X_test_features_sinnulos)
# y_pred_poly = svr_poly_entrenado.predict(X_test_features_sinnulos)

# r2_score_rbf = r2_score(Y_test_targets, y_pred_rbf)
# r2_score_lin = r2_score(Y_test_targets, y_pred_lin)
# r2_score_poly = r2_score(Y_test_targets, y_pred_poly)

# print('SVR-RBF --> Coeficientes: \n')
# print(*svr_rbf_entrenado.coef_, sep=', ')
# print("SVR-RBF --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_rbf))
# print("SVR-RBF --> R^2 (sobre DS-test) = %f" % r2_score_rbf)

# print('SVR-LIN --> Coeficientes: \n')
# print(*svr_lin_entrenado.coef_, sep=', ')
# print("SVR-LIN --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_lin))
# print("SVR-LIN --> R^2 (sobre DS-test) = %f" % r2_score_lin)

# print('SVR-POLY --> Coeficientes: \n')
# print(*svr_poly_entrenado.coef_, sep=', ')
# print("SVR-POLY --> Error cuadrático medio = %f" % mean_squared_error(Y_test_targets, y_pred_poly))
#print("SVR-POLY --> R^2 (sobre DS-test) = %f" % r2_score_poly)


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

    fichero_resultados = open(validation_targets_path, 'w')
    for item in pasado_validation_targets_predichos:
        fichero_resultados.write(str(item[0]) + "\n")

    return pasado_validation_targets_predichos


##############################################
############################################


########################
print(
    "Como no tengo ni idea de ESTADISTICA (estoy estudiando en papel), he elegido RIDGE en una primera prueba, pero deberia elegir el modelo con mayor R^2...")
# predecirValidationTargets(modeloLinearEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
predecirValidationTargets(modeloRidgeCVEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
# predecirValidationTargets(modeloLassoEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
# predecirValidationTargets(modeloLassoLarsCVEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
#predecirValidationTargets(modeloElasticNetEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)

print("Guardando modelo ganador (RIDGE de momento)...")
joblib.dump(modeloRidgeCVEntrenado,
            '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
########################


# if (r2_score_lasso >= r2_score_elasticnet ):
#    print("Gana modelo Lasso")
#    joblib.dump(modeloLassoEntrenado,
#                                 '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
#    predecirValidationTargets(modeloLassoEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)
# elif(r2_score_linear ...):
#
# else:
#    print("Gana modelo ElasticNet")
#    joblib.dump(modeloElasticNetEntrenado,
#                                 '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
#    predecirValidationTargets(modeloElasticNetEntrenado, X_validation_features_sinnulos, sufijoEtiqueta)



print("\nTRAIN y TEST: FIN")
