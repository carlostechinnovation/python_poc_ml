from __future__ import print_function

import sys

import numpy as np
import pymysql
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

print("**** GALGOS ****")
print("Tipo de modelo: REGRESION")
print("Objetivo: dado un galgo en una carrera, predecir su velocidad")
print("Entrada: PASADO --> TTV")
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

print("\nModelo TTV (entrenar y persistir) - FIN")

print("Numero de parametros de entrada:", len(sys.argv))
print("Parametros de entrada --> ", str(sys.argv))
sufijoEtiqueta = str(sys.argv[1])
# sufijoEtiqueta = "_TRAINER_BUENOS_GALGOS" #DEBUG

X_ttv_features = leerDataset(", 0 AS relleno", "_pasado", "_ttv", "_features", sufijoEtiqueta)
Y_ttv_targets = leerDataset("", "_pasado", "_ttv", "_targets", sufijoEtiqueta)

############################################
print(
    "\n\n*** Missing values ***\nCambiamos los NULL por otro valor (solo afecta a los FEATURES, pero no a los targets)...")
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

# Numero de filas (no columnas) de cada dataset
num_train = X_ttv_features.shape[0]

# Juntar TRAIN+TEST+VALIDATION
X_juntos_features = X_ttv_features
print("\nShape (JUNTOS SIN padding, pero con columna de RELLENO que la va a quitar el Imputer) = "
      + str(X_juntos_features.shape[0]) + "x" + str(X_juntos_features.shape[1]))

# Rellenar los NULL con el valor de la MEDIANA de cada columna
X_juntos_features_sinnulos = imp.fit(X_juntos_features).transform(X_juntos_features)
print("\nShape (juntos CON padding) = " + str(X_juntos_features_sinnulos.shape[0]) + "x" + str(
    X_juntos_features_sinnulos.shape[1]))

print("\nShape (X_juntos_features_sinnulos) = " + str(X_juntos_features_sinnulos.shape[0]) + "x" + str(
    X_juntos_features_sinnulos.shape[1]))
print("Primeras filas de X_juntos_features_sinnulos: \n" + str(X_juntos_features_sinnulos[0]) + "\n" + str(
    X_juntos_features_sinnulos[1]))

##################### Datasets preparados para usar ###########
# X_juntos_features_sinnulos
# Y_ttv_targets

################### Comprobacion de: Input contains NaN, infinity or a value too large for dtype('float64'). ########
print("checkInvalidos_ttv_features_sinnulos --> ")
checkInvalidos_train_features_sinnulos = np.isnan(X_juntos_features_sinnulos)
# print("checkInvalidos_X_train_targets --> ")
# checkInvalidos_X_train_targets=np.isnan(X_train_targets)


################################################################################################################
print("\n*********************\nMACHINE LEARNING: REGRESIONES\n*********************\n")
################################################################################################################

############# Linear regression ##############################################
print("\n\n****** REGRESION - Linear ******")
from sklearn import linear_model

modeloLinear = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(modeloLinear)

# Train the model using the training sets
modeloLinearEntrenado = modeloLinear.fit(X_juntos_features_sinnulos, Y_ttv_targets)
print('Linear --> Coeficientes: \n')
print(*modeloLinearEntrenado.coef_, sep=', ')

############# Ridge ##############################################
print("\n\n****** REGRESION - RidgeCV ******")
# He probado las combinaciones y el mejor alpha es 13
modeloRidgeCV = linear_model.RidgeCV(alphas=(13.0, 14.0),
                                     fit_intercept=True, normalize=True, scoring=None, cv=None, gcv_mode='auto',
                                     store_cv_values=False)
modeloRidgeCVEntrenado = modeloRidgeCV.fit(X_juntos_features_sinnulos, Y_ttv_targets)
print(modeloRidgeCVEntrenado)

# print('RidgeCV --> Cross-validation values para cada alpha: \n', modeloRidgeCVEntrenado.cv_values_)
print('RidgeCV --> Coeficientes: \n')
print(*modeloRidgeCVEntrenado.coef_, sep=', ')
print("RidgeCV --> Alfa: ", modeloRidgeCVEntrenado.alpha_)
print("RidgeCV --> Intercept: ", modeloRidgeCVEntrenado.intercept_)

########### Lasso ##################################################
print("\n\n****** REGRESION - Lasso ******")
from sklearn.linear_model import Lasso

modeloLasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False,
                    precompute=False, random_state=None, selection='cyclic', tol=0.001, warm_start=False)
print(modeloLasso)
modeloLassoEntrenado = modeloLasso.fit(X_juntos_features_sinnulos, Y_ttv_targets)

print('Lasso --> Coeficientes: \n')
print(*modeloLassoEntrenado.coef_, sep=', ')

########### LassoLarsCV ##################################################
print("\n\n****** REGRESION - LassoLarsCV ******")
from sklearn.linear_model import LassoLarsCV

modeloLassoLarsCV = LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto',
                                cv=None, max_n_alphas=1000, n_jobs=-1, eps=2.2204460492503131e-16, copy_X=True,
                                positive=False)
print(modeloLassoLarsCV)
modeloLassoLarsCVEntrenado = modeloLassoLarsCV.fit(X_juntos_features_sinnulos, Y_ttv_targets)

print('LassoLarsCV --> Coeficientes: \n')
print(*modeloLassoLarsCVEntrenado.coef_, sep=', ')
print('LassoLarsCV --> Alpha: \n', modeloLassoLarsCVEntrenado.alpha_)

################## ElasticNet #################################################
print("\n\n****** REGRESION - ElasticNet ******")
from sklearn.linear_model import ElasticNet

modeloElasticNet = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.7, max_iter=1000, normalize=False,
                              positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.001,
                              warm_start=False)
print(modeloElasticNet)
modeloElasticNetEntrenado = modeloElasticNet.fit(X_juntos_features_sinnulos, Y_ttv_targets)

print('ElasticNet --> Coeficientes: \n')
print(*modeloElasticNetEntrenado.coef_, sep=', ')

######## RANSAC ######################
modeloRansac = linear_model.RANSACRegressor()
# modeloRansacEntrenado = modeloRansac.fit(X_juntos_features_sinnulos, Y_ttv_targets)
# inlier_mask = modeloRansacEntrenado.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# print('RANSAC --> Coeficientes: \n')
# print(*modeloRansacEntrenado.coef_, sep=', ')

################## LogisticRegression #################################################
# print("\n\n****** REGRESION - LogisticRegression ******")

# modeloLogReg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
# print(modeloLogReg)
# modeloLogRegEntrenado = modeloLogReg.fit(X_juntos_features_sinnulos, Y_ttv_targets.ravel())

# print('LogisticRegression --> Coeficientes: \n')
# print(*modeloLogRegEntrenado.coef_, sep=', ')

################## SVR #################################################
# print("\n\n****** REGRESION - SVR ******")
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

# svr_rbf_entrenado=svr_rbf.fit(X_juntos_features_sinnulos, Y_ttv_targets.ravel())
# svr_lin_entrenado=svr_lin.fit(X_juntos_features_sinnulos, Y_ttv_targets.ravel())
# svr_poly_entrenado=svr_poly.fit(X_juntos_features_sinnulos, Y_ttv_targets.ravel())

# print("SVR-RBF --> " + svr_rbf_entrenado)
# print("SVR-LIN --> " + svr_lin_entrenado)
# print("SVR-POLY --> " + svr_poly_entrenado)

# print('SVR-RBF --> Coeficientes: \n')
# print(*svr_rbf_entrenado.coef_, sep=', ')

# print('SVR-LIN --> Coeficientes: \n')
# print(*svr_lin_entrenado.coef_, sep=', ')

# print('SVR-POLY --> Coeficientes: \n')
# print(*svr_poly_entrenado.coef_, sep=', ')


########################
print(
    "Como no tengo ni idea de ESTADISTICA (estoy estudiando en papel), he elegido RIDGE en una primera prueba, pero deberia elegir el modelo con mayor R^2...")

print("Guardando modelo ganador (RIDGE de momento)...")
joblib.dump(modeloRidgeCVEntrenado,
            '/home/carloslinux/Desktop/WORKSPACES/wksp_pycharm/python_poc_ml/galgos/galgos_regresion_MEJOR_MODELO.pkl')
########################

print("\nModelo TTV (entrenar y persistir) - FIN")
