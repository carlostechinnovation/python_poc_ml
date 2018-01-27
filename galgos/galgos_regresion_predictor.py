import numpy as np
import pymysql
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

print("**** GALGOS ****")
print("Tipo de modelo: REGRESION")
print("Objetivo: dado un galgo en una carrera, predecir su velocidad")
print("Entrada: FUTURO --> features")
print("****************")


def leerFeaturesDelCasoAPredecirDesdeBaseDatos():
    print("Conectando a Base de datos para descargar FEATURES DE CASOS A PREDECIR...")
    con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
    c = con.cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_prediccion_features_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: " + str(len(alist)))
    # print("Primera fila de datos: "+str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1


print("INICIO")

mejor_modelo = joblib.load(
    '/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')

######################################3
X = leerFeaturesDelCasoAPredecirDesdeBaseDatos()
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
targets_predichos = mejor_modelo.predict(X_conpadding)

print(targets_predichos)
print("Longitud de salida targets_predichos =" + str(len(targets_predichos)))

##############################################
fichero_resultados = open('/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/i001_targets.txt', 'w')
for item in targets_predichos:
    fichero_resultados.write("%s\n" % item)

print("FIN")
