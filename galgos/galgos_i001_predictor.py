from sklearn.externals import joblib
import pymysql
import numpy as np


print("GALGOS - Informe 001")
print("------------------")
print("Tipo de modelo: clasificacion")
print("Objetivo: dado un galgo en una carrera, quiero clasificarle en 2 grupos: 1o2 o 3a6")
print("Entrada: features de la carrera, features del galgo analizado, features de una galgo agregado (galgos competidores agrupados) y target para train/test.")
print("Categorias (classes): 1o2 (valor 1) y 3a6 (valor 0)")
print("------------------")

def leerFeaturesDelCasoAPredecirDesdeBaseDatos():
    print("Conectando a Base de datos para descargar FEATURES DE CASOS A PREDECIR...")
    con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
    c = con.cursor()

    c.execute('SELECT * from datos_desa.tb_galgos_dataset_prediccion_features_i001;')
    alist = c.fetchall()
    print("Numero de filas leidas: "+str(len(alist)))
    #print("Primera fila de datos: "+str(alist[0]))
    data1 = np.array(alist)

    c.close()
    return data1

print("INICIO")

mejor_modelo = joblib.load('/home/carloslinux/Desktop/GIT_REPO_PYTHON_POC_ML/python_poc_ml/galgos/galgos_i001_MEJOR_MODELO.pkl')

X=leerFeaturesDelCasoAPredecirDesdeBaseDatos()

######################## PREDICCION ##############
print("Prediciendo con matriz de entrada X...")
print("Shape de la matriz X =" + str(X.shape[0])+ "x"+ str(X.shape[1]))

targets_predichos=mejor_modelo.predict(X)

print(targets_predichos)
print("Longitug de salida targets_predichos =" + str(len(targets_predichos)))

print("Guardando resultado en BBDD: datos_desa.tb_galgos_dataset_prediccion_target_i001")
con = pymysql.connect(host='127.0.0.1', user='root', passwd='datos1986', db='datos_desa')
c = con.cursor()
c.execute("""DROP TABLE datos_desa.tb_galgos_dataset_prediccion_target_i001;""")
c.execute("""CREATE TABLE datos_desa.tb_galgos_dataset_prediccion_target_i001 (target int);""")

for target in targets_predichos:
    print("Insertando target="+str(target))
    c.execute("""INSERT INTO datos_desa.tb_galgos_dataset_prediccion_target_i001 (target) VALUES("""+str(target)+""");""")

print("FIN")








