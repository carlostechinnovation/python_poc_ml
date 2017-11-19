from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

print("TUTORIAL-00")
print("Entrada: array de imagenes (cada una es un chorro de 64 digitos)")
print("Objetivo: dada una imagen, predecir que digito representa."+
      "\nHay 10 clases posibles (por tanto usamos CLASIFICACION) en los que entrenamos (fit) un estimador para predecir sobre muestras nuevas")

print("Cargando datasets...")
digits = datasets.load_digits()

print("\nDataset DIGITS.data:")
print("Shape de la matriz =" + str(digits.data.shape[0])+ "x"+ str(digits.data.shape[1]))
#print(digits.data)
print("Dataset DIGITS.target:")
print("Longitud=" + str(len(digits.target)))
#print(digits.target)

#------- Learning and predicting -----
print("Algorimo SVC (Support Vector Classification)")
clf = svm.SVC(gamma=0.001, C=100.)

print("Ajustando modelo con matriz de entrenamiento X y su solucion correcta y...")
X=digits.data[:-1]
y=digits.target[:-1]
clf.fit(X, y)

print("Guardando modelo ajustado...")
modeloGuardado = joblib.dump(clf, 'ejemplos/tutorial_00_modelo.pkl')
clf2 = joblib.load('ejemplos/tutorial_00_modelo.pkl')

print("Prediciendo con matriz de entrada X2...")
X2=digits.data[-1:] #entrada 1x64
print("Shape de la matriz X2 =" + str(X2.shape[0])+ "x"+ str(X2.shape[1]))
y2=clf2.predict(X2)
print(y2)
print("Longitug de salida y2 =" + str(len(y2)))



