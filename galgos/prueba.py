mi_path = "/home/carloslinux/Desktop/DATOS_LIMPIO/galgos/borrar.txt"
miarray = [[0.12], [0.23]]
fichero_resultados = open(mi_path, 'w')
for item in miarray:
    print(str(item[0]))
    # fichero_resultados.write(str(item)+"\n")
