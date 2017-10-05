import mysql.connector


def funcionBaseDatos():
    print("POC- Conexion a Base de datos")
    con = mysql.connector.connect(user='root', password='datos1986',host='127.0.0.1',database='datos_desa')
    c = con.cursor()

    c.execute("""CREATE TABLE datos_desa.tb_poc_bbdd (campo1 int);""")
    c.execute("""INSERT INTO datos_desa.tb_poc_bbdd (campo1) VALUES(1),(2),(3);""")
    c.execute("""SELECT campo1 FROM datos_desa.tb_poc_bbdd LIMIT 5;""")
    filas = []
    for row in c:
        user = {
            'valor1': row[0]
        }
        print(row[0])
        filas.append(user)

    c.execute("""DROP TABLE datos_desa.tb_poc_bbdd;""")
    c.close()
    return filas


print("INICIO")
funcionBaseDatos()
print("FIN")