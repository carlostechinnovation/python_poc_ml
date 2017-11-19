from __future__ import print_function

from os.path import dirname, join

import numpy as np
from sklearn.utils import Bunch

print("INFORME 003")
print("Tipo de modelo: clasificacion")
print("Objetivo: ")
print("Motivo: quiero operar a 3 días (no intradía ni largo plazo).")
print("Entrada: features de periodos cortos consecutivos arrastrados (t1t2t3, t2t3t4, t3t4t5...)")
print("Categorias (classes): variable cat_variacion_3_dias ")


def cargar_datasets_entrada(n_class=10, return_X_y=False):
    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'data', 'digits.csv.gz'),
                      delimiter=',')
    with open(join(module_path, 'descr', 'digits.rst')) as f:
        descr = f.read()
    target = data[:, -1].astype(np.int)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    if return_X_y:
        return flat_data, target

    return Bunch(data=flat_data,
                 target=target,
                 target_names=np.arange(10),
                 images=images,
                 DESCR=descr)
