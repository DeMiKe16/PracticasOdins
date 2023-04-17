import os

import numpy as np
import pandas as pd

from FuncionesAux import *

TOTAL_SIZE = 7934685
EP = 1

clientes = []
modelos = []
x_trains = []
y_trains = []
x_tests = []
y_tests = []
accs = []
pesos = []
accs_local = []

dir_base = os.getcwd()
dir_lista_tam = dir_base + "/SMOTETomek/AaTabla_tamanos.csv"

lista_clientes = pd.read_csv(dir_lista_tam)
lista_clientes = np.array(lista_clientes['Vehicle'])

"""
def weighted_median(data, weights):
 
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median
"""
def custom_median(data, weights):
    median = []
    for j in range(len(data[0])):
        me = []
        for k in range(len(data)):
            m = data[k][j]
            me.append(m)
        me = np.transpose(me)
        medi = []
        for i in range(len(m)):
            med = weighted_median(me[i], weights)
            medi.append(med)
        median.append(medi)

    median = np.array(median)
    return median

def custom_median_1d(data, weights):
    median = []
    data = np.transpose(data)
    medi = []
    for i in range(len(data)):
        med = weighted_median(data[i], weights)
        medi.append(med)
    median.append(medi)
    medi = np.array(medi)
    return medi


def entrenar(modelos, x_trains, y_trains, x_tests, y_tests, median):
    accs_local = []
    for i in range(10):
        print("coche " + str(i))
        modelos[i].set_weights(median)
        modelos[i].fit(x_trains[i], y_trains[i], batch_size=300, epochs=EP)
        _, ac = model.evaluate(x_tests[i], y_tests[i], 32, steps=5)
        print("Accuracy= " + str(ac))
        accs_local.append(ac)
        #accs_local.append(modelos[i].evaluate(x_tests[i], y_tests[i], 32, steps=5))
    am = np.mean(accs_local)
    accs_local=[]
    accs.append(am)


def mediana(modelos,pesos):
    pesos_capa1 = []
    pesos_bias1 = []
    pesos_capa2 = []
    pesos_bias2 = []
    pesos_capa3 = []
    pesos_bias3 = []
    for i in range(10):
        pesos_capa1.append(modelos[i].get_weights()[0])
        pesos_bias1.append(modelos[i].get_weights()[1])

        pesos_capa2.append(modelos[i].get_weights()[2])
        pesos_bias2.append(modelos[i].get_weights()[3])

        pesos_capa3.append(modelos[i].get_weights()[4])
        pesos_bias3.append(modelos[i].get_weights()[5])
    """
    median_capa1 = np.median(pesos_capa1, axis=0)
    median_bias1 = np.median(pesos_bias1, axis=0)
    median_capa2 = np.median(pesos_capa2, axis=0)
    median_bias2 = np.median(pesos_bias2, axis=0)
    median_capa3 = np.median(pesos_capa3, axis=0)
    median_bias3 = np.median(pesos_bias3, axis=0)
    # print(median_capa1)
    """
    median_capa1 = w_median(pesos_capa1, pesos)
    median_bias1 = w_median(pesos_bias1, pesos)
    median_capa2 = w_median(pesos_capa2, pesos)
    median_bias2 = w_median(pesos_bias2, pesos)
    median_capa3 = w_median(pesos_capa3, pesos)
    median_bias3 = w_median(pesos_bias3, pesos)
    #print(median_capa1)

    D = np.array([median_capa1, median_bias1, median_capa2, median_bias2, median_capa3, median_bias3])
    return D
"""
def mediana2(modelos, pesos):
    new_weights = []
    for variables in zip(*[client_model.trainable_variables for client_model in modelos]):
"""

def media_pon(modelos,pesos):
    pesos_capa1 = []
    pesos_bias1 = []
    pesos_capa2 = []
    pesos_bias2 = []
    pesos_capa3 = []
    pesos_bias3 = []
    for i in range(10):
        a=modelos[i].get_weights()[0]
        pesos_capa1.append(a)
        b = modelos[i].get_weights()[1]
        pesos_bias1.append(b)

        a = modelos[i].get_weights()[2]
        pesos_capa2.append(a)
        b = modelos[i].get_weights()[3]
        pesos_bias2.append(b)

        a = modelos[i].get_weights()[4]
        pesos_capa3.append(a)
        b = modelos[i].get_weights()[5]
        pesos_bias3.append(b)

    median_capa1 = np.average(pesos_capa1, axis=0, weights=pesos)
    median_bias1 = np.average(pesos_bias1, axis=0, weights=pesos)
    median_capa2 = np.average(pesos_capa2, axis=0, weights=pesos)
    median_bias2 = np.average(pesos_bias2, axis=0, weights=pesos)
    median_capa3 = np.average(pesos_capa3, axis=0, weights=pesos)
    median_bias3 = np.average(pesos_bias3, axis=0, weights=pesos)

    D = np.array([median_capa1, median_bias1, median_capa2, median_bias2, median_capa3, median_bias3])
    return D

def media_bruto(modelos,pesos):
    weights = []
    for i in range(449):
        weight = modelos[i].get_weights()
        weights.append(pesos[i]*weight)

    media = np.sum(weight, axis=0)
    return media

def weighted_median(arrays, weights):#el bueno
    weighted_medians = []
    for i in range(arrays[0].shape[0]):
        for j in range(arrays[0].shape[1]):
            values = [array[i][j] for array in arrays]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            weights = np.array(weights)[indices]
            cumulative_weights = np.cumsum(weights)
            median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
            weighted_medians.append(values[median_index])
    return np.array(weighted_medians).reshape(arrays[0].shape)

def weighted_median_1d(arrays, weights):
    weighted_medians = []
    for i in range(len(arrays[0])):
        values = [array[i] for array in arrays]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        cumulative_weights = np.cumsum(weights)[indices]
        median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
        weighted_medians.append(values[median_index])
    return np.array(weighted_medians)

def w_median(arrays, weights):
    print(np.shape(arrays[0]))
    if np.shape(arrays[0])[1] == 1:
        return weighted_median_1d(arrays, weights)
    else:
        return weighted_median(arrays, weights)

def w_median_2(arrays, weights):
    print(np.shape(arrays[0]))
    new_weights = []
    for variables in zip(*[client_model.trainable_variables for client_model in modelos]):
        new_weight = weighted_median(variables, weights)
        new_weights.append(new_weight)
    return new_weights
def media2(modelos, pesos):
    new_weights = []
    for variables in zip(*[client_model.trainable_variables for client_model in modelos]):
        new_weight = np.average(variables, weights= pesos, axis=0)
        new_weights.append(new_weight)
    return new_weights



"""
def weighed_median(arrays, weights):
    assert len(arrays) == len(weights)
    shape = arrays[0].shape
    n_elements = np.prod(shape)
    flat_data = np.stack([array.flatten() for array in arrays])
    flat_weights = np.tile(weights, (n_elements, 1)).T
    weighted_data = flat_data * flat_weights
    sorted_data = np.sort(weighted_data, axis=0)
    cum_weights = np.cumsum(flat_weights, axis=0)
    total_weight = np.sum(flat_weights, axis=0)
    median = np.zeros(n_elements)
    for i in range(n_elements):
        for j, (value, cum_weight) in enumerate(zip(sorted_data[:, i], cum_weights[:, i])):
            if cum_weight >= total_weight[i] / 2:
                median[i] = value / weights[j]
                break
    return median.reshape(shape)
"""


print("Empezamos...")
for client_n in range(10):
    print(client_n)
    """
    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                'Cohen_Kappa_Score']
    path = dir_base + '/fedMedian/metricas_parties/history_' + str(client_n) + '.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    """
    # dir_datos = "/SMOTETomek/data_party" + str(indx) + ".csv"
    csv = leer_cliente_veremi(client_n + 1)
    csv = pd.read_csv(csv)
    peso = len(csv)/532560
    #print("PESO " + str(peso))
    pesos.append(peso)
    (x_train, y_train), (x_test, y_test) = load_data(client_n + 1)
    x_trains.append(x_train)
    y_trains.append(y_train)
    x_tests.append(x_test)
    y_tests.append(y_test)
    input_shape = len(x_train[0])
    model = create_model_mlp(input_shape)
    model.fit(x_train, y_train, batch_size=200, epochs=EP)
    _, ac = model.evaluate(x_test, y_test, 32, steps=5)
    print("Accuracy= " + str(ac))
    accs_local.append(ac)
    modelos.append(model)

am = np.mean(accs_local)
print(am)
accs_local=[]
accs.append(am)
accs_csv = pd.DataFrame(accs, columns=['Accuracy'])
accs_csv.to_csv(os.getcwd()+"/fedMedian/accuracy_rondas_mediana_4.csv")


print("Termina lectura inicial, vamos con las rondas")

for ronda in range(5):
    print("Ronda " + str(ronda + 1))
    med = mediana(modelos,pesos)
    #med = media2(modelos, pesos)
    print("mediana hecha")
    entrenar(modelos, x_trains, y_trains, x_tests, y_tests, med)
    print("entrenamiento hecho")
    accs_csv = pd.DataFrame(accs, columns=['Accuracy'])
    accs_csv.to_csv(os.getcwd()+"/fedMedian/accuracy_rondas_mediana_4.csv")

print(accs)
