
from FuncionesAux import *
import warnings
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
"""
import gc
gc.collect()
"""

warnings.filterwarnings("ignore")
import multiprocessing
import time

#print("Number of cpu : ", multiprocessing.cpu_count())
"""
Parametros del modelo
    - Numero de rondas usadas.
    - Numero de clientes qie participan. Al tener yo los dataset numerados y ordenados,
    el numero significa que lee nos n primeros clientes
"""
rondas = 20
n_clientes = 2
clients = []

"""
multiprocessing es un paquete para lanzar el servidor y todos los clientes a la vez.
se podian lanzar tambien haciendo varios terminales y haciendolos uno a uno.

Se lanza el servidor con la funcion start_server(), se le para el numero de clientes que se van a usar 
y las rondas totales. 
"""
server = multiprocessing.Process(target=start_server, args=(n_clientes, rondas))
server.start()
"""
Primero se tiene que lanzar el servidor, si antes de que se lance el servidor -se lanza algun cliente dara error,
por tanto hago que el programa espera varios segundos para asegurar de que se lanza el servidor. Dependiendo de lo 
que se tarde en leer los datos se pone mas tiempo o menos.
"""
time.sleep(20)

"""
bucle for para leer los clientes. Se usa la funcion start_client(), se le pasa el indice porque como el dataset esta 
numerado, cliente(1) significa que ese cliente lee el dataset 1.
"""
for i in range(n_clientes):
    inx = i + 1
    p = multiprocessing.Process(target=start_client, args=[inx])
    time.sleep(20)
    p.start()
    time.sleep(20)
    clients.append(p)
#esto son parametros del modulo multiprocessing.
server.join()
for client in clients:
    client.join()

