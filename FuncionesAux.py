import os
import time

import numpy
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import flwr as fl
#import emlearn
from keras.layers import Embedding, GRU

# Import Data Science Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools
import random

# Import visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Tensorflow Libraries
from tensorflow import keras
from tensorflow.keras import layers,models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing

# System libraries
from pathlib import Path
import os.path

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from random import sample
import itertools

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot

"""
Funcion que crea el modelo que se va a usar en el federated learning, paso simplemente como parametro
la dimension de la primera capa segun la cantidad de columnas que tenga el dataset
"""
def create_model_mlp():
        # Resize Layer
    resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(224,224),
    layers.experimental.preprocessing.Rescaling(1./255),
    ])

    # Setup data augmentation
    data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),                       
    ], name="data_augmentation")

        
    # Load the pretained model
    pretrained_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(300, 300, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    # Create checkpoint callback
    checkpoint_path = "insect_classification_model_checkpoint"
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        monitor="val_accuracy",
                                        save_best_only=True)
    

    inputs = pretrained_model.input
    x = resize_and_rescale(inputs)
    x = data_augmentation(x)

    x = Dense(256, activation='relu')(pretrained_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    #model = keras.models.load_model('/home/miguel/Flower_ejemplo/insectos.h5')

    model.compile(
        optimizer=Adam(0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

"""
Funcion que crea el cliente del modulo flower
"""
class tfmlpClient(fl.client.NumPyClient):
    def __init__(self, model, train_images, val_images, test_images, early_stopping, checkpoint_callback, party_number):
        self.model = model
        self.train_images = train_images
        self.test_images = test_images
        self.val_images = val_images
        self.early_stopping = early_stopping
        self.checkpoint_callback = checkpoint_callback
        self.party_number = party_number

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    """
    La parte mas importante es aqui, donde el cliente entrena su modelo y manda los pesos al servidor.
    """
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        dir_base = os.getcwd()
        # Update local model parameters

        #parameters significa los pesos que ha enviado el server a los clientes
        self.model.set_weights(parameters)
        mean_weights = parameters

        # Aqui se guardan los parametros establecidos por la funcion fit_config()
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        rnd = config["round"]
        steps = config["val_steps"]
        rondas = config["rounds"]
        """
        if rnd == 1:
            self.model.set_weights(parameters)
        """

        print(("Ronda: " + str(rnd)))

        """
        if rnd == rondas & self.party_number == 1:
            dir_ = os.getcwd()
            modelo_50 = self.model
            modelo_50.set_weights(parameters)
            modelo_50.save(dir_ + '/modelo_50.h5')
        """

        """
        Aqui es donde se entrena al modelo, yo utilizo otra funcion, pero te pongo if False para que siempre
        vaya al else, y asi es igual a FedAvg
        """
        if False:#rnd > 1:
            for epoch in range(epochs):
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    batch_size,
                    1,
                    validation_split=0.1,
                )
                theta = 0.98
                new_param = fedplus(self.model.get_weights(), mean_weights, theta)
                self.model.set_weights(new_param)
        else:

            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.train_images,
                steps_per_epoch=len(self.train_images),
                validation_data=self.val_images,
                validation_steps=len(self.val_images),
                epochs=50
            )
            new_param = self.model.get_weights()

        #actual = history.history["accuracy"][-1]

        # Return updated model parameters and results
        # parameters_prime = self.model.get_weights()
        """
        Aqui se guardan los pesos generados por el modelo en esta ronda
        """
        new_param = self.model.get_weights()
        parameters_prime = new_param #esta variable es la que se envia
        num_examples_train = len(self.train_images) #FedAvg hace la media ponderada segun el tamaño del dataset de los clientes
                                               #y en esta variable se guarda

        """
        Como no entra en la parte de validacion para obtener el accuracy la hago aqui que entra seguro. Primero hacemos
        las predicciones usando la parte de validacion que establecimos al principio
        """
        y_pred = self.model.predict(self.val_images)
        y_pred = np.argmax(y_pred, axis=1)
        #y_pred = (self.model.predict(self.x_test) > 0.5).astype(int)

        # Esta variable guarda los resultados que se han hecho despues del entrenamiento
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],

            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],

        }

        # Guardar en csv externo
        """
        Toda esta parte es simplemente para calcular todas las metricas (Accuracy y demas) e irlas guardando en un 
        csv con los resultados de cada ronda, para ver la evolucion
        """
        aux = []
        path = dir_base + '/metricas_parties/history_' + str(self.party_number) + '.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        loss, accuracy = self.model.evaluate(self.test_images, steps= steps, verbose= 1)
        """lista = {
            "Accuracy": accuracy,
            "Recall": recall_score(self.y_test, y_pred, average='weighted'),
            "Precision": precision_score(self.y_test, y_pred, average='weighted'),
            "F1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "FPR": my_FPR(self.y_test, y_pred),
            "Matthew_Correlation_coefficient": matthews_corrcoef(self.y_test, y_pred),
            "Cohen_Kappa_Score": cohen_kappa_score(self.y_test, y_pred)
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))"""
        #del loss, accuracy, lista

        """ Esto es simplemente porque en mi caso veiamos tambien los resultados de cada clase
        aux = []
        aux_pre = []
        path = dir_base + '/class acc/Acc_class_party_' + str(self.party_number) + '.csv'
        path_precision = dir_base + '/precision_classes/Pre_class_party_' + str(self.party_number) + '.csv'
        col_name = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']

        acc_classes = sklearn.metrics.classification_report(self.y_test, y_pred, output_dict=True)
        acc_classes = pd.DataFrame(acc_classes).transpose()

        accuracy_c = {
            "class 0": acc_classes.iloc[0, 1],
            "class 1": acc_classes.iloc[1, 1],
            "class 2": acc_classes.iloc[2, 1],
            "class 3": acc_classes.iloc[3, 1],
            "class 4": acc_classes.iloc[4, 1],
            "class 5": acc_classes.iloc[5, 1],
        }

        precision_class = {
            "class 0": acc_classes.iloc[0, 0],
            "class 1": acc_classes.iloc[1, 0],
            "class 2": acc_classes.iloc[2, 0],
            "class 3": acc_classes.iloc[3, 0],
            "class 4": acc_classes.iloc[4, 0],
            "class 5": acc_classes.iloc[5, 0],
        }

        aux.append(accuracy_c)
        aux_pre.append(precision_class)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        df2 = pd.DataFrame(aux_pre, columns=col_name)
        df2.to_csv(path_precision, index=None, mode="a", header=not os.path.isfile(path_precision))
        del aux, aux_pre, col_name, acc_classes, accuracy_c, precision_class, df1, df2
        """
        self.model.save('./Modelos/insecto' + str(self.party_number) + '.h5')
        # Y ya al final mandamos los pesos, la longitud del dataset y los resultados al servidor
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.test_images, steps= steps, verbose= 1)
        num_examples_test = len(self.test_images)
        return loss, num_examples_test, {"accuracy": accuracy}


#Esta es simplemente la funcion de agragacion que uso
def fedplus(weights, mean, theta):
    z = numpy.asarray(mean)
    weights = numpy.asarray(weights)

    fedp = theta * weights + (1 - theta) * z
    return fedp


"""
Esta funcion es la que uso para crear los clientes
"""
def start_client(client_n):

    BATCH_SIZE = 32
    IMAGE_SIZE = (300, 300)

    dataset = "./Dataset/Cliente" + str(client_n)
    walk_through_dir(dataset)

    image_dir = Path(dataset)

    # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=1)

    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        validation_split=0.2
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    # Split the data into three categories.
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # Create checkpoint callback
    checkpoint_path = "insect_classification_model_checkpoint"
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        monitor="val_accuracy",
                                      save_best_only=True)

    # Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
    early_stopping = EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                    patience=3,
                                                    restore_best_weights=True) # if val loss decreases for 3 epochs in a row, stop training

    model = create_model_mlp()


    """
    Aqui se crea el cliente en la forma que esta en la libreria, se para el modelo, el dataset y el numero
    de cliente que es
    """
    client = tfmlpClient(model, train_images,test_images, val_images, early_stopping, checkpoint_callback, client_n)
    # client = sklearnClient(model, x_train, y_train, x_test, y_test)
    print(("party " + str(client_n) + " lista"))
    del model,client_n

    # Y aqui se lanza el cliente
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)

"""
Funcion que crea el servidor.
    - Parties: numero de clientes que formaran parte del escenario federado.
    - rounds: numero de rondas
"""
def start_server(parties, rounds):
    """
    Se cargan los datos de entrenamiento y validacion. En este caso solo hace falta leer los datos de
    entrenamiento, por motivos de estuctura del modulo flower el servidor usa los datos de entrenamiento
    para crear los primeros pesos del modelo y enviarlos antes de la ronda 1.
    """
    
    BATCH_SIZE = 32
    IMAGE_SIZE = (300, 300)

    dataset = "./Dataset/Cliente1"
    walk_through_dir(dataset)

    image_dir = Path(dataset)

    # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=1)

    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        validation_split=0.2
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    # Split the data into three categories.
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # Create checkpoint callback
    checkpoint_path = "insect_classification_model_checkpoint"
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        monitor="val_accuracy",
                                      save_best_only=True)

    # Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
    early_stopping = EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                    patience=3,
                                                    restore_best_weights=True) # if val loss decreases for 3 epochs in a row, stop training

    model = create_model_mlp()



    # Create strategy
    """
    Se crean los parametros que se usaran en el entorno ademas del metodo para combinar los pesos de los clientes,
    en este caso es FedAvg, que hace la media de los pesos
        - fraction fit: % de numero de clientes que tiene que haber estar disponibles durante el entorno
         en el proceso de entrenamiento.
        - fraction evaluate: % de numero de clientes que tiene que haber estar disponibles durante el entorno
         en el proceso de evaluacion. (No se porque pero a mi numca me entre en ese punto, pero hay que ponerlo)
        - min_evaluate_clients: numero de clientes que se toman en la ronda para la evaluacion
        - min_available_clients: numero de clientes que se toman en la ronda para el entrenamiento (yo tomo
        todos)
        - on_fit_config_fn: parametros usados durante el entrenamiento
        - on_evaluate_config_fn: parametros usados durante la evaluacion
        -initial_parameters: pesos que se pasan al inicio de la ronda 1 (yo no los paso, pero da igual en verdad)
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=parties,
        min_evaluate_clients=parties,
        min_available_clients=parties,
        #evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()))

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy)


def get_eval_fn(model):
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    BATCH_SIZE = 32
    IMAGE_SIZE = (300, 300)

    dataset = "./Dataset/Cliente1"
    walk_through_dir(dataset)

    image_dir = Path(dataset)

    # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    image_df = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=1)

    test_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    #IMAGENES DE ENTRENAMIENTO

    train_images = []       
    train_labels = []
    img_size = 224
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    class_count = len(list(train_images.class_indices.keys()))

    # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
            weights,
    ):  # -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]: model.set_weights(weights)

        # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_images)
        """
        aux = []
        path = '/home/enrique/Flower/sklearn-logreg-mnist/tf red neuronal/metricas_parties/history_Server.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        lista = {
            "Accuracy": accuracy,
            "Recall": 0,
            "Precision": 0,
            "F1_score": 0,
            "Matthew_Correlation_coefficient": 0,
            "Cohen_Kappa_Score": 0
        }

        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        """
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 256,
        "local_epochs": 5,  # if rnd < 2 else 2,
        "round": rnd,
        "val_steps": 5,
        "rounds": 20
    }
    return config


def evaluate_config(rnd):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}




