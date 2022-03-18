#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import matplotlib.image as mpimg
# from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import random, os
import numpy as np
import pandas as pd
# import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
#%matplotlib inline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import gc
# from skimage.transform import resize

from tensorflow.keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPool2D # Layers to be used for building our model
from tensorflow.keras.models import Model # The class used to create a model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

from sklearn.metrics import f1_score, recall_score, precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#! pip install -q -U keras-tuner
import keras_tuner as kt

import json

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from google.colab import files
# from google.colab import drive
print("Tensorflow version " + tf.__version__)

def parameters_for_tuning():
    
    #########################################
    ## set hyperparameters for fine tuning ##
    #########################################

    # tune the number of units
    hp_units = hp.Int('units', min_value=32, 
                               max_value=32*8, 
                               step=32)
    
    # tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[min_value=1e-4,
                                                          max_value=1e-2, 
                                                          sampling="log"])
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # convulational layers
    conv_layers = hp.Int('conv_blocks', 2, 5, default=3)

    # dropout rate
    dropout_rate = hp.Float('dropout', 0.1, 0.5, step=0.1, default=0.2)
    
    # activation
    activation=hp.Choice("activation", ["relu", "tanh"])
    
    # filters
    filters = hp.Int('filters_' + str(i), 32, 224, step=32)
    
    return hp_units, optimizer, conv_layers, dropout_rate, activation, filters
    
def build_model(input_shape,
                conv_layers,
                conv_activation,
                conv_dropout,
                units,
                output_activation,
                optimizer,
                loss,
                metrics,
                verbose=1):
    
    np.random.seed(1234) # Define the seed for numpy to have reproducible experiments.
    set_seed(9876) # Define the seed for Tensorflow to have reproducible experiments.
    
    # Define the input layer.
    input = Input(
        shape=input_shape,
        name='Input'
    )

    x = input
    # Define the convolutional layers.
    for i in range(conv_layers):
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=activation,
            name='Conv2D-{0:d}'.format(i + 1)
        )(x)
        x = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='MaxPool2D-{0:d}'.format(i + 1)
        )(x)
        x = Dropout(
            rate=dropout_rate,
            name='Dropout-{0:d}'.format(i + 1)
        )(x)
    # Flatten the convolved images so as to input them to a Dense Layer
    x = Flatten(name='Flatten')(x)
    
    # Add a dense before output layer and drop out
    x = tf.keras.layers.Dense(units=hp_units,
                              activation=activation,
                              name='Dense-1')(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate, 
                                name='Dropout-{0:d}'.format(conv_layers + 1))(x)
    # Define the output layer.
    output = Dense(
        units=units,
        activation=output_activation,
        name='Output'
    )(x)

    # Define the model and train it.
    model = Model(inputs=input, 
                  outputs=output)
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=metrics)
    
    # print summary and architecture
    model.summary()

    return model

def tuner_hyperband(model, 
                    objective, # 'val_accuracy'
                    max_epochs,
                    hyperband_iterations, # 2
                    directory=directory, #where to save the weight
                    project_name=project_name
                    ):
    # build tuner to fine-tune the hyper-parameters
    # for the model
    tuner = kt.Hyperband(
                        model,
                        objective=objective,
                        max_epochs=max_epochs,
                        hyperband_iterations=hyperband_iterations,
                        directory=directory,
                        project_name=project_name
    )
    
    return tuner

def tuner_search(model, 
                objective, # 'val_accuracy'
                max_epochs,
                hyperband_iterations, # 2
                directory, #where to save the weight
                project_name,
                monitor, # 'val_loss'
                patience,
                epochs):
    
    tuner = tuner_Hyperband(model, 
            objective, # 'val_accuracy'
            max_epochs,
            hyperband_iterations, # 2
            directory=directory, #where to save the weight
            project_name=project_name
            )
    # define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                 patience=patience)

    # perfrom search for best parameters
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    tuner.search(train_generators[bodypart],
                validation_data=valid_generators[bodypart],
                validation_steps = validation_steps,
                steps_per_epoch = steps_per_epoch,
                epochs=epochs, 
                callbacks=[stop_early])

    # get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Units: ',best_hps.get('units'), ', Learning rate: ', best_hps.get('learning_rate'), ', Convolutional blocks: ', best_hps.get('conv_blocks'), ', Dropout rate: ', best_hps.get('dropout'))
    return best_hps

def build_best_model(best_hps):
    model = tuner.hypermodel.build(best_hps)
    return model

def find_best_epoch(model,
                    monitor,
                    patience,
                    epochs):
    callbacks = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                patience=patience,
                                                verbose=1)
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    history = model.fit(train_generators[bodypart],
                        validation_data = valid_generators[bodypart],
                        validation_steps = validation_steps,
                        steps_per_epoch = steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callbacks]
    )
    
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch))
    return best_epoch

def train_best_model(model,
                    monitor,
                    patience,
                    best_epoch):
    callbacks = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                patience=patience,
                                                verbose=1)
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    history = model.fit(train_generators[bodypart],
                        validation_data = valid_generators[bodypart],
                        validation_steps = validation_steps,
                        steps_per_epoch = steps_per_epoch,
                        epochs=best_epoch,
                        verbose=1,
                        callbacks=[callbacks]
    )
    return history
    
