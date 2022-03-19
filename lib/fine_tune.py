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





def model_builder(hp, conv_dropout=True):

    #########################################
    ## set hyperparameters for fine tuning ##
    #########################################

    # tune the number of units
    hp_units = hp.Int('units', min_value=32, 
                               max_value=32*8, 
                               step=32)
    
    # tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', 
                                 values=[1e-3, 1e-2, 1e-4])
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # convulational layers
    conv_layers = hp.Int('conv_blocks', 2, 6, 
                         default=3)

    # dropout rate
    dropout_rate = hp.Float('dropout', 0, 0.5, 
                            step=0.1, default=0.2)
#     # activation
#     activation=hp.Choice("activation", ["relu", "tanh"])
    
    np.random.seed(1000) # Define the seed for numpy to have reproducible experiments.
    set_seed(150) # Define the seed for Tensorflow to have reproducible experiments.

    ###################
    ## define model ##
    ###################

    # define the input layer
    input = Input(
        shape=input_shape,
        name='Input'
    )

    x = input
    # define the convolutional layers
    # using functional api
    for i in range(conv_layers):
        # filters
        filters = hp.Int('filters_' + str(i), 32, 224, 
                         step=32)

        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation="relu",
            name='Conv2D-{0:d}'.format(i + 1)
        )(x)

        x = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='MaxPool2D-{0:d}'.format(i + 1)
        )(x)

        if conv_dropout:
            x = Dropout(
                rate = dropout_rate,
                name='Dropout-{0:d}'.format(i + 1)
            )(x)

    # flatten the convolved images 
    # fit to feed the dense layer
    x = Flatten(name='Flatten')(x)
    
    # Add a dense before output layer and drop out
    x = tf.keras.layers.Dense(units=hp_units,
                              activation="relu",
                              name='Dense-1')(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate, 
                                name='Dropout-{0:d}'.format(conv_layers + 1))(x)
    
    
    # define the output layer
    output = Dense(
        units = 1,
        activation = 'softmax',
        name='Output'
    )(x)

    # define the model and train it
    model = Model(inputs=input, outputs=output)

    model.compile(
                  optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
    
def tuner_search(directory,
                 project_name,
                 max_epochs,
                 hyperband_iterations):
    print(bodypart)
    
    tuner = kt.Hyperband(
                    model_builder,
                    objective='val_accuracy',
                    max_epochs=max_epochs,
                    hyperband_iterations=hyperband_iterations,
                    directory=directory,
                    project_name=project_name,
                    patience,
                    bodypart,
                    epochs
                    )

    # define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', 
                                                 patience=patience)

    perfrom search for best parameters
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    print("Starting tuner search")
    
    tuner.search(train_generators[bodypart],
                validation_data=valid_generators[bodypart],
                steps_per_epoch=steps_per_epoch,
                epochs=epochs, 
                callbacks=[stop_early])

    # get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print('\nUnits:',best_hps.get('units'), '\nLearning rate:', best_hps.get('learning_rate'), '\nConvolutional blocks:', best_hps.get('conv_blocks'), '\nDropout rate:', best_hps.get('dropout'))    

    return tuner, best_hps


# find best epoch

def find_best_epoch(best_hps,
                    patience,
                    bodypart,
                    epochs,
                    train_generators,
                    valid_generators):
    print(bodypart)
    print('Build hypermodel')
    model = tuner.hypermodel.build(best_hps)
    # define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', 
                                                  patience=patience)
    # perfrom search for best parameters
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    print("Fit model")
    history=model.fit(train_generators[bodypart],
                        validation_data=valid_generators[bodypart],
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, 
                        callbacks=[stop_early])


    val_acc_per_epoch = history.history['val_accuracy']
    print("Find best epoch")
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch))
    return best_epoch

def build_train_best_model(best_hps,
                           best_epoch,
                           patience,
                           bodypart,
                           train_generators,
                           valid_generators):
    print(bodypart)
    # instantiate the hypermodel 
    # train it with the optimal number of epochs
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', 
                                                  patience=patience)
    hypermodel = tuner.hypermodel.build(best_hps)
    # plot summary
    hypermodel.summary()
    # retrain the model
    history    = hypermodel.fit(train_generators[bodypart],
                            validation_data=valid_generators[bodypart],
                            steps_per_epoch=steps_per_epoch,
                            epochs=best_epoch, 
                            callbacks=[stop_early])
    return history


#############################
#############################

def parameters_for_tuning():
    
    ################################
    set hyperparameters for fine tuning ##
    ################################
    hp = kt.HyperParameters()

    tune the number of units
    hp_units = hp.Int('units', min_value=32, 
                               max_value=32*8, 
                               step=32)
    
    tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', 
                                 values=[1e-3, 1e-2, 1e-4])
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    convulational layers
    conv_layers = hp.Int('conv_blocks', 2, 6, 
                         default=3)

    dropout rate
    dropout_rate = hp.Float('dropout', 0.1, 0.5, 
                            step=0.1, default=0.2)
    
    activation
    activation=hp.Choice("activation", ["relu", "tanh"])
    
    filters
    for i in range(conv_layers):
        filters = hp.Int('filters_' + str(i), 32, 224, 
                     step=32)
    
    return hp_units, optimizer, conv_layers, dropout_rate, activation, filters
    
def build_model(input_shape,
                conv_layers,
                filters,
                activation,
                dropout_rate,
                units,
                optimizer,
                verbose=1):
    
    np.random.seed(1234) # Define the seed for numpy to have reproducible experiments.
    set_seed(9876) # Define the seed for Tensorflow to have reproducible experiments.
    
    Define the input layer.
    input = Input(
        shape=input_shape,
        name='Input'
    )

    x = input
    Define the convolutional layers.
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
    Flatten the convolved images so as to input them to a Dense Layer
    x = Flatten(name='Flatten')(x)
    
    Add a dense before output layer and drop out
    x = tf.keras.layers.Dense(units=units,
                              activation=activation,
                              name='Dense-1')(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate, 
                                name='Dropout-{0:d}'.format(conv_layers + 1))(x)
    Define the output layer.
    output = Dense(
        units=units,
        activation='sigmoid',
        name='Output'
    )(x)

    Define the model and train it.
    model = Model(inputs=input, 
                  outputs=output)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy'])
    
    print summary and architecture
    model.summary()

    return model

def tuner_hyperband(model, 
                    objective, # 'val_accuracy'
                    max_epochs,
                    hyperband_iterations, # 2
                    directory, #where to save the weight
                    project_name
                    ):
    build tuner to fine-tune the hyper-parameters
    for the model
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
                epochs,
                valid_generators,
                bodypart,
                train_generators
                ):
    
    build tuner to fine-tune the hyper-parameters
    for the model
    tuner = kt.Hyperband(
                        model,
                        objective=objective,
                        max_epochs=max_epochs,
                        hyperband_iterations=hyperband_iterations,
                        directory=directory,
                        project_name=project_name
    )
    define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                 patience=patience)

    perfrom search for best parameters
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

    get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Units: {}\nLearning rate: {}\n Convolutional blocks: {}\nDropout rate: {}\nActivation Function: {}\nFilters'.format(best_hps.get('units'), best_hps.get('learning_rate'), best_hps.get('conv_blocks'), best_hps.get('dropout'), best_hps.get('activation'), best_hps.get('filters')))
          
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
    
