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

def model_builder_cnn(hp, conv_dropout=True):

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
    input_shape = (320,320,3)
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
                  metrics=['binary_accuracy'])
    
    return model
    
def eval_func(y_true, y_prob):
    """
    A function calculating the different evaluation metrics on the test set.
    Converts prediction probabilities y_prob to predicted labels y_pred
    """
    y_pred = np.array([1 if prob >= 0.5 else 0 for prob in y_prob])
    y_true = np.array(y_true)

    print(f"Test accuracy: {round(accuracy_score(y_true, y_pred)*100, 2)} %")
    print(f"Test F1 score: {round(f1_score(y_true, y_pred)*100, 2)} %")
    print(f"Test Precision score: {round(precision_score(y_true, y_pred)*100, 2)} %")
    print(f"Test Recall score: {round(recall_score(y_true, y_pred)*100, 2)} %")
    
# build tuner to fine-tune the hyper-parameters
# for the model

def tune_train_best_model(bodypart,
                          valid_generators,
                          train_generators,
                          test):
    

    print('Start proccess {}.\n'.format(bodypart))

    classes = 2
    tuner = kt.Hyperband(
                        model_builder_cnn,
                        objective='val_binary_accuracy',
                        max_epochs=50,
                        hyperband_iterations=10,
                        directory='data/tuner/'+bodypart,
                        project_name='finetune'
    )

    # define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=10)

    # perfrom search for best parameters
    validation_steps = math.ceil(valid_generators[bodypart].n/ (valid_generators[bodypart].batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators[bodypart].n / (train_generators[bodypart].batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)
    print('Tuner search for {}.\n'.format(bodypart))
    tuner.search(train_generators[bodypart],
                  validation_data=valid_generators[bodypart],
                  validation_steps = validation_steps,
                  steps_per_epoch=steps_per_epoch,
                  epochs=100, 
                  callbacks=[stop_early])
    
    tuners[bodypart] = tuner
    print('Optimal hyperparameters for {}.\n'.format(bodypart))
    # get the optimal hyperparameters
    best_hyperparams[bodypart]=tuner.get_best_hyperparameters(num_trials=1)[0]
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print('\nUnits:',best_hyperparams[bodypart].get('units'), '\nLearning rate:', best_hyperparams[bodypart].get('learning_rate'), '\nConvolutional blocks:', best_hyperparams[bodypart].get('conv_blocks'), '\nDropout rate:', best_hyperparams[bodypart].get('dropout'))
    
#     print('\nUnits:',best_hps.get('units'), '\nLearning rate:', best_hps.get('learning_rate'), '\nConvolutional blocks:', best_hps.get('conv_blocks'), '\nDropout rate:', best_hps.get('dropout'))

    # GET BEST EPOCH

    # build the model with the optimal hyperparameters 
    # train it on the data for 50 epochs

#     model = tuner.hypermodel.build(best_hps)
    model = tuner.hypermodel.build(best_hyperparams[bodypart])
    
    # define the early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=10)
    print('Find best epoch{}.\n'.format(bodypart))
    # perfrom search for best parameters
    history=model.fit(train_generators[bodypart],
                  validation_data=valid_generators[bodypart],
                  validation_steps = validation_steps,
                  steps_per_epoch=steps_per_epoch,
                  epochs=100, 
                  callbacks=[stop_early])


    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch))
    best_epochs[bodypart] = best_epoch
    # train best model for best epoch

    # instantiate the hypermodel 
    # train it with the optimal number of epochs
    print('Train model for with best epoch for {}.\n'.format(bodypart))
    hypermodel = tuner.hypermodel.build(best_hps)
    # plot summary
    hypermodel.summary()
    # retrain the model
    histories_best[bodypart] = hypermodel.fit(train_generators[bodypart],
                  validation_data=valid_generators[bodypart],
                  validation_steps = validation_steps,
                  steps_per_epoch=steps_per_epoch,
                  epochs=best_epoch, 
                  callbacks=[stop_early])
    print('\nSave training history\n')
    with open("data/history/history_fine_tuned_best_model" + bodypart + ".json", "w") as file:
            json.dump(histories_best[bodypart].history, file)
    print('\nSave hypermodel\n')
    hypermodel.save("data/models/_fine_tuned_best_model" + bodypart)
    hypermodel.save("data/models/_fine_tuned_best_model" + bodypart + ".h5")
    print('\nEvaluate hypermodel\n')
    eval_func(test[test['bodypart'] == bodypart]['class'], 
                    model.predict(test_generators[bodypart]))
    hypermodel.evaluate(test_generators[bodypart], verbose=1)
    print("\nDelete hypermodel")
    clean_up(hypermodel)        
    return tuners, best_hyperparams, best_epochs, histories
