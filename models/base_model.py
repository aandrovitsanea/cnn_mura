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
print("Tensorflow version " + tf.__version__)


def architecture_model(input_shape,
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
            filters=8*(2**(i+1)),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=conv_activation,
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
                rate=0.2,
                name='Dropout-{0:d}'.format(i + 1)
            )(x)
    # Flatten the convolved images so as to input them to a Dense Layer
    x = Flatten(name='Flatten')(x)
    
    # Add a dense before output layer and drop out
    x = tf.keras.layers.Dense(units=100,
                              activation='relu',
                              name='Dense-1')(x)

    x = tf.keras.layers.Dropout(rate= 0.2, 
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

def build_model(learning_rate): #learning_rate=1e-3
    model = architecture_model(input_shape = (320,320,3),
                        conv_layers = 4,
                        conv_activation = 'relu',
                        conv_dropout = 0.2,
                        units = 1,
                        output_activation = 'sigmoid',
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss = 'binary_crossentropy',
                        metrics = ['binary_accuracy'])
    return model

def clean_up(model):
    K.clear_session()
    del model
    gc.collect()
    
print("Train model")
  
def train_model(patience,
                model, 
                bodypart,
                epochs,
                valid_generators,
                train_generators): # patience=20, epochs=100
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             patience=patience,
                                             verbose=1)  
    validation_steps = math.ceil(valid_generators[bodypart].n/ valid_generators[bodypart].batch_size)
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
    return history

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
    
    

def build_train_evaluate(name_model, 
                         bodyparts,
                         learning_rate,
                         epochs, 
                         patience,
                         train_generators,
                         valid_generators,
                         test_generators,
                         test,
                         data_augmentation='noaugment'
                         ):
    histories = {}
    for bodypart in bodyparts:
        print(bodypart)
        print("\nBuild model\n")
        model = build_model(learning_rate)
        print('\nTrain model\n')
        print(bodypart)
        histories[bodypart] = train_model(patience,
                                            model, 
                                            bodypart,
                                            epochs,
                                            valid_generators,
                                            train_generators)
        print('\nSave training history\n')
        with open("data/history/history_" + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + ".json", "w") as file:
            json.dump(histories[bodypart].history, file)
        print('\nSave model\n')
        model.save("data/models/" + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart)
        model.save("data/models/" + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + ".h5")
        print('\nEvaluate model\n')
        eval_func(test[test['bodypart'] == bodypart]['class'], 
                        model.predict(test_generators[bodypart]))
        model.evaluate(test_generators[bodypart], verbose=1)
        print("\nDelete model")
        clean_up(model)
        
        return histories

