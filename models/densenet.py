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

# add model on top
def build_model_on_densenet():
  # DenseNet model
  densenet_model = tf.keras.applications.DenseNet121(weights='imagenet',
                                                    input_shape = (320,320,3),
                                                    include_top = False)
  densenet_model.trainable = False
  # add new model on top
  inputs = tf.keras.Input(shape=(320,320,3))
  x = densenet_model(inputs, 
                    training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(inputs, outputs)
  model.summary()

  # train the top layer
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'])
  return model

callbacks_2 = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', 
                                             patience=20,
                                             restore_best_weights=True)

def train_model(model, 
                bodypart,
                valid_generators,
                train_generators,
                epochs):
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
                        callbacks=[callbacks_2]
    )
    return history


# add model on top
def build_model_on_densenet_14classes():
  # DenseNet model
  densenet_model = tf.keras.applications.DenseNet121(weights='imagenet',
                                                    input_shape = (320,320,3),
                                                    include_top = False)
  densenet_model.trainable = False
  # add new model on top
  inputs = tf.keras.Input(shape=(320,320,3))
  x = densenet_model(inputs, 
                    training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  classes=14
  outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)
  model = tf.keras.Model(inputs, outputs)
  model.summary()

  # train the top layer
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  return model

def train_model_14classes(model, 
                            valid_generators,
                            train_generators,
                            epochs):
    callbacks3 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                             patience=8,
                                             restore_best_weights=True)
    
    validation_steps = math.ceil(valid_generators.n/ (8*valid_generators.batch_size))
    print("Using validation_steps = %d" % validation_steps)
    steps_per_epoch = math.ceil(train_generators.n / (16*train_generators.batch_size))
    print("Using steps_per_epoch = %d" % steps_per_epoch)

    history = model.fit(train_generators,
                        validation_data = valid_generators,
                        validation_steps = validation_steps,
                        steps_per_epoch = steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callbacks3]
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
