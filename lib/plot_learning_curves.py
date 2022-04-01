#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def plot_history(history, 
                 bodypart,
                 name_model,
                 epochs,
                 data_augmentation='noaugment'):

  fig1  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for accuracy
  plt.plot(history.history['binary_accuracy'], c = 'm')
  plt.plot(history.history['val_binary_accuracy'], "--", c = 'c')
  plt.title('Model accuracy for CNN: ' + bodypart, fontsize=16)
  plt.ylabel('accuracy', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 

  plt.show()
  plt.savefig('data/plots/accuracy_' + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + '.png')
  plt.clf()
    
  fig2  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for loss
  plt.plot(history.history['loss'], c = 'm')
  plt.plot(history.history['val_loss'], "--",  c = 'c')
  plt.title('Model loss for CNN: ' + bodypart, fontsize=16)
  plt.ylabel('loss', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 
  plt.show()
  plt.savefig('data/plots/loss_'  + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + '.png')
  plt.clf()

def plot_history_from_dict(history, 
                 bodypart,
                 name_model,
                 epochs,
                 data_augmentation='noaugment'):

  fig1  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for accuracy
  plt.plot(history['binary_accuracy'], c = 'm')
  plt.plot(history['val_binary_accuracy'], "--", c = 'c')
  plt.title('Model accuracy for CNN: ' + bodypart, fontsize=16)
  plt.ylabel('accuracy', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 

  plt.show()
  plt.savefig('data/plots/accuracy_' + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + '.png')
  plt.clf()
    
  fig2  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for loss
  plt.plot(history['loss'], c = 'm')
  plt.plot(history['val_loss'], "--",  c = 'c')
  plt.title('Model loss for CNN: ' + bodypart, fontsize=16)
  plt.ylabel('loss', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 
  plt.show()
  plt.savefig('data/plots/loss_'  + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + "_" + bodypart + '.png')
  plt.clf()

def plot_history_all(history, 
                 name_model,
                 epochs,
                 data_augmentation='noaugment'):

  fig1  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for accuracy
  plt.plot(history.history['accuracy'], c = 'm')
  plt.plot(history.history['val_accuracy'], "--", c = 'c')
  plt.title('Model accuracy for CNN: ', fontsize=16)
  plt.ylabel('accuracy', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 

  plt.show()
  plt.savefig('data/plots/accuracy_' + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + '_.png')
  plt.clf()
    
  fig2  = plt.figure(figsize=(8,4), dpi = 150)

  # summarize history for loss
  plt.plot(history.history['loss'], c = 'm')
  plt.plot(history.history['val_loss'], "--",  c = 'c')
  plt.title('Model loss for CNN: ', fontsize=16)
  plt.ylabel('loss', fontsize=16)
  plt.xlabel('epoch', fontsize=16)
  plt.legend(['train', 'validation'], 
             loc='upper right', 
             fontsize=16)
  matplotlib.rc('xtick', labelsize=16) 
  matplotlib.rc('ytick', labelsize=16) 
  plt.show()
  plt.savefig('data/plots/loss_' + name_model + "_" + str(epochs) + "epochs" + "_" + data_augmentation + '_.png')
  plt.clf()
