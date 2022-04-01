#!/usr/bin/env python3

def data_loader(data_augmentation): # data_augmentation = "noaugment", "light_augment", "deep_augment"
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import load_img
    import os

    # mount data from gdrive and change directory

    data_path = 'data/MURA-v1.1/'
    directory = 'data/'
    os.getcwd()

    # Data  munging

    def import_data(path):
        df_imgs = pd.read_csv(path,
                                header=None,
                                names=['img_path'])
        df = pd.DataFrame(df_imgs['img_path'].str.split('/').to_list())
        df.rename(columns={2:'bodypart',3:'patient',4:'study',5:'img_path'}, 
                    inplace=True)
        df.drop(columns=[0,1],inplace=True)
        df['img_path'] = df_imgs['img_path']
        result_df = pd.DataFrame(df['study'].str.split('_').to_list())[1]
        df['result'] = result_df
        labels = np.where(df['result']=='positive',1,0)
        df['class'] = labels
        df['diagnosis'] = np.where(df['result']=='positive','abnormal','normal')
        return df

    # train and test dataset

    print("\nImport train and test data\n")
    train = import_data(data_path + 'train_image_paths.csv')
    test = import_data(data_path  + 'valid_image_paths.csv')
         
    # train and validation set for each per bodypart
    def train_validation_bodyparts(bodypart, df, percentage):
        df_bp = df[df['bodypart'] == bodypart]
        # unique patients for each bodypart
        patients = df_bp['patient'].unique()
        # percentage of patients for validation data
        idx_size = int(percentage * len(patients))
        idx = np.random.choice(len(patients), size=idx_size, replace=False)
        valid_patients = patients[idx]
        valid_flag = df_bp['patient'].isin(valid_patients)
        # training and validation sets for this specific bodypart
        # based on the patients selected
        train_df = df_bp[~valid_flag].reset_index()
        valid_df = df_bp[valid_flag].reset_index()

        return train_df, valid_df

    # dictionary for training and validation set
    print("\nCreate dictionary for training and validation set per body part\n")
    bodyparts_train_validation_dict = {}
    for bodypart in set(train.bodypart):
        bodyparts_train_validation_dict[bodypart] = train_validation_bodyparts(bodypart, train,  0.1)
        
    # Data Generator
    if data_augmentation == "noaugment":
        print("No data augmentation")
        train_generator = ImageDataGenerator(
            rescale = 1./255
            )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )
    if data_augmentation == "light_augment":
        print("Doing a light data augmentation")
        train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                horizontal_flip = True
        )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )
    if data_augmentation == "deep_augment":
        print("Doing a light data augmentation")
        train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                fill_mode='nearest',
                horizontal_flip = True,
                vertical_flip=True,
                featurewise_center=True,
                featurewise_std_normalization=True,
        )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )        
    # Per part of the body

    def generate_data(df, 
                    bodypart, 
                    generator,
                    indication): # 0 for train and 
                                # 1 for validation
        if type(df) == dict:
            dataframe = df[bodypart][indication]
        if type(df) == pd.core.frame.DataFrame:
            dataframe = df[df['bodypart'] == bodypart]
        generated = generator.flow_from_dataframe(
            dataframe = dataframe,
            directory = directory,
            x_col = 'img_path',
            y_col = 'class',
            class_mode = 'raw',
            batch_size = 32,
            seed = 34,
            target_size=(320,320)
        )
        return generated
    
    print("\nCreate dictionary for training, validation and test set per body part\n")
    bodyparts = set(train.bodypart)
    train_generators = {}
    valid_generators = {}
    test_generators = {}
    for bodypart in set(test.bodypart):
        train_generators[bodypart] = generate_data(bodyparts_train_validation_dict,
                                        bodypart,
                                        train_generator,
                                        0)
        valid_generators[bodypart] = generate_data(bodyparts_train_validation_dict,
                                        bodypart,
                                        valid_generator,
                                        1)
        test_generators[bodypart] = generate_data(test,
                                        bodypart,
                                        valid_generator,
                                        1)  

    print("Data munging completed")
    return train, test, bodyparts, bodyparts_train_validation_dict, train_generators, valid_generators, test_generators


def load_all(data_augmentation): # data_augmentation = "noaugment", "light_augment", "deep_augment"
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import load_img
    import os

    # mount data from gdrive and change directory

    data_path = 'data/MURA-v1.1/'
    directory = 'data/'
    os.getcwd()

    # Data  munging

    def import_data(path):
        df_imgs = pd.read_csv(path,
                                header=None,
                                names=['img_path'])
        df = pd.DataFrame(df_imgs['img_path'].str.split('/').to_list())
        df.rename(columns={2:'bodypart',3:'patient',4:'study',5:'img_path'}, 
                    inplace=True)
        df.drop(columns=[0,1],inplace=True)
        df['img_path'] = df_imgs['img_path']
        result_df = pd.DataFrame(df['study'].str.split('_').to_list())[1]
        df['result'] = result_df
        labels = np.where(df['result']=='positive',1,0)
        df['class'] = labels
        df['diagnosis'] = np.where(df['result']=='positive','abnormal','normal')
        return df

    # train and test dataset

    print("\nImport train and test data\n")
    train = import_data(data_path + 'train_image_paths.csv')
    test = import_data(data_path  + 'valid_image_paths.csv')
    
    def new_classes(df):
        new_class_normal_dict = {'XR_WRIST':0,
                         'XR_HUMERUS':1,
                         'XR_ELBOW':2,
                         'XR_FINGER':3,
                         'XR_SHOULDER':4,
                         'XR_HAND':5,
                         'XR_FOREARM':6               
                         }
        new_class_abnormal_dict = {'XR_WRIST':7,
                         'XR_HUMERUS':8,
                         'XR_ELBOW':9,
                         'XR_FINGER':10,
                         'XR_SHOULDER':11,
                         'XR_HAND':12,
                         'XR_FOREARM':13               
                         }
        if df['diagnosis'] == 'normal':        
            df['new_class'] = new_class_normal_dict[df['bodypart']]
        if df['diagnosis'] == 'abnormal':
            df['new_class'] = new_class_abnormal_dict[df['bodypart']]
        return df

    def apply_new_classes(train, test):
        train = train.apply(new_classes, axis=1)
        test = test.apply(new_classes, axis=1)
        return train, test
    train, test = apply_new_classes(train, test)
    
    # train and validation set for each per bodypart
    def train_validation(df, percentage):
        # unique patients for each bodypart
        patients = df['patient'].unique()
        # percentage of patients for validation data
        idx_size = int(percentage * len(patients))
        idx = np.random.choice(len(patients), 
                            size=idx_size, 
                            replace=False)
        valid_patients = patients[idx]
        valid_flag = df['patient'].isin(valid_patients)
        # training and validation sets for this specific bodypart
        # based on the patients selected
        train = df[~valid_flag].reset_index()
        valid = df[valid_flag].reset_index()

        return train, valid
    
    train, valid = train_validation(train, 0.1)     
    
    # Data Generator
    if data_augmentation == "noaugment":
        print("No data augmentation")
        train_generator = ImageDataGenerator(
            rescale = 1./255
            )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )
    if data_augmentation == "light_augment":
        print("Doing a light data augmentation")
        train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                horizontal_flip = True
        )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )
    if data_augmentation == "deep_augment":
        print("Doing a light data augmentation")
        train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                fill_mode='nearest',
                horizontal_flip = True,
                vertical_flip=True,
                featurewise_center=True,
                featurewise_std_normalization=True,
        )
        valid_generator = ImageDataGenerator(
            rescale = 1./255
            )        
    # Per part of the body

    def generate_data(generator,
                      df):
        generated = generator.flow_from_dataframe(
            dataframe = df,
            directory = directory,
            x_col = 'img_path',
            y_col = 'new_class',
            class_mode = 'raw',
            batch_size = 32,
            seed = 34,
            target_size=(320,320)
        )
        return generated
    
    print("\nCreate training, validation and test set\n")
    train_generators = generate_data(train_generator,
                                     train)
    valid_generators = generate_data(valid_generator,
                                     valid)
    test_generators = generate_data(valid_generator,
                                    test)  

    print("Data munging completed")
    
    return train_generators, valid_generators, test_generators, test

def load_history(bodyparts, url):
    import json
   
    histories = {}
    for bodypart in bodyparts:
        url = url
        f = open(url)
        histories[bodypart] = json.load(f)
    return histories

def load_evaluate_model(bodypart, 
                        url, 
                        test, 
                        test_generators, 
                        train_generators, 
                        valid_generators):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import math
    import warnings
    warnings.filterwarnings("ignore")
    from tensorflow import keras
    import random, os
    import numpy as np
    import pandas as pd
    from glob import glob
    import gc
    # from skimage.transform import resize
    from tensorflow.keras import backend as K # Importing Keras backend (by default it is Tensorflow)
    from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPool2D # Layers to be used for building our model
    from tensorflow.keras.models import load_model, Model # The class used to create a model
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.random import set_seed

    from sklearn.metrics import f1_score, recall_score, precision_score

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    def clean_up(model):
        K.clear_session()
        del model
        gc.collect()
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
    
    print(bodypart, "Load model\n")         
    model = load_model(url) 
    eval_func(test[test['bodypart'] == bodypart]['class'], 
                model.predict(test_generators[bodypart]))
    test_loss, test_acc  = model.evaluate(test_generators[bodypart], verbose=0)
    _, train_acc  = model.evaluate(train_generators[bodypart], verbose=0)
    _, valid_acc  = model.evaluate(valid_generators[bodypart], verbose=0)
    print("test_loss: ", round(100 * test_loss, 2))
    print("test_acc: ", round(100 * test_acc, 2))
    print("train_acc: ", round(100 * train_acc, 2))
    print("valid_acc: ", round(100 * valid_acc, 2))
    print("Delete model")
    clean_up(model)
    
    
    
