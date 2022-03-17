#!/usr/bin/env python3

def data_loader(data_augmentation=False):
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
    print("No data augmentation")
    train_generator = ImageDataGenerator(
        rescale = 1./255,
    #     rotation_range = 30,
    #     horizontal_flip = True
        )
    valid_generator = ImageDataGenerator(
        rescale = 1./255
        )
    if data_augmentation==True:
        print("Doing data augmentation")
        train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                horizontal_flip = True
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
    return train, test, bodyparts_train_validation_dict, train_generators, valid_generators, test_generators
