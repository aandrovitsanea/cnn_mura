#!/usr/bin/env python3

# dictionary with name, body part leading to the url of the model to be loaded
dict_name_bodypart_model = {'base_model_no_augment_20epochs,XR_SHOULDER': 'models_logs/model_cnn_20epochs_noaugment_XR_SHOULDER.h5',
            'base_model_no_augment_20epochs,XR_FOREARM': 'models_logs/model_cnn_20epochs_noaugment_XR_FOREARM.h5',
            'base_model_no_augment_20epochs,XR_HUMERUS': 'models_logs/model_cnn_20epochs_noaugment_XR_HUMERUS.h5',
            'base_model_no_augment_20epochs,XR_WRIST': 'models_logs/model_cnn_20epochs_noaugment_XR_WRIST.h5',
            'base_model_no_augment_20epochs,XR_HAND': 'models_logs/model_cnn_20epochs_noaugment_XR_HAND.h5',
            'base_model_no_augment_20epochs,XR_FINGER': 'models_logs/model_cnn_20epochs_noaugment_XR_FINGER.h5',
            'base_model_no_augment_20epochs,XR_ELBOW': 'models_logs/densenet_model_top_20epochs_noaugment_XR_ELBOW.h5',
            'densenet_no_augment_20epochs,XR_SHOULDER': 'models_logs/densenet_model_top_20epochs_noaugment_XR_SHOULDER.h5',
            'densenet_no_augment_20epochs,XR_FOREARM': 'models_logs/densenet_model_top_20epochs_noaugment_XR_FOREARM.h5',
            'densenet_no_augment_20epochs,XR_HUMERUS': 'models_logs/densenet_model_top_20epochs_noaugment_XR_HUMERUS.h5',
            'densenet_no_augment_20epochs,XR_WRIST': 'models_logs/densenet_model_top_20epochs_noaugment_XR_WRIST.h5',
            'densenet_no_augment_20epochs,XR_HAND': 'models_logs/densenet_model_top_20epochs_noaugment_XR_HAND.h5',
            'densenet_no_augment_20epochs,XR_FINGER': 'models_logs/densenet_model_top_20epochs_noaugment_XR_FINGER.h5',
            'densenet_no_augment_20epochs,XR_ELBOW': 'models_logs/densenet_model_top_20epochs_noaugment_XR_ELBOW.h5',
            'resnet_no_augment_20epochs,XR_SHOULDER': 'models_logs/resnet_model_top_20epochs_noaugment_XR_SHOULDER.h5',
            'resnet_no_augment_20epochs,XR_FOREARM': 'models_logs/resnet_model_top_20epochs_noaugment_XR_FOREARM.h5',
            'resnet_no_augment_20epochs,XR_HUMERUS': 'models_logs/resnet_model_top_20epochs_noaugment_XR_HUMERUS.h5',
            'resnet_no_augment_20epochs,XR_WRIST': 'models_logs/resnet_model_top_20epochs_noaugment_XR_WRIST.h5',
            'resnet_no_augment_20epochs,XR_HAND': 'models_logs/resnet_model_top_20epochs_noaugment_XR_HAND.h5',
            'resnet_no_augment_20epochs,XR_FINGER': 'models_logs/resnet_model_top_20epochs_noaugment_XR_FINGER.h5',
            'resnet_no_augment_20epochs,XR_ELBOW': 'models_logs/resnet_model_top_20epochs_noaugment_XR_ELBOW.h5',
            'base_model_light_augment_20epochs,XR_SHOULDER': 'models_logs/model_cnn_20epochs_light_augment_XR_SHOULDER.h5',
            'base_model_light_augment_20epochs,XR_FOREARM': 'models_logs/model_cnn_20epochs_light_augment_XR_FOREARM.h5',
            'base_model_light_augment_20epochs,XR_HUMERUS': 'models_logs/model_cnn_20epochs_light_augment_XR_HUMERUS.h5',
            'base_model_light_augment_20epochs,XR_WRIST': 'models_logs/model_cnn_20epochs_light_augment_XR_WRIST.h5',
            'base_model_light_augment_20epochs,XR_HAND': 'models_logs/model_cnn_20epochs_light_augment_XR_HAND.h5',
            'base_model_light_augment_20epochs,XR_FINGER': 'models_logs/model_cnn_20epochs_light_augment_XR_FINGER.h5',
            'base_model_light_augment_20epochs,XR_ELBOW': 'models_logs/model_cnn_20epochs_light_augment_XR_ELBOW.h5',
            'densenet_light_augment_25epochs,XR_SHOULDER': 'models_logs/densenet_25epochs_light_augment_XR_SHOULDER.h5',
            'densenet_light_augment_25epochs,XR_FOREARM': 'models_logs/densenet_25epochs_light_augment_XR_FOREARM.h5',
            'densenet_light_augment_25epochs,XR_HUMERUS': 'models_logs/densenet_25epochs_light_augment_XR_HUMERUS.h5',
            'densenet_light_augment_25epochs,XR_WRIST': 'models_logs/densenet_25epochs_light_augment_XR_WRIST.h5',
            'densenet_light_augment_25epochs,XR_HAND': 'models_logs/densenet_25epochs_light_augment_XR_HAND.h5',
            'densenet_light_augment_25epochs,XR_FINGER': 'models_logs/densenet_25epochs_light_augment_XR_FINGER.h5',
            'densenet_light_augment_25epochs,XR_ELBOW': 'models_logs/densenet_25epochs_light_augment_XR_ELBOW.h5',
            'resnet_light_augment_20epochs,XR_SHOULDER': 'models_logs/resnet_20epochs_light_augment_XR_SHOULDER.h5',
            'resnet_light_augment_20epochs,XR_FOREARM': 'models_logs/resnet_20epochs_light_augment_XR_FOREARM.h5',
            'resnet_light_augment_20epochs,XR_HUMERUS': 'models_logs/resnet_20epochs_light_augment_XR_HUMERUS.h5',
            'resnet_light_augment_20epochs,XR_WRIST': 'models_logs/resnet_20epochs_light_augment_XR_WRIST.h5',
            'resnet_light_augment_20epochs,XR_HAND': 'models_logs/resnet_20epochs_light_augment_XR_HAND.h5',
            'resnet_light_augment_20epochs,XR_FINGER': 'models_logs/resnet_20epochs_light_augment_XR_FINGER.h5',
            'resnet_light_augment_20epochs,XR_ELBOW': 'models_logs/resnet_20epochs_light_augment_XR_ELBOW.h5',
            'base_model_deep_augment_20epochs,XR_SHOULDER': 'models_logs/model_cnn_20epochs_deep_augment_XR_SHOULDER.h5',
            'base_model_deep_augment_20epochs,XR_FOREARM': 'models_logs/model_cnn_20epochs_deep_augment_XR_FOREARM.h5',
            'base_model_deep_augment_20epochs,XR_HUMERUS': 'models_logs/model_cnn_20epochs_deep_augment_XR_HUMERUS.h5',
            'base_model_deep_augment_20epochs,XR_WRIST': 'models_logs/model_cnn_20epochs_deep_augment_XR_WRIST.h5',
            'base_model_deep_augment_20epochs,XR_HAND': 'models_logs/model_cnn_20epochs_deep_augment_XR_HAND.h5',
            'base_model_deep_augment_20epochs,XR_FINGER': 'models_logs/model_cnn_20epochs_deep_augment_XR_FINGER.h5',
            'base_model_deep_augment_20epochs,XR_ELBOW': 'models_logs/model_cnn_20epochs_deep_augment_XR_ELBOW.h5',
            'densenet_deep_augment_25epochs,XR_SHOULDER': 'models_logs/densenet_20epochs_deep_augment_XR_SHOULDER.h5',
            'densenet_deep_augment_25epochs,XR_FOREARM': 'models_logs/densenet_20epochs_deep_augment_XR_FOREARM.h5',
            'densenet_deep_augment_25epochs,XR_HUMERUS': 'models_logs/densenet_20epochs_deep_augment_XR_HUMERUS.h5',
            'densenet_deep_augment_25epochs,XR_WRIST': 'models_logs/densenet_20epochs_deep_augment_XR_WRIST.h5',
            'densenet_deep_augment_25epochs,XR_HAND': 'models_logs/densenet_20epochs_deep_augment_XR_HAND.h5',
            'densenet_deep_augment_25epochs,XR_FINGER': 'models_logs/densenet_20epochs_deep_augment_XR_FINGER.h5',
            'densenet_deep_augment_25epochs,XR_ELBOW': 'models_logs/densenet_20epochs_deep_augment_XR_ELBOW.h5',
            'resnet_deep_augment_20epochs,XR_SHOULDER': 'models_logs/resnet_20epochs_deep_augment_XR_SHOULDER.h5',
            'resnet_deep_augment_20epochs,XR_FOREARM': 'models_logs/resnet_20epochs_deep_augment_XR_FOREARM.h5',
            'resnet_deep_augment_20epochs,XR_HUMERUS': 'models_logs/resnet_20epochs_deep_augment_XR_HUMERUS.h5',
            'resnet_deep_augment_20epochs,XR_WRIST': 'models_logs/resnet_20epochs_deep_augment_XR_WRIST.h5',
            'resnet_deep_augment_20epochs,XR_HAND': 'models_logs/resnet_20epochs_deep_augment_XR_HAND.h5',
            'resnet_deep_augment_20epochs,XR_FINGER': 'models_logs/resnet_20epochs_deep_augment_XR_FINGER.h5',
            'resnet_deep_augment_20epochs,XR_ELBOW': 'models_logs/resnet_20epochs_deep_augment_XR_ELBOW.h5',
            'base_model_no_augment_100epochs,XR_SHOULDER': 'models_logs/model_cnn_100epochs_noaugment_XR_SHOULDER.h5',
            'base_model_no_augment_100epochs,XR_FOREARM': 'models_logs/model_cnn_100epochs_noaugment_XR_FOREARM.h5',
            'base_model_no_augment_100epochs,XR_HUMERUS': 'models_logs/model_cnn_100epochs_noaugment_XR_HUMERUS.h5',
            'base_model_no_augment_100epochs,XR_WRIST': 'models_logs/model_cnn_100epochs_noaugment_XR_WRIST.h5',
            'base_model_no_augment_100epochs,XR_HAND': 'models_logs/model_cnn_100epochs_noaugment_XR_HAND.h5',
            'base_model_no_augment_100epochs,XR_FINGER': 'models_logs/model_cnn_100epochs_noaugment_XR_FINGER.h5',
            'base_model_no_augment_100epochs,XR_ELBOW': 'models_logs/model_cnn_100epochs_noaugment_XR_ELBOW.h5',
            'densenet_no_augment_100epochs,XR_SHOULDER': 'models_logs/densenet_model_top_100epochs_noaugment_XR_SHOULDER.h5',
            'densenet_no_augment_100epochs,XR_FOREARM': 'models_logs/densenet_model_top_100epochs_noaugment_XR_FOREARM.h5',
            'densenet_no_augment_100epochs,XR_HUMERUS': 'models_logs/densenet_model_top_100epochs_noaugment_XR_HUMERUS.h5',
            'densenet_no_augment_100epochs,XR_WRIST': 'models_logs/densenet_model_top_100epochs_noaugment_XR_WRIST.h5',
            'densenet_no_augment_100epochs,XR_HAND': 'models_logs/densenet_model_top_100epochs_noaugment_XR_HAND.h5',
            'densenet_no_augment_100epochs,XR_FINGER': 'models_logs/densenet_model_top_100epochs_noaugment_XR_FINGER.h5',
            'densenet_no_augment_100epochs,XR_ELBOW': 'models_logs/densenet_model_top_100epochs_noaugment_XR_ELBOW.h5',
            'resnet_no_augment_100epochs,XR_SHOULDER': 'models_logs/resnet_model_top_100epochs_noaugment_XR_SHOULDER.h5',
            'resnet_no_augment_100epochs,XR_FOREARM': 'models_logs/resnet_model_top_100epochs_noaugment_XR_FOREARM.h5',
            'resnet_no_augment_100epochs,XR_HUMERUS': 'models_logs/resnet_model_top_100epochs_noaugment_XR_HUMERUS.h5',
            'resnet_no_augment_100epochs,XR_WRIST': 'models_logs/resnet_model_top_100epochs_noaugment_XR_WRIST.h5',
            'resnet_no_augment_100epochs,XR_HAND': 'models_logs/resnet_model_top_100epochs_noaugment_XR_HAND.h5',
            'resnet_no_augment_100epochs,XR_FINGER': 'models_logs/resnet_model_top_100epochs_noaugment_XR_FINGER.h5',
            'resnet_no_augment_100epochs,XR_ELBOW': 'models_logs/resnet_model_top_100epochs_noaugment_XR_ELBOW.h5',
            'xception_all_parts_no_augment_50epochs,all_parts': 'models_logs/xception_14classes_top_50epochs_deep_augment.h5',
            'xception_all_parts_deep_augment_50epochs,all_parts': 'models_logs/xception_14classes_top_50epochs_deep_augment.h5',
            'densenet_all_parts_deep_augment_50epochs,all_parts': 'models_logs/densenet_14classes_top_50epochs_deep_augment_fixed_softmax.h5',
            'densenet_all_parts_no_augment_50epochs,all_parts': 'models_logs/densenet_14classes_top_50epochs_no_augment_.h5',
            'base_model_all_parts_no_augment_50epochs,all_parts': 'models_logs/base_model_14classes_noaugmentation_50epochs.h5',
            'base_model_all_parts_deep_augment_50epochs,all_parts': 'models_logs/base_model_14classes_deep_augmentation_50epochs.h5',
            'inception_deep_augment_70epochs,XR_SHOULDER': 'models_logs/inception_model_top_70epochs_deep_augment_XR_SHOULDER.h5',
            'inception_deep_augment_70epochs,XR_FOREARM': 'models_logs/inception_model_top_70epochs_deep_augment_XR_FOREARM.h5',
            'inception_deep_augment_70epochs,XR_HUMERUS': 'models_logs/inception_model_top_70epochs_deep_augment_XR_HUMERUS.h5',
            'inception_deep_augment_70epochs,XR_WRIST': 'models_logs/inception_model_top_70epochs_deep_augment_XR_WRIST.h5',
            'inception_deep_augment_70epochs,XR_HAND': 'models_logs/inception_model_top_70epochs_deep_augment_XR_HAND.h5',
            'inception_deep_augment_70epochs,XR_FINGER': 'models_logs/inception_model_top_70epochs_deep_augment_XR_FINGER.h5',
            'inception_deep_augment_70epochs,XR_ELBOW': 'models_logs/inception_model_top_70epochs_deep_augment_XR_ELBOW.h5',
            'densenet_deep_augment_70epochs,XR_SHOULDER': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_SHOULDER.h5',
            'densenet_deep_augment_70epochs,XR_FOREARM': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_FOREARM.h5',
            'densenet_deep_augment_70epochs,XR_HUMERUS': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_HUMERUS.h5',
            'densenet_deep_augment_70epochs,XR_WRIST': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_WRIST.h5',
            'densenet_deep_augment_70epochs,XR_HAND': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_HAND.h5',
            'densenet_deep_augment_70epochs,XR_FINGER': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_FINGER.h5',
            'densenet_deep_augment_70epochs,XR_ELBOW': 'models_logs/densenet_model_top_70epochs_deep_augment_XR_ELBOW.h5',
            'resnet_deep_augment_70epochs,XR_SHOULDER': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_SHOULDER.h5',
            'resnet_deep_augment_70epochs,XR_FOREARM': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_FOREARM.h5',
            'resnet_deep_augment_70epochs,XR_HUMERUS': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_HUMERUS.h5',
            'resnet_deep_augment_70epochs,XR_WRIST': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_WRIST.h5',
            'resnet_deep_augment_70epochs,XR_HAND': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_HAND.h5',
            'resnet_deep_augment_70epochs,XR_FINGER': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_FINGER.h5',
            'resnet_deep_augment_70epochs,XR_ELBOW': 'models_logs/resnet_model_top_70epochs_deep_augment_XR_ELBOW.h5',
            'resnet_all_parts_deep_augment_50epochs,all_parts': 'models_logs/resnet_14classes_top_50epochs_deep_augment_fixed_softmax.h5',
            'resnet_all_parts_no_augment_50epochs,all_parts': 'models_logs/resnet_14classes_top_50epochs_noaugment_fixed_softmax.h5'}

def load_model(model_name, bodypart): # 'data/models/densenet_model_top_70epochs_deep_augment_XR_HUMERUS.h5'
    from tensorflow.keras.models import load_model
    url = dict_name_bodypart_model[model_name+','+bodypart]
    model = load_model(url)
    return model

def import_image(url):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    image = load_img(url, target_size=(320, 320))
    image = img_to_array(image)/255.0
    image = image.reshape((1, 
                        image.shape[0], 
                        image.shape[1], 
                        image.shape[2]))
    image = preprocess_input(image)
    
    return image


    
def calculate_binary(model_name, bodypart, image_url):
    model = load_model(model_name, bodypart)
    image = import_image(image_url)
    def predict_binary(image, bodypart):
        prediction = model.predict(image)
        if prediction > 0.5:
            print("X-ray of " + bodypart + " is abnormal.")
        else:
            print("X-ray of " + bodypart + " is normal.")
        return prediction
    prediction = predict_binary(image, bodypart)
    return prediction
    
def calculate_14cls(model_name, bodypart, image_url):
    model = load_model(model_name, bodypart)
    image = import_image(image_url)
    def predict_14cls(image):
        dict_14classes = {0: 'XR_WRIST is normal.',
                     1: 'XR_HUMERUS is normal.',
                     2: 'XR_ELBOW is normal.',
                     3: 'XR_FINGER is normal.',
                     4: 'XR_SHOULDER is normal.',
                     5: 'XR_HAND is normal.',
                     6: 'XR_FOREARM is normal.',
                     7: 'XR_WRIST is abnormal.',
                     8: 'XR_HUMERUS is abnormal.',
                     9: 'XR_ELBOW is abnormal.',
                     10: 'XR_FINGER is abnormal.',
                     11: 'XR_SHOULDER is abnormal.',
                     12: 'XR_HAND is abnormal.',
                     13: 'XR_FOREARM is abnormal.'
                     }

        prediction = model.predict(image)
        print(dict_14classes[prediction.argmax()])
        return prediction
    prediction = predict_14cls(image)
    return prediction
