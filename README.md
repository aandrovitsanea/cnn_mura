# X-Ray classification


## TO-DOS

* Extract metrics
* Check other architectures
* Fine tune base model with Keras Tuner
* Other models for transfer learning
* Create py files for densenet and resnet models and build them to the notebooks
* Add cleaned notebooks to repo
* Write report

This project trains and evaluates different CNN architectures with the aim of image classification of the MURA dataset.

MURA is a dataset of musculoskeletal radiographs consisting of studies from 11,967 patients, with a total of 40,005 multi-view radiographic images. Each belongs to one of seven standard upper extremity radiographic study types: 
* elbow, 
* finger, 
* forearm, 
* hand,
* humerus, 
* shoulder, and 
* wrist. 

Each study is manually labeled as **normal** or **abnormal** by radiologists at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012.

The **outcome** of the project is to feed the model with an image of an X-Ray and get as output the wether the patient has a normal or abnormal finding.

## Exploratory Data Analysis (EDA)

The dataset is already split in training and validation set. Training set consists of
36,808 images (92 %) and the validation set of 3,197 images (8 %). The distribution of these images to the seven types is not homogeneous, since i.e. in the training set we find more X-rays attributed to the shoulder and wrist and very few humerus or forearm X-rays. 

In all cases of the training set the negative/normal condition of the region are more then the positive/abnormal. At the validation set there are two cases (finger and forearm) where the abnormal diagnosis outcomes the normal.

![image](plots/barplot_per_bodypart_train.png)

More insight to the data can be found in this [notebook](notebooks/EDA.ipynb).

## Architectures

## Evaluation


## Getting Started

## Requirements:


### Jupyter Notebook

All [notebooks](notebooks/) were developed in Python 3.8.10.

### Scripts

All scripts of [models](models/) and [libraries](lib/)were developed in Python 3.7.3.


To install all packages in the version used here:

```
pip install -r requirements.txt
```


## Use application

You can use the application as following:

```
git clone git@github.com:aandrovitsanea/cnn_mura.git
cd cnn_mura
```
Then either take an ipython shell or open a jupyter notebook in there.

```
import lib.prediction as predict

# If dealing with the binary problem

prediction = predict.calculate_binary(name_code,
                                        part_body,
                                        image_url)
                                
# If you don't know the category the x ray belongs to

prediction = predict.calculate_14cls(name_code,
                                        part_body,
                                        image_url)

```
`name_code`, `part_body` and `image_url` params must be passed as strings.

Example:

```
pred = predict.calculate_binary('densenet_model_top_70epochs_deep_augment',
                                'XR_HUMERUS',
                                'data/MURA-v1.1/valid/XR_HUMERUS/patient11641/study2_positive/image1.png')

Output: X-ray of XR_HUMERUS is abnormal.
``` 


```
pred = predict.calculate_14cls('resnet_all_parts_no_augment_50epochs',
                                'all_parts',
                                'data/MURA-v1.1/valid/XR_HUMERUS/patient11641/study2_positive/image1.png'))

Output: XR_HUMERUS is abnormal.



