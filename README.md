# X-Ray classification

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

## TO-DOS



## Exploratory Data Analysis (EDA)

The dataset is already split in training and validation set. Training set consists of
36,808 images (92 %) and the validation set of 3,197 images (8 %). The distribution of these images to the seven types is not homogeneous, since i.e. in the training set we find more X-rays attributed to the shoulder and wrist and very few humerus or forearm X-rays. 

In the validation set we find more X-rays from the region of
humerus and wrist and less from the shoulder. 

In all cases of the training set the negative/normal condition of the region are more then the positive/abnormal. At the validation set there are two cases (finger and forearm) where the abnormal diagnosis outcomes the normal.

![image](plots/barplot_per_bodypart_train.png)

![image](plots/barplot_per_bodypart_train.png)

More insight to the data can be found in this [notebook](notebooks/EDA.ipynb).

## Architectures

## Evaluation


## Getting Started

This repo is managed with pipenv. 

### Environment Setup

Use [pyenv](https://github.com/pyenv/pyenv) to manage python interpreter versions, and [pipenv](https://pipenv.pypa.io/en/latest/) for dependency management. More described in this [blog post](https://hackernoon.com/reaching-python-development-nirvana-bb5692adf30c).

### Jupyter Notebooks


To run the jupyter notebook server:

```bash
pipenv install
pipenv shell
jupyter notebook
```



