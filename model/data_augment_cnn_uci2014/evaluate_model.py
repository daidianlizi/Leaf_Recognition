## Ignore duplicated library
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import pickle

## Measure execution time, becaus Kaggle cloud fluctuates

import time
start = time.time()

## Importing standard libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

## Importing sklearn libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

## Keras Libraries for Neural Networks

from keras.models import Sequential
import keras.models
from keras.layers import merge
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from util.HelperFunction import helperfunction as helper

model = Sequential()

model = keras.models.load_model('Leaf_classification_model.h5')

## Read test data from UCI 2014 Train directory
## Data is the masked gery input file

img_channel_num = 1
path_to_test_image_dir = "data/uci_dataset_2014_with_RGB_pics/TEST"

# Only read in test data
test_image_list = None
test_species_list = []
for species_name in sorted(os.listdir(path_to_test_image_dir)):
    print(species_name)
    for file_name in sorted(os.listdir("/".join([path_to_test_image_dir, species_name]))):

        # Create input images
        full_path = "/".join([path_to_test_image_dir, species_name, file_name])

        # Fetch original training data
        im_pil_orig = Image.open(full_path)

        # Put species name into list
        test_species_list.append(species_name)

        im_np = np.array(im_pil_orig)
        im_np = np.reshape(im_np, (1, img_channel_num, im_np.shape[0], im_np.shape[1]))
        test_image_list = helper.cascade_npdata(image_list=test_image_list, np_input=im_np)

print(test_image_list.shape)

## Since the labels are textual, so we encode them categorically
test_y = LabelEncoder().fit(test_species_list).transform(test_species_list)
test_y_cat = to_categorical(test_y)


## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)
test_X = test_image_list.astype(float)

print("Finish read in test data")

# Evaluate model on test data
score, acc = model.evaluate(test_X, test_y_cat, batch_size=10, verbose=1)

print('Test score:', score)
print('Test accuracy:', acc)

## print run time
end = time.time()
print(round((end-start),2), "seconds")
