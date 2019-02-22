## Ignore duplicated library
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob


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
from keras.layers import merge
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from util.HelperFunction import helperfunction as helper

## Read data from UCI 2014 Train directory
## Data is the masked gery input file

img_channel_num = 1

path_to_train_image_dir = "data/uci_dataset_2014_with_RGB_pics/TRAIN/"
path_to_validation_image_dir = "data/uci_dataset_2014_with_RGB_pics/VALIDATION"

train_image_list = None
train_species_list = []
for species_name in sorted(os.listdir(path_to_train_image_dir)):
    print(species_name)
    for file_name in sorted(os.listdir("/".join([path_to_train_image_dir, species_name]))):

        # Create input images
        full_path = "/".join([path_to_train_image_dir, species_name, file_name])

        # Fetch original training data
        im_pil_orig = Image.open(full_path)

        augmented_img_list = helper.gen_augmented_image_list(im_pil_orig)
        assert(len(augmented_img_list) > 0)


        # Put species name into list
        train_species_list.append(species_name)

        im_np = np.array(im_pil_orig)
        im_np = np.reshape(im_np, (1, img_channel_num, im_np.shape[0], im_np.shape[1]))
        train_image_list = helper.cascade_npdata(image_list=train_image_list, np_input=im_np)

        # TODO: we have to do data augmentation
        for augmented_img in augmented_img_list:
            train_species_list.append(species_name)

            im_np = np.array(augmented_img)
            im_np = np.reshape(im_np, (1, img_channel_num, im_np.shape[0], im_np.shape[1]))
            train_image_list = helper.cascade_npdata(image_list=train_image_list, np_input=im_np)

        print("train_image_list len: " + str(len(train_image_list)))

print(train_image_list.shape)

## Since the labels are textual, so we encode them categorically
y = LabelEncoder().fit(species_list).transform(species_list)
print(y.shape)


## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)
X = image_list.astype(float)


#X = MinMaxScaler().fit(X).transform(X)
#TODO: masked out scaler
#scalers = {}
#for i in range(X.shape[0]):
#    scalers[i] = MinMaxScaler()
#    #scalers[i] = MinMaxScaler(feature_range=(-1,1))
#    X[i, :, :] = scalers[i].fit_transform(X[i, :, :])


#X = image_list

print(X.shape)
#print(X)


## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation
from keras import utils as np_utils
y_cat = to_categorical(y)
print(y_cat.shape)

fold_size = 5
## retain class balances
sss = StratifiedShuffleSplit(n_splits=fold_size, test_size=0.2,random_state=12345)
sss_iter = iter(sss.split(X, y))

# Generate input shape
channel_num = 1
input_shape = (X.shape[1], X.shape[2], channel_num)
print("input shape: " + str(input_shape))

## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes
## https://keras.io/initializations/ ;glorot_uniform, glorot_normal, lecun_uniform, orthogonal,he_normal

model = Sequential()

model.add(Convolution2D(8, kernel_size=(7, 7), strides=(2, 2),
                        activation='relu',
                        padding='same',
                        input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(40, activation='softmax'))

## Error is measured as categorical crossentropy or multiclass logloss
## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

#model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ["accuracy"])

## Fitting the model on the whole training data with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=300)

history_loss = []
history_val_loss = []
history_acc = []
history_val_acc = []

for _ in range(0,fold_size):
    train_index, val_index = next(sss_iter)
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y_cat[train_index], y_cat[val_index]

    channel_num = 1
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], channel_num))
    x_val   = np.reshape(x_val,   (x_val.shape[0],   x_val.shape[1],   x_val.shape[2],   channel_num))
    input_shape = (x_train.shape[1], x_train.shape[2], channel_num)

    history = model.fit(x_train, y_train,batch_size=10,epochs=5 ,verbose=1,
                    validation_data=(x_val, y_val),callbacks=[early_stopping])

    history_loss += history.history['loss']
    history_val_loss += history.history['val_loss']
    history_acc += history.history['acc']
    history_val_acc += history.history['val_acc']

## we need to consider the loss for final submission to leaderboard
## print(history.history.keys())
print('val_acc: ',max(history_val_acc))
print('val_loss: ',min(history_val_loss))
print('train_acc: ',max(history_acc))
print('train_loss: ',min(history_loss))

print()
print("train/val loss ratio: ", min(history_loss)/min(history_val_loss))

## summarize history for loss
## Plotting the loss with the number of iterations
plt.semilogy(history_loss)
plt.semilogy(history_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.jpg')
#plt.show()

## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.clf()
plt.plot(history_acc)
plt.plot(history_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.jpg')
#plt.show()

## print run time
end = time.time()
print(round((end-start),2), "seconds")


