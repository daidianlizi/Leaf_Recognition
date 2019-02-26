import os
import re
import random
from PIL import Image
from shutil import copyfile
import numpy as np
import pickle
from util.HelperFunction import helperfunction as helper
img_channel_num = 1


path_to_train_augment_image_dir = "data/uci_dataset_2014_with_RGB_pics/BW"

train_image_list = None
train_species_list = []

for species_name in sorted(os.listdir(path_to_train_augment_image_dir)):
    for file_name in sorted(os.listdir("/".join([path_to_train_augment_image_dir, species_name]))):

        # Create input images
        full_path = "/".join([path_to_train_augment_image_dir, species_name, file_name])

        print(full_path)
        # Fetch original training data
        im_pil_orig = Image.open(full_path).convert('1')

        
        im_np = np.array(im_pil_orig)


        im_np = np.reshape(im_np, (1, img_channel_num, im_np.shape[0], im_np.shape[1]))
        print(full_path +" " + str(im_np.shape))

        train_image_list = helper.cascade_npdata(image_list=train_image_list, np_input=im_np)
        train_species_list.append(species_name)

train_image_list.tofile("trainX.data")

with open("trainY.data", "wb") as fp:
    pickle.dump(train_species_list, fp)
