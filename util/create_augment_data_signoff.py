import os
import re
import random
from PIL import Image
from shutil import copyfile
import numpy as np
from util.HelperFunction import helperfunction as helper

img_channel_num = 1

path_to_train_image_dir = "data/uci_dataset_2014_with_RGB_pics/GREY_MASKED"
path_to_train_augment_image_dir = "data/uci_dataset_2014_with_RGB_pics/GREY_MASKED_AUGMENT"

train_image_list = None
train_species_list = []

if os.path.isdir(path_to_train_image_dir) is not True:
    os.makedirs(path_to_train_augment_image_dir)

for species_name in sorted(os.listdir(path_to_train_image_dir)):

    if os.path.isdir("/".join([path_to_train_augment_image_dir, species_name])) is not True:
        os.makedirs("/".join([path_to_train_augment_image_dir, species_name]))

    for file_name in sorted(os.listdir("/".join([path_to_train_image_dir, species_name]))):

        # Create input images
        full_path = "/".join([path_to_train_image_dir, species_name, file_name])
        print(full_path)

        # Fetch original training data
        im_pil_orig = Image.open(full_path)

        augmented_img_list = helper.gen_augmented_image_list(im_pil_orig)
        assert(len(augmented_img_list) > 0)

        file_original = re.sub('.JPG$', '', file_name)

        id = 0
        file_dst_name = file_original + "_" + str(id) + ".JPG"
        dst_path = "/".join([path_to_train_augment_image_dir, species_name, file_dst_name])
        #print(dst_path)
        #id = id + 1
        #im_pil_orig.save(dst_path)

        # TODO: we have to do data augmentation
        for augmented_img in augmented_img_list:
            file_dst_name = file_original + "_" + str(id) + ".JPG"
            dst_path = "/".join([path_to_train_augment_image_dir, species_name, file_dst_name])
            print(dst_path)
            augmented_img.save(dst_path)
            id = id + 1

