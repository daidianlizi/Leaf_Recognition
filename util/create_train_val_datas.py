import os
import re
import random
from PIL import Image
from shutil import copyfile

## Read data from UCI 2014 BW image files
path_to_GREY_MASKED_dir = "data/uci_dataset_2014_with_RGB_pics/GREY_MASKED"
path_to_TRAIN_dir = "data/uci_dataset_2014_with_RGB_pics/TRAIN"
path_to_VALIDATION_dir = "data/uci_dataset_2014_with_RGB_pics/VALIDATION"

if os.path.isdir(path_to_TRAIN_dir) is not True:
    os.makedirs(path_to_TRAIN_dir, exist_ok=True)

if os.path.isdir(path_to_VALIDATION_dir) is not True:
    os.makedirs(path_to_VALIDATION_dir, exist_ok=True)

for species_name in os.listdir(path_to_GREY_MASKED_dir):  # assuming gif

    file_list = os.listdir("/".join([path_to_GREY_MASKED_dir, species_name]))

    # randomly shuffle the file lists
    random.shuffle(file_list)

    val_size = 0.2
    val_case_num = int(len(file_list) * val_size)
    train_case_num = len(file_list) - val_case_num

    assert(len(file_list)>=train_case_num>=0)
    assert(len(file_list)>=val_case_num>=0)

    try:
        train_species_dir_path = "/".join([path_to_TRAIN_dir, species_name])
        if os.path.isdir(train_species_dir_path) is not True:
            os.makedirs(train_species_dir_path, exist_ok=True)

        val_species_dir_path   = "/".join([path_to_VALIDATION_dir, species_name])
        if os.path.isdir(val_species_dir_path) is not True:
            os.makedirs(val_species_dir_path, exist_ok=True)

        for train_file_name in file_list[0:train_case_num]:
            src_train_file_path = "/".join([path_to_GREY_MASKED_dir, species_name, train_file_name])
            dst_train_file_path = "/".join([path_to_TRAIN_dir      , species_name, train_file_name])
            copyfile(src_train_file_path, dst_train_file_path)

        for val_file_name in file_list[train_case_num:]:
            src_val_file_path = "/".join([path_to_GREY_MASKED_dir, species_name, val_file_name])
            dst_val_file_path = "/".join([path_to_VALIDATION_dir,  species_name, val_file_name])
            copyfile(src_val_file_path, dst_val_file_path)
    except Exception as e:
        print(str(e))



