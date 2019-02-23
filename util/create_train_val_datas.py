import os
import re
import random
from PIL import Image
from shutil import copyfile

## Read data from UCI 2014 BW image files
path_to_GREY_MASKED_dir = "data/uci_dataset_2014_with_RGB_pics/GREY_MASKED"
path_to_TRAIN_dir = "data/uci_dataset_2014_with_RGB_pics/TRAIN"
path_to_TEST_dir = "data/uci_dataset_2014_with_RGB_pics/TEST"

if os.path.isdir(path_to_TRAIN_dir) is not True:
    os.makedirs(path_to_TRAIN_dir)

if os.path.isdir(path_to_TEST_dir) is not True:
    os.makedirs(path_to_TEST_dir)

for species_name in os.listdir(path_to_GREY_MASKED_dir):  # assuming gif

    file_list = os.listdir("/".join([path_to_GREY_MASKED_dir, species_name]))

    # randomly shuffle the file lists
    random.shuffle(file_list)

    test_size = 0.2
    test_case_num = int(len(file_list) * test_size)
    train_case_num = len(file_list) - test_case_num

    assert(len(file_list)>=train_case_num>=0)
    assert(len(file_list)>=test_case_num>=0)

    train_species_dir_path = "/".join([path_to_TRAIN_dir, species_name])
    if os.path.isdir(train_species_dir_path) is not True:
        os.makedirs(train_species_dir_path)

    test_species_dir_path   = "/".join([path_to_TEST_dir, species_name])
    if os.path.isdir(test_species_dir_path) is not True:
        os.makedirs(test_species_dir_path)

    maxsize = (360, 480)

    for train_file_name in file_list[0:train_case_num]:
        src_train_file_path = "/".join([path_to_GREY_MASKED_dir, species_name, train_file_name])
        dst_train_file_path = "/".join([path_to_TRAIN_dir      , species_name, train_file_name])

        img = Image.open(src_train_file_path)
        assert(img is not None)
        img.thumbnail(maxsize, Image.ANTIALIAS)
        img.save(dst_train_file_path)
        print(src_train_file_path + str(img.size))

        #copyfile(src_train_file_path, dst_train_file_path)

    for test_file_name in file_list[train_case_num:]:
        src_test_file_path = "/".join([path_to_GREY_MASKED_dir, species_name, test_file_name])
        dst_test_file_path = "/".join([path_to_TEST_dir,  species_name, test_file_name])

        img = Image.open(src_test_file_path)
        img.thumbnail(maxsize, Image.ANTIALIAS)
        img.save(dst_test_file_path)
        print(src_test_file_path + str(img.size))
