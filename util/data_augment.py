import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import random


## Read data from UCI 2014 BW image files
path_to_original_dir = "data/script_trial/original"
path_to_generated_dir = "data/script_trial/generated"


for file_name in os.listdir(path_to_original_dir):

    full_file_path = "/".join([path_to_original_dir, file_name])
    if os.path.isfile(full_file_path) is not True:
        continue
    
    img_orig = Image.open(full_file_path)
    
    print(img_orig.getbands())
    
    x_min, y_min = [0] * 2
    x_max, y_max = img_orig.size
    
    xBound, yBound = img_orig.size
    
    for x_min in range(0, xBound):
        if len([y for y in range(0, yBound) if img_orig.getpixel((x_min, y))!=0]) != 0:
            break

    for x_max in range(xBound-1, -1, -1):
        if len([y for y in range(0, yBound) if img_orig.getpixel((x_max, y))!=0]) != 0:
            break
            
    for y_min in range(0, yBound):
        if len([x for x in range(0, xBound) if img_orig.getpixel((x, y_min))!=0]) != 0:
            break

    for y_max in range(yBound-1, -1, -1):
        if len([x for x in range(0, xBound) if img_orig.getpixel((x, y_max))!=0]) != 0:
            break
    
    print(file_name + " " + ", ".join(str(_) for _ in [x_min, x_max, y_min, y_max]))
    
    # Assert no empty images in the dataset
    assert(x_min <= x_max)
    assert(y_min <= y_max)

    x_img_len = x_max - x_min + 1
    y_img_len = y_max - y_min + 1

    x_reg_len = xBound - x_img_len
    y_reg_len = yBound - y_img_len

    img_content = img_orig.copy()
    img_content = img_content.crop(box=[x_min, y_min, x_max, y_max])

    img_content.show()
    
    for idx in range(0, 4):
        x_index = random.randrange(x_reg_len)
        y_index = random.randrange(y_reg_len)
        assert(x_index < x_reg_len)
        assert(y_index < y_reg_len)

        # create black new image
        img_displaced = Image.new('L', img_orig.size, color='black')

        # paste the content images
        img_displaced.paste(img_content, box=[x_index, y_index])
        #img_displaced.show()

print("finish")
