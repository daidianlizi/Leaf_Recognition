import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

## Read data from UCI 2014 BW image files
path_to_original_dir = "data/script_trial/original"
path_to_generated_dir = "data/script_trial/generated"

for file_name in os.listdir(path_to_original_dir):

    full_file_path = "/".join([path_to_original_dir, file_name])
    full_gen_path = "/".join([path_to_generated_dir, file_name])

    if os.path.isfile(full_file_path) is not True:
        continue

    #img_orig = Image.open(full_file_path).convert('L')

    img_orig = cv2.imread(full_file_path, 0)
    if img_orig is None:
        continue


    img_np = np.float32(img_orig)
    dst = cv2.cornerHarris(img_np, blockSize=9, ksize=3, k=0.04)

    img_dst = Image.fromarray(dst).convert('L')
    img_dst.save(full_gen_path)
    #img_dst = img_dst.convert('L')
    #plt.plot()
    #plt.imshow(img_dst)
    #plt.imsave("dst.jpg", img_dst)

    #cv2.imshow('orig', img_orig)

print("finish")

