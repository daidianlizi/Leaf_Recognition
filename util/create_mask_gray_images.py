import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

## Read data from UCI 2014 BW image files
path_to_BW_dir = "data/uci_dataset_2014_with_RGB_pics/BW"
path_to_GREY_dir = "data/uci_dataset_2014_with_RGB_pics/GREY"
path_to_GREY_MASKED_dir = "data/uci_dataset_2014_with_RGB_pics/GREY_MASKED"

if os.path.isdir(path_to_GREY_MASKED_dir) is not True:
    os.makedirs(path_to_GREY_MASKED_dir, exist_ok=True)
    
for bw_species_name in os.listdir(path_to_BW_dir):  # assuming gif
    if os.path.isdir("/".join([path_to_GREY_MASKED_dir, bw_species_name])) is not True:
        os.makedirs("/".join([path_to_GREY_MASKED_dir, bw_species_name]), exist_ok=True)
        
    for bw_file_name in os.listdir("/".join([path_to_BW_dir, bw_species_name])):
        
        # Create input images
        full_BW_path = "/".join([path_to_BW_dir, bw_species_name, bw_file_name])
        #file_original = os.path.splitext(bw_file_name)[0]
        file_original = re.sub('_B.TIFF$', '', bw_file_name)
        
        grey_species_name = bw_species_name
        grew_file_name = file_original + ".JPG"
        
        full_GREY_path = "/".join([path_to_GREY_dir, grey_species_name, grew_file_name])
        
        if os.path.isfile(full_GREY_path) is not True:
            continue

        img_bw = Image.open(full_BW_path).convert('1')
        img_grey = Image.open(full_GREY_path).convert('L')
        
        # create masked gray image
        img_grey_masked = img_grey.copy()
        img_grey_masked = Image.new('L', img_grey.size, color='black')
        img_grey_masked.paste(img_grey, mask=img_bw)

        full_GREY_MASK_path = "/".join([path_to_GREY_MASKED_dir, grey_species_name, grew_file_name])
        print("write image: " + str(full_GREY_MASK_path))
        img_grey_masked.save(full_GREY_MASK_path)
        