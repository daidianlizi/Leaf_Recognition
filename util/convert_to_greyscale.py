import os
from PIL import Image
from pathlib import Path

## Read data from UCI 2014 BW image files
path_to_RGB_dir = "data/uci_dataset_2014_with_RGB_pics/RGB"
path_to_GREY_dir = "data/uci_dataset_2014_with_RGB_pics/GREY"

for species_name in os.listdir(path_to_RGB_dir): #assuming gif
    if os.path.isdir("/".join([path_to_GREY_dir,species_name])) is not True:
        os.makedirs("/".join([path_to_GREY_dir,species_name]), exist_ok=True)
    
    for file_name in os.listdir("/".join([path_to_RGB_dir, species_name])):
        # Create input images
        full_RGB_path = "/".join([path_to_RGB_dir, species_name, file_name])
        full_GREY_path = "/".join([path_to_GREY_dir, species_name, file_name])
        
        img = Image.open(full_RGB_path).convert('L')
        print("write image: " + str(full_GREY_path))
        img.save(full_GREY_path)
        
        
        
