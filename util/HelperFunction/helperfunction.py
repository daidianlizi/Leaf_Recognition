import numpy as np
import random
from PIL import Image

def cascade_npdata(image_list=None, np_input=None):

    if image_list is None:
        image_list = np_input.copy()
    else:
        try:
            image_list = np.concatenate((image_list, np_input))
        except ValueError as e:
            print(str(e))

    return image_list

def gen_augmented_image_list(plt_input=None):
    if plt_input is None:
        raise ValueError("Should input valid plt image object")

    ret_list = []

    img_orig = plt_input

    x_min, y_min = [0] * 2
    x_max, y_max = img_orig.size

    xBound, yBound = img_orig.size

    for x_min in range(0, xBound):
        if len([y for y in range(0, yBound) if img_orig.getpixel((x_min, y)) != 0]) != 0:
            break

    for x_max in range(xBound - 1, -1, -1):
        if len([y for y in range(0, yBound) if img_orig.getpixel((x_max, y)) != 0]) != 0:
            break

    for y_min in range(0, yBound):
        if len([x for x in range(0, xBound) if img_orig.getpixel((x, y_min)) != 0]) != 0:
            break

    for y_max in range(yBound - 1, -1, -1):
        if len([x for x in range(0, xBound) if img_orig.getpixel((x, y_max)) != 0]) != 0:
            break

    # Assert no empty images in the dataset
    assert (x_min <= x_max)
    assert (y_min <= y_max)

    x_img_len = x_max - x_min + 1
    y_img_len = y_max - y_min + 1

    x_reg_len = xBound - x_img_len
    y_reg_len = yBound - y_img_len

    img_content = img_orig.copy()
    img_content = img_content.crop(box=[x_min, y_min, x_max, y_max])

    for idx in range(0, 10):
        x_index = random.randrange(x_reg_len)
        y_index = random.randrange(y_reg_len)
        assert (x_index < x_reg_len)
        assert (y_index < y_reg_len)

        # create black new image
        img_displaced = Image.new('L', img_orig.size, color='black')

        # paste the content images
        img_displaced.paste(img_content, (x_index, y_index))

        ret_list.append(img_displaced)

    return ret_list
