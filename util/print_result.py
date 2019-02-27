## Ignore duplicated library
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import pickle

## Measure execution time, becaus Kaggle cloud fluctuates

## Importing standard libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


train_acc = pickle.load(open("GREY_NON_MASKED_30species.train_acc.log", 'rb'))
train_mask_acc = pickle.load(open("GREY_MASKED.train_acc.log", 'rb'))
train_mask_aug_acc = pickle.load(open("MASK_GREY_AUMENT.train_acc.log", 'rb'))

val_acc = pickle.load(open("GREY_NON_MASKED_30species.val_acc.log", 'rb'))
val_mask_acc = pickle.load(open("GREY_MASKED.val_acc.log", 'rb'))
val_mask_aug_acc = pickle.load(open("MASK_GREY_AUMENT.val_acc.log", 'rb'))

print(train_acc)
print(val_acc)


plt.plot(train_acc)
plt.plot(val_acc)
plt.plot(train_mask_acc)
plt.plot(val_mask_acc)
plt.plot(train_mask_aug_acc)
plt.plot(val_mask_aug_acc)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
