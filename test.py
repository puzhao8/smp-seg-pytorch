import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from prettyprinter import pprint

# img = tiff.imread(r"D:\wildfire-s1s2-dataset-ak-tiles\train\mask\poly/ak6069116074620190612_2_2.tif")

# print(img[np.newaxis,].shape)
# print(np.unique(img))

# plt.imshow(img, 'gray')
# plt.show()


import json

json_url = "D:\wildfire-s1s2-dataset-ak-tiles/train_test.json"
with open(json_url) as json_file:
    split_dict = json.load(json_file)
test_list = split_dict['test']['sarname']
pprint(test_list)

