# import numpy as np
# import tifffile as tiff
# import matplotlib.pyplot as plt
# from prettyprinter import pprint

# img = tiff.imread(r"D:\wildfire-s1s2-dataset-ak-tiles\train\mask\poly/ak6069116074620190612_2_2.tif")

# print(img[np.newaxis,].shape)
# print(np.unique(img))

# plt.imshow(img, 'gray')
# plt.show()


# import json

# json_url = "D:\wildfire-s1s2-dataset-ak-tiles/train_test.json"
# with open(json_url) as json_file:
#     split_dict = json.load(json_file)
# test_list = split_dict['test']['sarname']
# pprint(test_list)


# cd /cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch ; 
# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 main_s1s2_unet.py



# def get_band_index(ALL_BANDS, INPUT_BANDS):
#     BAND_INDEX = []
#     for band in INPUT_BANDS:
#         BAND_INDEX.append(ALL_BANDS.index(band))
#     return BAND_INDEX

# ALL_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
# INPUT_BANDS = ['B4', 'B8', 'B11']
# print(get_band_index(ALL_BANDS, INPUT_BANDS))


# post_url = "/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/US_2021_Dixie/S1/post/20210622T01_ASC137.tif"
# post_image = tiff.imread(post_url).transpose(2,0,1) # C*H*W
# post_image = (np.clip(post_image, -30, 0) + 30) / 30
# post_image = np.nan_to_num(post_image, 0)
# print(f"post: {post_image.shape}")
# print(post_image.min(), post_image.max())

import os
print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S2/pre")))
print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S2/post")))
print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S1/pre")))
print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S1/post")))

print(set(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S2/post"))-\
set(set(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S1/post"))))