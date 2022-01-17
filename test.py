# # import numpy as np
# # import tifffile as tiff
# # import matplotlib.pyplot as plt
# # from prettyprinter import pprint

# # img = tiff.imread(r"D:\wildfire-s1s2-dataset-ak-tiles\train\mask\poly/ak6069116074620190612_2_2.tif")

# # print(img[np.newaxis,].shape)
# # print(np.unique(img))

# # plt.imshow(img, 'gray')
# # plt.show()


# # import json

# # json_url = "D:\wildfire-s1s2-dataset-ak-tiles/train_test.json"
# # with open(json_url) as json_file:
# #     split_dict = json.load(json_file)
# # test_list = split_dict['test']['sarname']
# # pprint(test_list)


# # cd /cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch ; 
# # singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 main_s1s2_unet.py



# # def get_band_index(ALL_BANDS, INPUT_BANDS):
# #     BAND_INDEX = []
# #     for band in INPUT_BANDS:
# #         BAND_INDEX.append(ALL_BANDS.index(band))
# #     return BAND_INDEX

# # ALL_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
# # INPUT_BANDS = ['B4', 'B8', 'B11']
# # print(get_band_index(ALL_BANDS, INPUT_BANDS))


# # post_url = "/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/US_2021_Dixie/S1/post/20210622T01_ASC137.tif"
# # post_image = tiff.imread(post_url).transpose(2,0,1) # C*H*W
# # post_image = (np.clip(post_image, -30, 0) + 30) / 30
# # post_image = np.nan_to_num(post_image, 0)
# # print(f"post: {post_image.shape}")
# # print(post_image.min(), post_image.max())

# import os
# from cv2 import imread
# # print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/test/S2/pre")))
# # print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/test/S2/post")))
# # print(len(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/test/S1/pre")))
# # print(len(os.listdir("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles/test/S1/post")))

# # print(set(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S2/post"))-\
# # set(set(os.listdir("/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/train/S1/post"))))





# # dict = {'S1': [], 'S2': []}
# # for k in sorted(dict.keys()):
# #     print(k)




# # import pandas as pd
# # from prettyprinter import pprint

# # d = {'a': 1,
# #      'c': {'a': 2, 'b': {'x': 5, 'y' : 10}},
# #      'd': [1, 2, 3]}

# # c = {
# #     'project': {'name': 'IGARSS-2022', 'entity': 'wildfire'},
# #     'data': {'dir': '/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles', 'name': 's1s2', 'satellites': ['S1', 'S2'], 'prepost': ['pre', 'post'], 'stacking': True, 'ALL_BANDS': {'S1': ['ND', 'VH', 'VV'], 'ALOS': ['ND', 'VH', 'VV'], 'S2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']}, 'INPUT_BANDS': {'S1': ['ND', 'VH', 'VV'], 'ALOS': ['ND', 'VH', 'VV'], 'S2': ['B4', 'B8', 'B12']}, 'useDataWoCAug': False, 'SEED': 42, 'random_state': 42, 'CLASSES': ['burned'], 'REF_MASK': 'poly', 'train_ratio': 0.9},
# #     'model': {'ARCH': 'distill_unet', 'TOPO': [16, 32, 64, 128], 'S2_PRETRAIN': '/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_distill_unet_S2_pretrain_B4812_20220108T164608/model.pth', 'DISTILL': True, 'LOSS_COEF': [0, 0], 'ACTIVATION': 'sigmoid', 'ENCODER': 'resnet18', 'ENCODER_WEIGHTS': 'imagenet', 'max_epoch': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'cross_domain_coef': 0, 'use_lr_scheduler': True, 'warmup_coef': 2, 'max_score': 0.1, 'save_interval': 5, 'verbose': True},
# #     'eval': {'patchsize': 512},
# #     'experiment': {'note': None, 'name': '${data.name}_${model.ARCH}_${data.satellites}_${experiment.note}', 'output': './outputs/run_${experiment.name}_${now:%Y%m%dT%H%M%S}'}
# # }

# # # df = pd.json_normalize(d, sep='.').to_dict(orient='records')[0]

# # df = pd.json_normalize(c, sep='.').to_dict(orient='records')[0]
# # # df.to_dict(orient='records')[0]

# # print(df)




# # import torch
# # from torchvision import datasets, transforms as T

# # transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# # dataset = datasets.ImageNet("/home/p/u/puzhao/tiny-imagenet-200", split="train", transform=transform)

# # means = []
# # stds = []
# # for img in (dataset):
# #     means.append(torch.mean(img))
# #     stds.append(torch.std(img))

# # mean = torch.mean(torch.tensor(means))
# # std = torch.mean(torch.tensor(stds))

# # import torchvision.transforms as transforms
# # import torchvision.datasets as datasets
# # import torchvision.models as models

# # traindir = ''
# # train_loader = torch.utils.data.DataLoader(
# #         datasets.ImageFolder(traindir, transforms.Compose([
# #             transforms.RandomSizedCrop(224),
# #             transforms.RandomHorizontalFlip(),
# #             transforms.ToTensor(),
# #             normalize,
# #         ])),



# from imageio import imsave
# import tifffile as tiff
# import matplotlib.pyplot as plt

# img = tiff.imread("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles/test_images/S2/post/CA_2019_BC_691.tif")
# img = img[[2,3,5],:,:].transpose(1,2,0)
# mean = img.mean()
# std = img.std()

# img_norm = (img - mean) / std
# vmin = img_norm.min()
# vmax = img_norm.max()
# img_norm_vis = (img_norm - vmin) / (vmax - vmin)
# print(img_norm.min(), img_norm.max())

# # plt.imshow(img)
# # plt.imshow(img_norm_vis)
# # plt.show()

# imsave("~/tmp/input_image.png", img)
# imsave("~/tmp/img_norm.png", img_norm)

# import torch
# import torch.nn as nn
# m = nn.Dropout3d(p=0.2)
# input = torch.rand(1,10,3,3)

# out = m(input)
# print(input)
# print(out)


# import tifffile as tiff
# import numpy as np
# import random
# from pathlib import Path

# from torch import random
# from dataset.augument import get_training_augmentation
# root_dir = Path("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles/test")


# augment = get_training_augmentation()

# img1 = tiff.imread(root_dir / 'S2' / 'post' / "CA_2019_AB_133_8_5.tif")[[2,3,5],]
# img2 = tiff.imread(root_dir / 'S1' / 'post' / "CA_2019_AB_133_8_5.tif")
# mask = tiff.imread(root_dir / 'mask' / 'poly' / "CA_2019_AB_133_8_5.tif")[np.newaxis,]


# # print(img1.shape)
# print(img1.shape, img2.shape, mask.shape)
# res = augment(image=img1.transpose(1,2,0), mask=mask.transpose(1,2,0))

# img_list = [img1, img2, img1]
# mask_list = [mask, mask, mask]

# # random.seed(0)
# np.random.seed(0)

# x = list(map(lambda image, mask: augment(image=image, mask=mask), img_list, mask_list))

# for key in ['image', 'mask']:
#     print(key)
#     print(np.mean(x[0][key]-x[-1][key]))
#     print(np.mean(mask-x[-1]['mask']))
#     print("-----")


# res1 = augment(image=img1.transpose(1,2,0), mask=mask.transpose(1,2,0)) 

# print(mask[0,].shape, res1["mask"].shape, res1["image"].shape)
# print(res['image']-res["mask"])
# # print(np.mean(res['image']-res["mask"]))
# print(np.mean(res['image']-res1["image"]))



import torch
import torch.nn.functional as F

input = torch.randn(1,2,3,3)
output = F.normalize(input, dim=1, p=2)

print(input)
print(output)