
# smp-seg-pytorch project
This project aims to train U-Net segementation models for large-scale burned area mapping.

``` python
main_s1s2_unet.py
main_s1s2_fuse_unet.py
main_mtbs.py
```


scp -r puzhao@alvis1.c3se.chalmers.se:/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/outputs/errMap D:\PyProjects\IGARSS-2022-S1S2\

scp -r D:\wildfire-s1s2-dataset-ca-2019-median-tiles-V1\test_images\* puzhao@alvis1.c3se.chalmers.se:/cephyr/NOBACKUP/groups/snic2021-7-104/wildfire-s1s2-dataset-ca-tiles/test_images

# SNIC to geoinfo-gpu
scp -r puzhao@alvis1.c3se.chalmers.se:/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/main_s1s2_fuse_unet_V1.py /home/p/u/puzhao/smp-seg-pytorch/main_s1s2_fuse_unet_V1.py