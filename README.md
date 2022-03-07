
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

scp -r /home/p/u/puzhao/smp-seg-pytorch/fcnn4cd/paddle_unet.py puzhao@alvis1.c3se.chalmers.se:/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/fcnn4cd



# Paddle Installation
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html

# change permission
<!-- https://linuxize.com/post/chmod-command-in-linux/ -->
chmod [OPTIONS] [ugoa…][-+=]perms…[,…] FILE...
chmod u=rwx,g=r,o= filename

u - The file owner.
g - The users who are members of the group.
o - All other users.
a - All users, identical to ugo.

-rw-r--r-- 12 linuxize users 12.0K Apr  8 20:51 filename.txt
|[-][-][-]-   [------] [---]
| |  |  | |      |       |
| |  |  | |      |       +-----------> 7. Group
| |  |  | |      +-------------------> 6. Owner
| |  |  | +--------------------------> 5. Alternate Access Method
| |  |  +----------------------------> 4. Others Permissions
| |  +-------------------------------> 3. Group Permissions
| +----------------------------------> 2. Owner Permissions
+------------------------------------> 1. File Type

chmod 744 /home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles
-rw------- (600) -- Only the user has read and write permissions.
-rw-r--r-- (644) -- Only user has read and write permissions; the group and others can read only.
-rwx------ (700) -- Only the user has read, write and execute permissions.
-rwxr-xr-x (755) -- The user has read, write and execute permissions; the group and others can only read and execute.
-rwx--x--x (711) -- The user has read, write and execute permissions; the group and others can only execute.
-rw-rw-rw- (666) -- Everyone can read and write to the file. Bad idea.
-rwxrwxrwx (777) -- Everyone can read, write and execute. Another bad idea.

r (read) = 4
w (write) = 2
x (execute) = 1
no permissions = 0

Owner: rwx=4+2+1=7
Group: r-x=4+0+1=5
Others: r-x=4+0+0=4

chmod a+rw 