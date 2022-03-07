

import tifffile as tiff
import os
from pathlib import Path
import matplotlib.pyplot as plt
from imageio import imread, imsave

dataPath = Path("/home/p/u/puzhao/wildfire-progression-dataset/CA_2021_Kamloops")
savePath = dataPath / "outputs"
# %%
mask = tiff.imread(dataPath / "mask" / "poly.tif")
# plt.imshow(mask)
print(mask.shape)

# %%
# S1 = tiff.imread(dataPath / "S1" / "20210625T14_DSC13.tif")
# print(S1.shape)

# S1 = tiff.imread(dataPath / "S2" / "20210629T19_S2.tif")
# print(S1.shape)

# imsave(savePath / "test.png", mask)


#
tau = 0.1
fileList = sorted(os.listdir(dataPath / "S2"))
img_pre = tiff.imread(dataPath / "S2" / fileList[0])
for filename in fileList[1:]:
    print(filename)
    img_post = tiff.imread(dataPath / "S2" / filename)
    dnbr = (img_post - img_pre) > tau
    imsave(savePath / "S2" / f"{filename[:-4]}.png", dnbr.astype(float))
 
