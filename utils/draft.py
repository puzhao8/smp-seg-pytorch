import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile as tiff

mask_dir = Path("G:\SAR4Wildfire_Dataset\Fire_Perimeters_US\Monitoring_Trend_in_Burn_Severity\MTBS_L8_Dataset\MTBS_US_toTiles\mask")

img = cv2.imread(str(mask_dir / "az3455711116020180427.tif"), 0)
img_ = tiff.imread(str(mask_dir / "az3455711116020180427.tif"))

print(img.shape, np.unique(img))
print(img_.shape, np.unique(img_))

plt.imshow(img_, cmap='gray')
plt.show()



# y_gt_cpu = predPatch.cpu().detach().numpy()

# import os
# from imageio import imread, imsave
# for b in range(y_gt_cpu.shape[0]): 
#     mask = y_gt_cpu[b,]

#     for c in range(mask.shape[0]):
#         print(b, c)

#         if not os.path.exists(f"batch_{b}"): os.makedirs(f"batch_{b}")
#         imsave(f"batch_{b}/class_{c}.png", mask[c,])


# mask = predPatch.squeeze().cpu().detach().numpy()

# mask = predLabel
# import os
# from imageio import imread, imsave
# for c in range(mask.shape[0]):
#     # print(b, c)
#     channel = mask[c,]
#     print(c, channel.min(), channel.max())

#     # if not os.path.exists(f"batch_{b}"): os.makedirs(f"batch_{b}")
#     imsave(f"class_{c}.png", mask[c,])