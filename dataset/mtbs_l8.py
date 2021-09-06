import os, cv2
from pathlib import Path

import numpy as np
import tifffile as tiff 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['bgd', 'unburn', 'low', 'moderate', 'high', 'greener', 'cloud']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # images_dir
        images_dir = Path(images_dir)

        # image and mask dir
        pre_dir = images_dir / "pre"
        post_dir = images_dir / "post"
        masks_dir = images_dir / "mask"

        # fps
        self.ids = os.listdir(pre_dir)
        self.pre_fps = [os.path.join(pre_dir, image_id) for image_id in self.ids]
        self.post_fps = [os.path.join(post_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # aug + preprocess
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image_pre = tiff.imread(self.pre_fps[i])
        image_post = tiff.imread(self.post_fps[i])

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.concatenate((image_pre, image_post), axis=0)
        # image = image.transpose(1,2,0)
        mask = tiff.imread(self.masks_fps[i])
        mask[mask==0] = 1 # set background as unburned 1, added by Puzhao on June 2
        mask = mask.astype(float) - 1
                        
        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values] # 1~6
        # # mask = np.stack(masks, axis=-1).astype('float')
        # mask = np.stack(masks, axis=0).astype('float') # modified by puzhao on June 1st, 2021
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

