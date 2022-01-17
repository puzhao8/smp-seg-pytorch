import os, cv2
from pathlib import Path

import numpy as np
import tifffile as tiff 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class S1S2(BaseDataset):
    """ MTBS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unburn', 'burned']
    
    def __init__(
            self, 
            data_dir, 
            cfg, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.cfg = cfg
        self.band_index_dict = self.get_band_index_dict()
        print(self.band_index_dict)

        # images_dir
        data_dir = Path(data_dir)
        masks_dir = data_dir / "mask" / "poly"
        print("data_dir: ", data_dir)

        def get_fps(sat):
            # image and mask dir
            pre_dir = data_dir / sat / "pre"
            post_dir = data_dir / sat / "post"
            
            # fps
            ids = sorted(os.listdir(post_dir)) # modified on Jan-09
            pre_fps = [os.path.join(pre_dir, image_id) for image_id in ids]
            post_fps = [os.path.join(post_dir, image_id) for image_id in ids]
            return pre_fps, post_fps, ids

        self.fps_dict = {}
        for sat in self.cfg.DATA.SATELLITES:
            self.fps_dict[sat] = get_fps(sat)
        # self.ids = self.fps_dict[sat][-1]
        self.ids = sorted(self.fps_dict[self.cfg.DATA.SATELLITES[0]][-1])
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # aug + preprocess
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        ''' read data '''
        image_list = []
        # for sat in sorted(self.fps_dict.keys()):
        for sat in self.cfg.DATA.SATELLITES: # modified on Jan-9
            post_fps = self.fps_dict[sat][1]
            image_post = tiff.imread(post_fps[i]) # C*H*W
            image_post = np.nan_to_num(image_post, 0)
            if sat in ['S1','ALOS']: image_post = self.normalize_sar(image_post)
            image_post = image_post[self.band_index_dict[sat],] # select bands

            if 'pre' in self.cfg.DATA.PREPOST:
                pre_fps = self.fps_dict[sat][0]
                image_pre = tiff.imread(pre_fps[i])
                image_pre = np.nan_to_num(image_pre, 0)
                if sat in ['S1','ALOS']: image_pre = self.normalize_sar(image_pre)
                image_pre = image_pre[self.band_index_dict[sat],] # select bands
                
                if self.cfg.DATA.STACKING: # if stacking bi-temporal data
                    stacked = np.concatenate((image_pre, image_post), axis=0) 
                    image_list.append(stacked) #[x1, x2]
                else:
                    image_list += [image_pre, image_post] #[t1, t2]
            else:
                image_list.append(image_post) #[x1_t2, x2_t2]

        ''' read mask '''
        mask = tiff.imread(self.masks_fps[i])
        
        if 'poly' == self.cfg.DATA.REF_MASK:
            masks = [(mask == v) for v in self.class_values] # 1~6
            mask = np.stack(masks, axis=0).astype('float32')
            image_list.append(mask)

        # # apply augmentations
        # if self.augmentation:
        #     # sample = self.augmentation(image=image, mask=mask)
        #     # image, mask = sample['image'], sample['mask']
        #     image_list = [self.augmentation(image=image.transpose(1,2,0))['image'].transpose(2,0,1) for image in image_list]
        
        # # apply preprocessing
        # if self.preprocessing:
        #     # sample = self.preprocessing(image=mask, mask=mask)
        #     # image, mask = sample['image'], sample['mask']
        #     image_list = [self.preprocessing(image=image.transpose(1,2,0))['image'].transpose(2,0,1) for image in image_list]

        # return tuple(image_list)
        return (tuple(image_list[:-1]), image_list[-1])
        
    def __len__(self):
        return len(self.ids)

    def get_band_index_dict(self):
        ALL_BANDS = self.cfg.DATA.ALL_BANDS
        INPUT_BANDS = self.cfg.DATA.INPUT_BANDS

        def get_band_index(sat):
            all_bands = list(ALL_BANDS[sat])
            input_bands = list(INPUT_BANDS[sat])

            band_index = []
            for band in input_bands:
                band_index.append(all_bands.index(band))
            return band_index

        band_index_dict = {}
        for sat in ['ALOS', 'S1', 'S2']:
            band_index_dict[sat] = get_band_index(sat)
        
        return band_index_dict

    def normalize_sar(self, img):
        return (np.clip(img, -30, 0) + 30) / 30

class MTBS(BaseDataset):
    """ MTBS Dataset. Read images, apply augmentation and preprocessing transformations.
    
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
            
        # return image, mask
        return image_pre, image_post, mask # edited on Setp. 6
        
    def __len__(self):
        return len(self.ids)



