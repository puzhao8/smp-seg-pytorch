from operator import index
import albumentations as albu
from matplotlib import transforms

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=1),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=256, width=256, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         # albu.channel_shuffle(),
        #         albu.ChannelDropout(channel_drop_range=(1,1), fill_value=0, p=1),
        #         albu.InvertImg(p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         # albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)




import torchvision.transforms as T
import matplotlib.pyplot as plt

def augment_data(imgs):
    '''
    imgs: list of array
    '''
    inputs = torch.cat(imgs, dim=0)
    channels_list = [im.shape[0] for im in imgs]
    idxs = [np.sum(np.array(channels_list[:i+1])) for i in range(0,len(channels_list))]
    idxs = [0] + idxs

    _transforms = T.Compose([
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=270),
        T.RandomResizedCrop(size=(256,256), scale=(0.2,1), interpolation=T.InterpolationMode.NEAREST),
        # T.ToTensor()
    ])

    input_aug = _transforms(inputs)
    outputs = [input_aug[idxs[i]:idxs[i+1]] for i in range(len(imgs))]
    return outputs


if __name__ == "__main__":

    from pathlib import Path
    import os
    import tifffile as tiff
    import torch
    import numpy as np

    train_dir = Path('/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles') / "test"
    fileList = os.listdir(train_dir / "mask" / "poly")
    
    filename = fileList[400]

    im1 = tiff.imread(train_dir / "S2" / "pre" / f"{filename}")[[5,3,2],] 
    im2 = tiff.imread(train_dir / "S2" / "post" / f"{filename}")[[5,3,2],] 
    mask = tiff.imread(train_dir / "mask" / "poly" / f"{filename}")

    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]

    im1 = np.nan_to_num(im1, 0)
    im2 = np.nan_to_num(im2, 0)

    min, max = im2.min(), im2.max()
    im1 = (im1 - min) / (max - min)
    im2 = (im2 - min) / (max - min)


    # plt.imsave("aug_org.png", im1.transpose(1,2,0))

    im1 = torch.from_numpy(im1)
    im2 = torch.from_numpy(im2)
    mask = torch.from_numpy(mask[np.newaxis,])

    imgs = [im1, im2, mask]

    # imgs = [(im*255).type(torch.uint8) for im in imgs]

    outputs = augment_data(imgs)
    print(type(outputs[0]))
    print(len(fileList))
    # print(np.sum((im1_ - im2_).type(torch.float16)))

    # print(im1_.shape, mask_.shape)
    # print(im1_ - mask_)

    cnt = 0
    fig, axs = plt.subplots(nrows=len(imgs), ncols=2, constrained_layout=True)
    for i in range(len(imgs)):
        img = imgs[i]
        img_ = outputs[i]

        if len(img.shape) >= 3:
            axs[i,0].imshow(img.numpy().transpose(1,2,0))
            axs[i,1].imshow(img_.numpy().transpose(1,2,0))
        else:
            axs[i,0].imshow(img.numpy())
            axs[i,1].imshow(img_.numpy())

    plt.savefig("aug.png")

    print(np.unique(im1))