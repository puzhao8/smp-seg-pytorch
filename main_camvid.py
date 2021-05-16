


import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run():
    torch.multiprocessing.freeze_support()
    print('torch.multiprocessing.freeze_support()')


if __name__ == "__main__":

    """ set data dir """
    DATA_DIR = Path('./data/CamVid/')

    x_train_dir = DATA_DIR / 'train'
    y_train_dir = DATA_DIR / 'trainannot'

    x_valid_dir = DATA_DIR / 'val'
    y_valid_dir = DATA_DIR / 'valannot'

    x_test_dir = DATA_DIR /  'test'
    y_test_dir = DATA_DIR / 'testannot'

    """ Data loader """
    from dataset.camvid import Dataset
    dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])


    """ visualize data """
    from utils.visualize import visualize
    if False:
        image, mask = dataset[4] # get some sample
        visualize(
            image=image, 
            cars_mask=mask.squeeze(),
        )

    """ Data Augumentation """
    # Augmentations
    from dataset.augument import get_training_augmentation, \
        get_validation_augmentation, get_preprocessing

    if False:
        #### Visualize resulted augmented images and masks
        augmented_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            augmentation=get_training_augmentation(), 
            classes=['car'],
        )

        # same image with different random transforms
        for i in range(3):
            image, mask = augmented_dataset[1]
            visualize(image=image, mask=mask.squeeze(-1))

    """ Model """
    import torch
    from torch.utils.data import DataLoader
    import smp

    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])


    """ Data Preparation """
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)



    """ train configuration """
    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, 10):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            # torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


