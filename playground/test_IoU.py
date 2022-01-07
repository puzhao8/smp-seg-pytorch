import io
import os
import numpy as np
import torch
from pathlib import Path
from imageio import imread, imsave

# from smp.utils import train

def IoU_score(gt, pr, eps=1e-3):
    intersection = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - intersection + eps

    return intersection / union, intersection, union


# Fresh
# result_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_Paddle_unet_resnet18_['S1']_20211224T210446/errMap_tiles")
# event_dir = Path("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles/test/S2/post")

# V0
phase = 'test_images'
result_dir = Path(f"/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_UNet_resnet18_['S2']_TEST_20220107T175600/errMap")
event_dir = Path(f"/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles/{phase}/S2/post")

# /home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_Paddle_unet_resnet18_['S1']_V0_20211225T005247/errMap_train_tiles


eventList = [filename[:-4] for filename in os.listdir(event_dir)]
print(len(eventList))

if False:
    ss = np.random.randint(0, len(eventList),len(eventList))
    eventList = [eventList[i] for i in ss]

step = len(eventList) #, valid batch size

IS = []
UN = []
IoU = []
IoU_per10 = []
for i, event in enumerate(eventList):
    print()
    print(event)

    gt = imread(result_dir / f"{event}_gts.png")[:,:,0] / 255
    pr = imread(result_dir / f"{event}_pred.png")[:,:,0] / 255

    # print(gt.shape)
    # print(pr.shape)

    iou, intersection, union = IoU_score(gt, pr)
    print(intersection, union)
    print(f"IoU: {iou:.4f}")

    IS.append(intersection)
    UN.append(union)
    IoU.append(iou)

    mIoU = sum(IS) / sum(UN)
    aIoU = sum(IoU) / len(IoU)

    print(f"mIoU: {mIoU:.4f}, aIoU: {aIoU:.4f}")


print("----> batch IoU <-----")
group_IS = []
group_UN = []
group_IoU = []

for i in range(0, len(eventList), step):
    group_IS = sum(IS[i:i+step])
    group_UN = sum(UN[i:i+step])
    group_IoU.append(group_IS / group_UN)

    batch_mIoU = sum(group_IoU) / len(group_IoU)
    print(f"{i}, {len(group_IoU)}, batch_mIoU: {batch_mIoU:.4f}")