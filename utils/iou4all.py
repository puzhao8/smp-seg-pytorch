

import io
import os, glob
import numpy as np
import torch
from pathlib import Path
from imageio import imread, imsave
import tifffile as tiff
import wandb

# from smp.utils import train

def IoU_score(gt, pr, eps=1e-3):
    intersection = np.sum(gt * pr) # TP
    union = np.sum(gt) + np.sum(pr) - intersection + eps

    iou = intersection / union
    f1 = 2 * intersection / (intersection + union)
    return [iou, f1, intersection, union]


################################################
###### MTBS Two-Class IoU and F1-Score #######
################################################

def compute_IoU_F1(phase, result_dir, dataset_dir):
    event_dir = Path(f"{dataset_dir}/{phase}/S2/post")

    eventList = [filename[:-4] for filename in os.listdir(event_dir)]
    eventList = sorted(eventList)
    # print(len(eventList))

    if False:
        ss = np.random.randint(0, len(eventList),len(eventList))
        eventList = [eventList[i] for i in ss]

    step = 16 #len(eventList) #, valid batch size

    results = []
    for i, event in enumerate(eventList):
        print()

        gt = imread(result_dir / f"{event}_gts.png")[:,:,0] / 255
        pr = imread(result_dir / f"{event}_pred.png")[:,:,0] / 255

        measure = IoU_score(gt, pr)
        results.append(measure)

        arr = np.array(results)
        # print(arr.shape)
        
        # method 1
        # IoU, F1, TP, FP, FN = np.sum(arr, axis=0)

        # total_IoU = TP / (TP + FP + FN + eps)
        # total_F1 = 2 * TP / (2 * TP + FP + FN + eps)

        # method 2
        total_iou, total_f1, total_intersection, total_union = np.sum(arr, axis=0)
        IoU = total_intersection / total_union
        F1 = 2 * total_intersection / (total_intersection + total_union)

        print(event)
        print(f"IoU: {measure[0]:.4f}, F1: {measure[1]:.4f}")
        # print("total: ", TP, FP, FN)

    N = arr.shape[0]
    print(f"================== total metrics on {phase} ====================")
    print(f"(dataset as a whole) IoU: {IoU:.4f}, F1: {F1:.4f}")
    print(f"(average across events) IoU: {total_iou/N:.4f}, avg_F1: {total_f1/N:.4f}")
    print()

    wandb.log({'final': {f'{phase}.IoU': IoU, f'{phase}.F1': F1}})

    if phase in ['train', 'test']:
        print("----> batch IoU <-----")
        batch_measure = []
        num_batches = len(eventList) // step
        print(num_batches)
        for i in range(0, 1 + num_batches):
            start = i * step
            end = (i + 1) * step

            if end > len(eventList):
                end = len(eventList)
            # print(i, end)

            total_iou, total_f1, total_intersection, total_union = np.sum(np.array(results)[start:end,], axis=0)
            IoU = total_intersection / total_union
            F1 = 2 * total_intersection / (total_intersection + total_union)

            batch_measure.append([IoU, F1])
            avg_batch_IoU, avg_batch_F1 = np.mean(np.array(batch_measure), axis=0)

            print(f"batch {i} [{start}:{end}], batchsize: {end-start}, avg_batch_IoU: {avg_batch_IoU:.4f}, avg avg_batch_F1: {avg_batch_F1:.4f}")

        wandb.log({'batch': {f'{phase}.avg_IoU': avg_batch_IoU, f'{phase}.avg_F1': avg_batch_F1}})


################################################
###### MTBS Multi-Class IoU and F1-Score #######
################################################

def mtbs_label_preprocess(label):
    label[label==0] = 1 # both 0 and 1 are unburned
    label[label==5] = 1 # treat 'greener' (5) as unburned
    label[label==6] = 0 # ignore 'cloud' (6)
    label = label - 1 # [4 classes in total: 0, 1, 2, 3], cloud: -1
    return label

def multiclass_IoU_F1(pred_dir, gts_dir, NUM_CLASS=4, phase='test_images'):

    # gts_dir = Path(f"{dataset_dir}/{phase}/mask/mtbs")
    TEST_MASK =  os.path.split(gts_dir)[-1]

    eventList = [filename[:-4] for filename in os.listdir(gts_dir)]
    eventList = sorted(eventList)

    # initialize a dict to store results
    results = {}
    for cls in range(0, NUM_CLASS):
        results[f'class{cls}'] = []
    
    # loop over all test events
    for event in eventList:
        print(event)

        gt = tiff.imread(gts_dir / f"{event}.tif")
        if 'mtbs' == TEST_MASK: gt = mtbs_label_preprocess(gt)
        pr = tiff.imread(pred_dir / f"{event}_pred.tif")
        
        # compute IoU and F1 for each class per event
        for cls in range(0, NUM_CLASS):
            cls_gt = (gt==cls).astype(float)
            cls_pr = (pr==cls).astype(float)

            measure = IoU_score(cls_gt, cls_pr)
            measure = measure + [cls_gt.sum()]
            results[f'class{cls}'].append(measure)

            print(f"class{cls} IoU: {measure[0]:.4f}, class{cls} F1: {measure[1]:.4f}")
        
        print()

    # compute IoU and F1 for each class over all test images
    class_IoUs = []
    class_F1s = []
    class_pixels = []
    for key in results.keys():
        arr = np.array(results[key]) # iou, f1, intersection, union, pixel number

        # method 2
        total_iou, total_f1, total_intersection, total_union, total_pixels = np.sum(arr, axis=0)
        IoU = total_intersection / total_union
        F1 = 2 * total_intersection / (total_intersection + total_union)

        N = arr.shape[0]
        print(f"-------------------- total metrics on {phase} -----------------------")
        print(f"(dataset as a whole) {key} IoU: {IoU:.4f}, {key} F1: {F1:.4f}")
        print(f"(average across events) {key} avg_IoU: {total_iou/N:.4f}, {key} avg_F1: {total_f1/N:.4f}")
        print()

        class_IoUs.append(IoU.round(4))
        class_F1s.append(F1.round(4))
        class_pixels.append(total_pixels)

        # log results into wandb for each class
        wandb.log({'final': {
                f'{phase}.IoU_{key}': IoU, 
                f'{phase}.F1_{key}': F1}
            })

    if NUM_CLASS == 2:
        wandb.log({'final': {
                f'{phase}.IoU': class_IoUs[-1], 
                f'{phase}.F1': class_F1s[-1]}
            })

    mIoU = np.array(class_IoUs).mean()

    # # Frequency-weighted IoU (FwIoU)
    class_frequency = np.array(class_pixels) / np.array(class_pixels).sum()
    FwIoU = (np.array(class_IoUs) * class_frequency).sum()

    print(f"class IoU: {np.array(class_IoUs).round(4)}")
    print(f"class frequency: {np.array(class_frequency).round(4)}")
    print(f"mIoU: {mIoU:.4f}, FwIoU: {FwIoU:.4f}")

    wandb.log({'final': {
            f'{phase}.mIoU': mIoU,
            f'{phase}.FwIoU': FwIoU
        }
    })


if __name__ == "__main__":

    # Fresh
    phase = 'test_images'
    wandb.init(project='wildfire', name='test-multi-class-IoU')

    # dataset_dir = "/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles"
    # result_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/outputs_igarss/run_s1s2_UNet_['S1']_modis_20220204T181119/errMap")
    # compute_IoU_F1(phase, result_dir, dataset_dir)


    # gts_dir = Path("/home/p/u/puzhao/wildfire-s1s2-dataset-us-tiles") / "test_images/mask/mtbs"
    # pred_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_UNet_['S2']_mtbs_20220227T233525_work/errMap")
    # multiclass_IoU_F1(pred_dir, gts_dir, NUM_CLASS=3)


    gts_dir = Path("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles") / "test_images/mask/poly"
    pred_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/Canada_RSE_2022/run_poly_UNet_['S1']_EF_20220308T000802/errMap")
    multiclass_IoU_F1(pred_dir, gts_dir, NUM_CLASS=2)


    wandb.finish()