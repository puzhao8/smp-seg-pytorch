

import io
import os
import numpy as np
import torch
from pathlib import Path
from imageio import imread, imsave
import wandb

# from smp.utils import train

def IoU_score(gt, pr, eps=1e-3):
    intersection = np.sum(gt * pr) # TP
    union = np.sum(gt) + np.sum(pr) - intersection + eps

    # TP = np.sum(gt * pr)
    # # TN = np.sum(gt * (1-pr))
    # FP = np.sum((1 - gt) * pr)
    # FN = np.sum((1 - gt) * (1 - pr))
    
    # iou = TP / (TP + FP + FN + eps)
    # f1 = 2 * TP / (2 * TP + FP + FN + eps)
    # return iou, f1, TP, FP, FN

    iou = intersection / union
    f1 = 2 * intersection / (intersection + union)

    return iou, f1, intersection, union


def compute_IoU_F1(phase, result_dir, dataset_dir):
    event_dir = Path(f"{dataset_dir}/{phase}/S2/post")

    eventList = [filename[:-4] for filename in os.listdir(event_dir)]
    eventList = sorted(eventList)
    # print(len(eventList))

    if False:
        ss = np.random.randint(0, len(eventList),len(eventList))
        eventList = [eventList[i] for i in ss]

    step = 16 #len(eventList) #, valid batch size

    eps = 1e-3
    results = []
    for i, event in enumerate(eventList):
        print()

        gt = imread(result_dir / f"{event}_gts.png")[:,:,0] / 255
        pr = imread(result_dir / f"{event}_pred.png")[:,:,0] / 255

        measure = IoU_score(gt, pr, eps=eps)
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

if __name__ == "__main__":

    # Fresh
    phase = 'test_images'
    dataset_dir = "/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles"
    result_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_UNet_['S2']_allBands_20220107T222557/errMap")

    compute_IoU_F1(phase, result_dir, dataset_dir)