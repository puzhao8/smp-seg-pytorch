from ntpath import join
import os
import matplotlib.pyplot as plt
from imageio import imread, imsave
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tifffile as tiff
import smp



def inference(model, test_dir, test_id):
    # model.cpu()
    model.to("cuda")
    patchsize = 512
    NUM_CLASS = 2

    # if '.tif' == str(url)[-4:]: # tif
    #     img0 = tiff.imread(url)
    #     img = interval_95(np.nan_to_num(img0, 0)) * 255
    # elif '.png' == str(url)[-4:]: # png
    #     img = imread(url)
    pre_url = test_dir / "pre" / f"{test_id}.tif"
    post_url = test_dir / "post" / f"{test_id}.tif"
    # mask_url = test_dir / "mask" / f"{test_id}.tif"

    pre_image = tiff.imread(pre_url) 
    post_image = tiff.imread(post_url) 
    # mask = tiff.imread(mask_url) 

    img = np.concatenate((pre_image, post_image), axis=-1) # H * W * C
    # img = img.transpose(1,2,0) # H * W * C

    # S1
    img = (np.clip(img, -30, 0) + 30) / 30
    

    def zero_padding(arr, patchsize):
        # print("zero_padding patchsize: {}".format(patchsize))
        (h, w, c) = arr.shape
        pad_h = (1 + np.floor(h/patchsize)) * patchsize - h
        pad_w = (1 + np.floor(w/patchsize)) * patchsize - w

        arr_pad = np.pad(arr, ((0, int(pad_h)), (0, int(pad_w)), (0, 0)), mode='symmetric')
        return arr_pad


    input_patchsize = 2 * patchsize
    padSize = int(patchsize/2)

    H, W, C = img.shape
    img_pad0 = zero_padding(img, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
    img_pad = np.pad(img_pad0, ((padSize, padSize), (padSize, padSize), (0, 0)), mode='symmetric')

    # img_preprocessed = self.preprocessing_fn(img_pad)
    img_preprocessed = img_pad
    in_tensor = torch.from_numpy(img_preprocessed.transpose(2, 0, 1)).unsqueeze(0) # C * H * W

    (Height, Width, Channels) = img_pad.shape
    pred_mask_pad = np.zeros((Height, Width))
    prob_mask_pad = np.zeros((NUM_CLASS, Height, Width))
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)
            inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]

            # if self.cfg.ARCH == 'FCN':
            #     predPatch = self.model(inputPatch.type(torch.cuda.FloatTensor))['out']
            # else:
            inputPatch.to("cuda")
            predPatch = model.forward(inputPatch.type(torch.cuda.FloatTensor))
            # predPatch = torch.sigmoid(predPatch)

            predPatch = predPatch.squeeze().cpu().detach().numpy()#.round()
            predLabel = np.argmax(predPatch, axis=0).squeeze()

            pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predLabel[padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify
            prob_mask_pad[:, i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[:, padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

    pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape
    prod_mask = prob_mask_pad[:, padSize:padSize+H, padSize:padSize+W] # clip back to original shape

    return pred_mask, prod_mask

def gen_errMap(grouthTruth, preMap, save_url=False):
    errMap = np.zeros(preMap.shape)
    # errMap[np.where((OptREF==0) & (SARREF==0))] = 0
    errMap[np.where((grouthTruth==1) & (preMap==1))] = 1.0 # TP
    errMap[np.where((grouthTruth==1) & (preMap==0))] = 2.0 # FN, green
    errMap[np.where((grouthTruth==0) & (preMap==1))] = 3.0 # FP

    num_color = len(np.unique(errMap))
    # color_tuple = ([1,1,1], [0.6,0,0], [0,0.8,0], [1, 0.6, 0.6])
    color_tuple = ([1,1,1], [0.6,0,0], [1, 0.6, 0.6], [0,0.8,0])
    my_cmap = ListedColormap(color_tuple[:num_color])

    # plt.figure(figsize=(15, 15))
    # plt.imshow(errMap, cmap=my_cmap)

    if save_url:
        plt.imsave(save_url, errMap, cmap=my_cmap)

        # saveName = os.path.split(save_url)[-1].split('.')[0]
        # errMap_rgb = imread(save_url)
        # wandb.log({f"test_errMap/{saveName}": wandb.Image(errMap_rgb)})
    return errMap


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from utils.GeoTIFF import GeoTIFF
    geotiff = GeoTIFF()

    model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/best_model.pth")
    output_dir = Path("G:/PyProjects/smp-seg-pytorch/outputs/test_output_new2")
    output_dir.mkdir(exist_ok=True)

    data_dir = Path("D:\wildfire-s1s2-dataset-ak")
    test_dir = Path(f"{str(data_dir)}\S1")

    # load test_id list
    import json
    json_url = "D:\wildfire-s1s2-dataset-ak-tiles/train_test.json"
    with open(json_url) as json_file:
        split_dict = json.load(json_file)
    test_id_list = split_dict['test']['sarname']

    test_id_list = [
        'ak6657114321520190709_ASC50',
        'ak6431815217120190620_ASC65',
        'ak6576015988620190620_DSC73',
        'ak6677714771520170722_ASC94',
        'ak6490515328220190621_ASC36'

        'ak6340015503620190614_DSC102',
        'ak6340015503620190614_ASC36',

        'ak6431815217120190620_ASC65',
        'ak6431815217120190620_DSC102',
        'ak6431815217120190620_DSC131',

        # 'CA_2017_BC_1157_ASC64'
            ]

    def apply_model_on_event(test_id):
        SAT = os.path.split(test_dir)[-1]

        print(f"------------------> {test_id} <-------------------")

        predMask, probMask = inference(model, test_dir, test_id)

        print(f"predMask shape: {predMask.shape}, unique: {np.unique(predMask)}")
        print(f"probMask: [{probMask.min()}, {probMask.max()}]")

        # # mtbs_palette =  ["000000", "006400","7fffd4","ffff00","ff0000","7fff00"]
        # # [0,100/255,0]
        # mtbs_palette = [[0,100/255,0], [127/255,1,212/255], [1,1,0], [1,0,0], [127/255,1,0], [1,1,1]]

        plt.imsave(output_dir / f"{test_id}_predLabel.png", predMask, cmap='gray', vmin=0, vmax=1)

       
        if 'S1' == SAT:
            orbKeyLen = len(test_id.split("_")[-1]) + 1 
            event = test_id[:(len(test_id)-orbKeyLen)]
        else:
            event = test_id
        print(event)

         # read and save true labels
        if os.path.isfile(data_dir / "mask" / "poly" / f"{event}.tif"):
            _, _, trueLabel = geotiff.read(data_dir / "mask" / "poly" / f"{event}.tif")
            geotiff.save(output_dir / f"{test_id}_predLabel.tif", predMask[np.newaxis,]) 

            trueLabel = trueLabel.squeeze()
            plt.imsave(output_dir / f"{test_id}_trueLabel.png", trueLabel, cmap='gray', vmin=0, vmax=1)
            gen_errMap(trueLabel, predMask, save_url=output_dir / f"{test_id}_errMap.png")
        
    
    for test_id in test_id_list:
        apply_model_on_event(test_id)