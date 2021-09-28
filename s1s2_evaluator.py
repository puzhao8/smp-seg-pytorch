from ntpath import join
import os
from cv2 import _InputArray_OPENGL_BUFFER
import matplotlib.pyplot as plt
from imageio import imread, imsave
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tifffile as tiff
import smp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from utils.GeoTIFF import GeoTIFF
geotiff = GeoTIFF()

import wandb

def image_padding(img, patchsize):
    def zero_padding(arr, patchsize):
        # print("zero_padding patchsize: {}".format(patchsize))
        (c, h, w) = arr.shape
        pad_h = (1 + np.floor(h/patchsize)) * patchsize - h
        pad_w = (1 + np.floor(w/patchsize)) * patchsize - w

        arr_pad = np.pad(arr, ((0, 0), (0, int(pad_h)), (0, int(pad_w))), mode='symmetric')
        return arr_pad

    padSize = int(patchsize/2)

    img_pad0 = zero_padding(img, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
    img_pad = np.pad(img_pad0, ((0, 0), (padSize, padSize), (padSize, padSize)), mode='symmetric')
    return img_pad

def inference(model, test_dir, test_id, cfg):

    patchsize = cfg.eval.patchsize
    NUM_CLASS = len(list(cfg.data.CLASSES)
    # model.cpu()
    model.to("cuda")
    test_id = test_id.split("_")[0]

    input_tensors = []
    for sat in cfg.data.satellites:
        
        post_url = test_dir / SAT / "post" / f"{test_id}.tif"
        post_image = tiff.imread(post_url) # C*H*W
        if sat in ['S1', 'ALOS']: post_image = (np.clip(post_image, -30, 0) + 30) / 30
        post_image_pad = image_padding(post_image, patchsize)

        # img_preprocessed = self.preprocessing_fn(img_pad)
        post_image_tensor = torch.from_numpy(post_image_pad).unsqueeze(0) # n * C * H * W
        
        if 'pre' in cfg.data.prepost:
            pre_url = test_dir / SAT / "pre" / f"{test_id}.tif"
            pre_image = tiff.imread(pre_url) 
            if sat in ['S1', 'ALOS']: pre_image = (np.clip(pre_image, -30, 0) + 30) / 30

            pre_image_pad = image_padding(pre_image, patchsize)
            pre_image_tensor = torch.from_numpy(pre_image_pad).unsqueeze(0) # n * C * H * W
        
            input_tensors.append((pre_image_tensor, post_image_tensor))

        else:
            input_tensors.append(post_image_tensor)

    C, H, W = post_image.shape
    _, _, Height, Width = input_tensors[0][0].shape
    pred_mask_pad = np.zeros((Height, Width))
    prob_mask_pad = np.zeros((NUM_CLASS, Height, Width))

    input_patchsize = 2 * patchsize
    padSize = int(patchsize/2) 
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)

            ''' ------------> tile input data <---------- '''
            input_patchs = []
            for sat_tensor in input_tensors:
                post_patch = (sat_tensor[1][..., i:i+input_patchsize, j:j+input_patchsize]).type(torch.cuda.FloatTensor)
                if 'pre' in cfg.data.prepost: 
                    pre_patch = (sat_tensor[0][..., i:i+input_patchsize, j:j+input_patchsize]).type(torch.cuda.FloatTensor)
                    
                    if cfg.data.stacking: 
                        inputPatch = torch.cat([pre_patch, post_patch], dim=1) # stacked inputs
                        input_patchs.append(inputPatch)
                    else:
                        input_patchs += [pre_patch, post_patch]
                else:
                    input_patchs.append(post_patch)

            ''' ------------> apply model <--------------- '''
            if 'Fuse' in cfg.model.ARCH:
                predPatch, _ = model.forward(input_patchs)
            else:
                predPatch = model.forward(inputPatch)
            ''' ------------------------------------------ '''

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

        saveName = os.path.split(save_url)[-1].split('.')[0]
        errMap_rgb = imread(save_url)
        wandb.log({f"test_errMap/{saveName}": wandb.Image(errMap_rgb)})
    return errMap


def apply_model_on_event(model, test_id, output_dir, cfg):

    output_dir.mkdir(exist_ok=True)
    data_dir = Path(cfg.data.dir) / "test_images"

    orbKeyLen = len(test_id.split("_")[-1]) + 1 
    event = test_id[:(len(test_id)-orbKeyLen)]
    print(event)

    print(f"------------------> {test_id} <-------------------")

    predMask, probMask = inference(model, data_dir, test_id, cfg)

    print(f"predMask shape: {predMask.shape}, unique: {np.unique(predMask)}")
    print(f"probMask: [{probMask.min()}, {probMask.max()}]")

    # # mtbs_palette =  ["000000", "006400","7fffd4","ffff00","ff0000","7fff00"]
    # # [0,100/255,0]
    # mtbs_palette = [[0,100/255,0], [127/255,1,212/255], [1,1,0], [1,0,0], [127/255,1,0], [1,1,1]]

    plt.imsave(output_dir / f"{test_id}_predLabel.png", predMask, cmap='gray', vmin=0, vmax=1)

        # read and save true labels
    if os.path.isfile(data_dir / "mask" / "poly" / f"{event}.tif"):
        _, _, trueLabel = geotiff.read(data_dir / "mask" / "poly" / f"{event}.tif")
        geotiff.save(output_dir / f"{test_id}_predLabel.tif", predMask[np.newaxis,]) 

        trueLabel = trueLabel.squeeze()
        # print(trueLabel.shape, predMask.shape)

        plt.imsave(output_dir / f"{test_id}_trueLabel.png", trueLabel, cmap='gray', vmin=0, vmax=1)
        gen_errMap(trueLabel, predMask, save_url=output_dir / f"{test_id}_errMap.png")



def evaluate_model(cfg, SegModel):

    import json
    json_url = Path(cfg.data.dir) / "train_test.json"
    with open(json_url) as json_file:
        split_dict = json.load(json_file)
    test_id_list = split_dict['test']['sarname']

    model = torch.load(SegModel.model_url)
    output_dir = Path(SegModel.project_dir) / 'outputs'
    output_dir.mkdir(exist_ok=True)

    for test_id in test_id_list:
        apply_model_on_event(model, test_id, output_dir, cfg)



import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./config", config_name="s1s2_fusion")
def run_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.init(config=cfg, project=cfg.project.name, name=cfg.experiment.name)
    # project_dir = Path(hydra.utils.get_original_cwd())
    #########################################################################

    # # load test_id list
    # import json
    # json_url = "D:\wildfire-s1s2-dataset-ak-tiles/train_test.json"
    # with open(json_url) as json_file:
    #     split_dict = json.load(json_file)
    # test_id_list = split_dict['test']['sarname']

    # model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/best_model_mse.pth")
    # output_dir = Path(f"G:/PyProjects/smp-seg-pytorch/outputs/test_output_mse")

    # for test_id in test_id_list:
    #     apply_model_on_event(model, test_id, output_dir, satellites=['S1', 'S2'])
    
    #########################################################################
    wandb.finish()

if __name__ == "__main__":
    
    run_app()


    # model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/best_model_s1s2.pth")
    # output_dir = Path(f"G:/PyProjects/smp-seg-pytorch/outputs/test_output_s1s2_")

    # for test_id in test_id_list:
    #     apply_model_on_event(model, test_id, output_dir, satellites=['S1', 'S2'])