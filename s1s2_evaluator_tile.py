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
from easydict import EasyDict as edict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from utils.GeoTIFF import GeoTIFF
# geotiff = GeoTIFF()

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

def get_band_index_dict(cfg):
    ALL_BANDS = cfg.data.ALL_BANDS
    INPUT_BANDS = cfg.data.INPUT_BANDS

    def get_band_index(sat):
        all_bands = list(ALL_BANDS[sat])
        input_bands = list(INPUT_BANDS[sat])

        band_index = []
        for band in input_bands:
            band_index.append(all_bands.index(band))
        return band_index

    band_index_dict = {}
    for sat in ['S1', 'ALOS', 'S2']:
        band_index_dict[sat] = get_band_index(sat)
    
    return band_index_dict

def inference(model, test_dir, test_id, cfg):

    patchsize = cfg.eval.patchsize
    NUM_CLASS = len(list(cfg.data.CLASSES))
    # model.cpu()

    if torch.cuda.is_available():
        model.to("cuda")

    input_tensors = []
    for sat in cfg.data.satellites:
        
        post_url = test_dir / sat / "post" / f"{test_id}.tif"
        post_image = tiff.imread(post_url) # C*H*W
        post_image = post_image[cfg.band_index_dict[sat],] # select bands

        if sat in ['S1', 'ALOS']: post_image = (np.clip(post_image, -30, 0) + 30) / 30
        # post_image_pad = image_padding(post_image, patchsize)

        # img_preprocessed = self.preprocessing_fn(img_pad)
        post_image_tensor = torch.from_numpy(post_image).unsqueeze(0) # n * C * H * W
        
        if 'pre' in cfg.data.prepost:
            pre_url = test_dir / sat / "pre" / f"{test_id}.tif"
            pre_image = tiff.imread(pre_url)
            pre_image = pre_image[cfg.band_index_dict[sat],] # select bands 

            if sat in ['S1', 'ALOS']: pre_image = (np.clip(pre_image, -30, 0) + 30) / 30

            # pre_image_pad = image_padding(pre_image, patchsize)
            pre_image_tensor = torch.from_numpy(pre_image).unsqueeze(0) # n * C * H * W
        
            input_tensors.append((pre_image_tensor, post_image_tensor))

        else:
            input_tensors.append(post_image_tensor)

    ''' ------------> tile input data <---------- '''
    input_patchs = []
    for sat_tensor in input_tensors:
        post_patch = (sat_tensor[1]).type(torch.cuda.FloatTensor)
        if 'pre' in cfg.data.prepost: 
            pre_patch = (sat_tensor[0]).type(torch.cuda.FloatTensor)
            
            if cfg.data.stacking: 
                inputPatch = torch.cat([pre_patch, post_patch], dim=1) # stacked inputs
                input_patchs.append(inputPatch)
            else:
                input_patchs += [pre_patch, post_patch]
        else:
            input_patchs.append(post_patch)

    ''' ------------> apply model <--------------- '''
    if 'Paddle_unet' == cfg.model.ARCH:
        predPatch = model.forward(input_patchs[0])

    elif 'FuseUNet' in cfg.model.ARCH:
        predPatch, decoder_out = model.forward(input_patchs, False)

    elif 'cdc_unet' in cfg.model.ARCH:
        predPatch, decoder_out = model.forward(input_patchs, False)
    
    else:
        predPatch = model.forward(input_patchs)
    ''' ------------------------------------------ '''

    # predPatch = decoder_out[1].squeeze().cpu().detach().numpy()#.round()
    # predLabel = 1 - np.argmax(predPatch, axis=0).squeeze()
    
    predPatch = torch.sigmoid(predPatch)
    predPatch = predPatch.cpu().detach().numpy()#.round()
    if predPatch.shape[0] > 1:
        pred_mask = np.argmax(predPatch, axis=0).squeeze()
        probility_mask = predPatch[1,]

    else:
        pred_mask = np.round(predPatch.squeeze())
        probility_mask = predPatch.squeeze()
    
    return pred_mask, probility_mask

def gen_errMap(grouthTruth, preMap, save_url=False):
    errMap = np.zeros(preMap.shape)
    # errMap[np.where((OptREF==0) & (SARREF==0))] = 0
    errMap[np.where((grouthTruth==1) & (preMap==1))] = 1.0 # TP, dark red
    errMap[np.where((grouthTruth==1) & (preMap==0))] = 2.0 # FN, light red
    errMap[np.where((grouthTruth==0) & (preMap==1))] = 3.0 # FP, green

    num_color = int(1 + max(np.unique(errMap)))
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
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    data_dir = Path(cfg.data.dir) / "test"

    # orbKeyLen = len(test_id.split("_")[-1]) + 1 
    # event = test_id[:-orbKeyLen]
    event = test_id
    print(event)

    print(f"------------------> {test_id} <-------------------")

    predMask, probMask = inference(model, data_dir, event, cfg)

    print(f"predMask shape: {predMask.shape}, unique: {np.unique(predMask)}")
    print(f"probMask: [{probMask.min()}, {probMask.max()}]")

    # # mtbs_palette =  ["000000", "006400","7fffd4","ffff00","ff0000","7fff00"]
    # # [0,100/255,0]
    # mtbs_palette = [[0,100/255,0], [127/255,1,212/255], [1,1,0], [1,0,0], [127/255,1,0], [1,1,1]]

    plt.imsave(output_dir / f"{test_id}_predLabel.png", predMask, cmap='gray', vmin=0, vmax=1)
    plt.imsave(output_dir / f"{test_id}_probMap.png", predMask, cmap='gray', vmin=0, vmax=1)

        # read and save true labels
    if os.path.isfile(data_dir / "mask" / "poly" / f"{event}.tif"):
        trueLabel = tiff.imread(data_dir / "mask" / "poly" / f"{event}.tif")
        # _, _, trueLabel = geotiff.read(data_dir / "mask" / "poly" / f"{event}.tif")
        # geotiff.save(output_dir / f"{test_id}_predLabel.tif", predMask[np.newaxis,]) 

        trueLabel = trueLabel.squeeze()
        # print(trueLabel.shape, predMask.shape)

        plt.imsave(output_dir / f"{test_id}_trueLabel.png", trueLabel, cmap='gray', vmin=0, vmax=1)
        gen_errMap(trueLabel, predMask, save_url=output_dir / f"{test_id}.png")


def evaluate_model(cfg, model_url, output_dir):

    # import json
    # json_url = Path(cfg.data.dir) / "train_test.json"
    # with open(json_url) as json_file:
    #     split_dict = json.load(json_file)
    # test_id_list = split_dict['test']['sarname']

    test_id_list = os.listdir(Path(cfg.data.dir) / "test" / "S2" / "post")
    test_id_list = [test_id[:-4] for test_id in test_id_list]
    print(test_id_list[0])

    model = torch.load(model_url, map_location=torch.device('cpu'))
    # output_dir = Path(SegModel.project_dir) / 'outputs'
    output_dir.mkdir(exist_ok=True)

    band_index_dict = get_band_index_dict(cfg)
    cfg = edict(cfg)
    cfg.update({"band_index_dict": band_index_dict})
    
    for test_id in test_id_list:
        apply_model_on_event(model, test_id, output_dir, cfg)



import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./config", config_name="s1s2_unet")
def run_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # wandb.init(config=cfg, project=cfg.project.name, name=cfg.experiment.name)
    wandb.init(config=cfg, project=cfg.project.name, entity=cfg.project.entity, name=cfg.experiment.name)

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

    run_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/outputs/run_s1s2_Paddle_unet_resnet18_['S1']_V0_20211225T005247")
    model_url = run_dir / "model.pth"
    output_dir = run_dir / "errMap_train"
    evaluate_model(cfg, model_url, output_dir)
    
    #########################################################################
    wandb.finish()

if __name__ == "__main__":
    
    run_app()


    # model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/best_model_s1s2.pth")
    # output_dir = Path(f"G:/PyProjects/smp-seg-pytorch/outputs/test_output_s1s2_")

    # for test_id in test_id_list:
    #     apply_model_on_event(model, test_id, output_dir, satellites=['S1', 'S2'])