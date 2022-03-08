from ntpath import join
import os
import random
from cv2 import _InputArray_OPENGL_BUFFER
import matplotlib.pyplot as plt
from imageio import imread, imsave
import torch
import numpy as np
from tqdm import tqdm

from pathlib import Path
import tifffile as tiff
from easydict import EasyDict as edict
from smp.base.modules import Activation

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
    ALL_BANDS = cfg.DATA.ALL_BANDS
    INPUT_BANDS = cfg.DATA.INPUT_BANDS

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

    patchsize = cfg.EVAL.PATCHSIZE
    # NUM_CLASS = cfg.MODEL.NUM_CLASSES
    # model.cpu()

    if torch.cuda.is_available():
        model.to("cuda")

    ''' read input data '''
    input_tensors = []  # [(S1_pre, S1_post), (S2_pre, S2_post), ...]
    for sat in cfg.DATA.SATELLITES:
        
        post_url = test_dir / sat / "post" / f"{test_id}.tif"
        post_image = tiff.imread(post_url) # C*H*W
        post_image = post_image[cfg.band_index_dict[sat],] # select bands

        if sat in ['S1', 'ALOS']: post_image = (np.clip(post_image, -30, 0) + 30) / 30
        post_image_pad = image_padding(post_image, patchsize)

        # img_preprocessed = self.preprocessing_fn(img_pad)
        post_image_tensor = torch.from_numpy(post_image_pad).unsqueeze(0) # n * C * H * W
        
        if 'pre' in cfg.DATA.PREPOST:
            pre_url = test_dir / sat / "pre" / f"{test_id}.tif"
            pre_image = tiff.imread(pre_url)
            pre_image = pre_image[cfg.band_index_dict[sat],] # select bands 

            if sat in ['S1', 'ALOS']: pre_image = (np.clip(pre_image, -30, 0) + 30) / 30

            pre_image_pad = image_padding(pre_image, patchsize)
            pre_image_tensor = torch.from_numpy(pre_image_pad).unsqueeze(0) # n * C * H * W
        
            input_tensors.append((pre_image_tensor, post_image_tensor))

        else:
            input_tensors.append((post_image_tensor, ))

    C, H, W = post_image.shape
    _, _, Height, Width = post_image_tensor.shape
    pred_mask_pad = np.zeros((Height, Width)) #HxW
    prob_mask_pad = np.zeros((Height, Width)) #HxW

    ''' tile-wise inference '''
    input_patchsize = 2 * patchsize
    padSize = int(patchsize/2) 
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)

            ''' ------------> tile input data <---------- '''
            input_patchs = []
            for sat_tensor in input_tensors:
                post_patch = (sat_tensor[-1][..., i:i+input_patchsize, j:j+input_patchsize]).type(torch.cuda.FloatTensor)
                if 'pre' in cfg.DATA.PREPOST: 
                    pre_patch = (sat_tensor[0][..., i:i+input_patchsize, j:j+input_patchsize]).type(torch.cuda.FloatTensor)
                    
                    if cfg.DATA.STACKING: 
                        inputPatch = torch.cat([pre_patch, post_patch], dim=1) # stacked inputs
                        input_patchs.append(inputPatch)
                    else:
                        input_patchs += [pre_patch, post_patch]
                else:
                    input_patchs.append(post_patch)

            ''' ------------> apply model <--------------- '''
            # if 'UNet' == cfg.MODEL.ARCH:
            #     # if len(cfg.DATA.SATELLITES) == 1: input = input_patchs[0] # single sensor
            #     # else: input = torch.cat(input_patchs) # stack multi-sensor data
            #     out = model.forward(input_patchs)

            if 'distill_unet' == cfg.MODEL.ARCH:
                if cfg.MODEL.DISTILL:
                    out = model.forward(input_patchs[:1])[-1] # ONLY USE S1 sensor in distill mode.
                else:
                    out = model.forward(input_patchs)[-1] # USE all data in pretrain mode.

            elif 'UNet_resnet' in cfg.MODEL.ARCH:
                out = model.forward(torch.cat(input_patchs, dim=1))

            # elif 'SiamResUNet' in cfg.MODEL.ARCH:
            #     out, decoder_out = model.forward(input_patchs, False)

            # elif 'cdc_unet' in cfg.MODEL.ARCH:
            #     out, decoder_out = model.forward(input_patchs, False)
            
            else: # UNet, SiamUnet
                # NEW: input_patchs should be a list or tuple, the last one is the wanted output.
                out = model.forward(input_patchs)[-1] 

            ''' ------------------------------------------ '''
            activation = Activation(name=cfg.MODEL.ACTIVATION)
            predPatch = activation(out) #NCWH for sigmoid, NWH for argmax, N=1, C=1
            if 'sigmoid' == cfg.MODEL.ACTIVATION:
                predLabel = np.round(predPatch.squeeze().cpu().detach().numpy()) # binarized with 0.5
            else: # 'argmax'
                predLabel = torch.argmax(predPatch, dim=1)
                predLabel = predLabel.squeeze().cpu().detach().numpy()
            
            ''' save predicted tile '''
            pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predLabel[padSize:padSize+patchsize, padSize:padSize+patchsize]

    ''' clip back into original shape '''        
    pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape
    # prod_mask = prob_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape

    return pred_mask

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
    data_dir = Path(cfg.DATA.DIR) / "test_images"

    # orbKeyLen = len(test_id.split("_")[-1]) + 1 
    # event = test_id[:-orbKeyLen]
    event = test_id
    print(event)

    print(f"------------------> {test_id} <-------------------")

    predMask = inference(model, data_dir, event, cfg)
    print(f"predMask shape: {predMask.shape}, unique: {np.unique(predMask)}")
    # print(f"probMask: [{probMask.min()}, {probMask.max()}]")

    # # mtbs_palette =  ["000000", "006400","7fffd4","ffff00","ff0000","7fff00"]
    # # [0,100/255,0]
    # mtbs_palette = [[0,100/255,0], [127/255,1,212/255], [1,1,0], [1,0,0], [127/255,1,0], [1,1,1]]

    tiff.imsave(output_dir / f"{test_id}_pred.tif", predMask)
    # imsave(output_dir / f"{test_id}_pred.png", predMask)
    
    # read and save true labels
    if os.path.isfile(data_dir / "mask" / cfg.DATA.TEST_MASK / f"{event}.tif"):
        trueLabel = tiff.imread(data_dir / "mask" / cfg.DATA.TEST_MASK / f"{event}.tif")
        # _, _, trueLabel = geotiff.read(data_dir / "mask" / "poly" / f"{event}.tif")
        # geotiff.save(output_dir / f"{test_id}_predLabel.tif", predMask[np.newaxis,]) 

        trueLabel = trueLabel.squeeze()

        # plt.imsave(output_dir / f"{test_id}_gts.png", trueLabel, cmap='gray', vmin=0, vmax=1)
        gen_errMap(trueLabel, predMask, save_url=output_dir / f"{test_id}.png")


def evaluate_model(cfg, model_url, output_dir):
    output_dir.mkdir(exist_ok=True)

    test_id_list = os.listdir(Path(cfg.DATA.DIR) / "test_images" / "S2" / "post")
    test_id_list = [test_id[:-4] for test_id in test_id_list]
    print(test_id_list[0])

    model = torch.load(model_url, map_location=torch.device('cpu'))
    # output_dir = Path(SegModel.project_dir) / 'outputs'

    band_index_dict = get_band_index_dict(cfg)
    cfg = edict(cfg)
    cfg.update({"band_index_dict": band_index_dict})
    
    for test_id in test_id_list:
        apply_model_on_event(model, test_id, output_dir, cfg)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./config", config_name="unet")
def run_app(cfg : DictConfig) -> None:

    ''' set randome seed '''
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    os.environ['PYTHONHASHSEED'] = str(cfg.RAND.SEED) #cfg.RAND.SEED
    if cfg.RAND.DETERMIN:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8" #https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        torch.use_deterministic_algorithms(True)
    set_random_seed(cfg.RAND.SEED, deterministic=cfg.RAND.DETERMIN)

    # wandb.init(config=cfg, project=cfg.project.name, name=cfg.EXP.name)
    import pandas as pd
    from prettyprinter import pprint
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_flat = pd.json_normalize(cfg_dict, sep='.').to_dict(orient='records')[0]
    wandb.init(config=cfg_flat, project=cfg.PROJECT.NAME, entity=cfg.PROJECT.ENTITY, name=cfg.EXP.NAME)
    pprint(cfg_flat)

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

    run_dir = Path("/home/p/u/puzhao/smp-seg-pytorch/Canada_RSE_2022/run_poly_UNet_['S1']_EF_20220308T000802")
    model_url = run_dir / "model.pth"
    output_dir = run_dir / "errMap"
    evaluate_model(cfg, model_url, output_dir)

    ''' compute IoU and F1 for all events '''
    from utils.iou4all import multiclass_IoU_F1
    multiclass_IoU_F1(
        pred_dir = run_dir / "errMap", 
        gts_dir = Path(cfg.DATA.DIR) / "test_images" / "mask" / cfg.DATA.TEST_MASK, 
        NUM_CLASS=max(2, cfg.MODEL.NUM_CLASS)
    )
    
    #########################################################################
    wandb.finish()

if __name__ == "__main__":
    
    run_app()


    # model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/best_model_s1s2.pth")
    # output_dir = Path(f"G:/PyProjects/smp-seg-pytorch/outputs/test_output_s1s2_")

    # for test_id in test_id_list:
    #     apply_model_on_event(model, test_id, output_dir, satellites=['S1', 'S2'])