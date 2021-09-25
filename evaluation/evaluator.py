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
    patchsize = 512

    # if '.tif' == str(url)[-4:]: # tif
    #     img0 = tiff.imread(url)
    #     img = interval_95(np.nan_to_num(img0, 0)) * 255
    # elif '.png' == str(url)[-4:]: # png
    #     img = imread(url)
    pre_url = test_dir / "pre" / f"{test_id}.tif"
    post_url = test_dir / "post" / f"{test_id}.tif"
    mask_url = test_dir / "mask" / f"{test_id}.tif"

    pre_image = tiff.imread(pre_url) 
    post_image = tiff.imread(post_url) 
    mask = tiff.imread(mask_url) 

    img = np.concatenate((pre_image, post_image), axis=0) # C * H * W
    img = img.transpose(1,2,0) # H * W * C
    

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
    prob_mask_pad = np.zeros((6, Height, Width))
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)
            inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]

            # if self.cfg.ARCH == 'FCN':
            #     predPatch = self.model(inputPatch.type(torch.cuda.FloatTensor))['out']
            # else:
            predPatch = model.forward(inputPatch.type(torch.cuda.FloatTensor))
            # predPatch = torch.sigmoid(predPatch)

            predPatch = predPatch.squeeze().cpu().detach().numpy()#.round()
            predLabel = np.argmax(predPatch, axis=0).squeeze()

            pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predLabel[padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify
            prob_mask_pad[:, i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[:, padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

    pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape
    prod_mask = prob_mask_pad[:, padSize:padSize+H, padSize:padSize+W] # clip back to original shape

    return pred_mask, prod_mask


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from utils.GeoTIFF import GeoTIFF

    model = torch.load("G:/PyProjects/smp-seg-pytorch/outputs/run_mtbs_UNet_resnet18_multiclass_20210602T134041/best_model.pth")
    output_dir = Path("G:/PyProjects/smp-seg-pytorch/outputs/test_output")

    test_dir = Path("G:/SAR4Wildfire_Dataset/Fire_Perimeters_US/Monitoring_Trend_in_Burn_Severity/MTBS_L8_Dataset/MTBS_US_toTiles")
    test_id_list = ['ca3442911910020171205',
                    'ca3924012311020180727',
                    'ca4065012263020180723',
                    'ca3982012144020181108',
                    'id4346711414520180729',
                    'wa4695112137420170812',
                    'nv4181211632420180817',
                    'ut4010011096020180701',
                    'ca3906012314020180727',
                    'ca3615911869620170829'
            ]

    def apply_model_on_event(test_id):
        print(f"------------------> {test_id} <-------------------")

        predMask, probMask = inference(model, test_dir, test_id)

        print(f"predMask shape: {predMask.shape}, unique: {np.unique(predMask)}")
        print(f"probMask: [{probMask.min()}, {probMask.max()}]")

        # mtbs_palette =  ["000000", "006400","7fffd4","ffff00","ff0000","7fff00"]
        # [0,100/255,0]
        mtbs_palette = [[0,100/255,0], [127/255,1,212/255], [1,1,0], [1,0,0], [127/255,1,0], [1,1,1]]

        plt.imsave(output_dir / f"{test_id}_predLabel.png", predMask, cmap=ListedColormap(mtbs_palette), vmin=0, vmax=5)

        
        geotiff = GeoTIFF()
        _, _, trueLabel = geotiff.read(test_dir / "mask" / f"{test_id}.tif")
        geotiff.save(output_dir / f"{test_id}_predLabel.tif", predMask[np.newaxis,]) 

        trueLabel = trueLabel.squeeze()
        trueLabel[trueLabel==0] = 1
        plt.imsave(output_dir / f"{test_id}_trueLabel.png", trueLabel-1, cmap=ListedColormap(mtbs_palette), vmin=0, vmax=5)
        # geotiff.save(output_dir / f"{test_id}_trueLabel.tif", trueLabel)

        # geotiff.save(output_dir / f"{test_id}_probMask.tif", probMask)
        print(f"probMask shape: {probMask.shape}")
        if not os.path.exists(output_dir / f"{test_id}"): os.makedirs(output_dir / f"{test_id}")
        for i in range(probMask.shape[0]):
            class_prob = probMask[i,]
            print(i, class_prob.shape)
            imsave(output_dir / f"{test_id}" / f"class_{i}.png", np.uint8(class_prob*255))


    
    for test_id in test_id_list:
        apply_model_on_event(test_id)