
import os, json 
from pathlib import Path
from random import sample
import shutil
from typing import Tuple
import tifffile as tiff
import numpy as np


def sampling_over_event(dataPath='xx', filename="CA_2019_AB_172", SEED=0, normalize=False):
    print(f"SEED: {SEED}")
    np.random.seed(SEED)

    # post
    s2_post = dataPath / "S2" / "post" / f"{filename}.tif"
    s1_post = dataPath / "S1" / "post" / f"{filename}.tif"
    alos_post = dataPath / "ALOS" / "post" / f"{filename}.tif"

    # pre
    s2_pre = dataPath / "S2" / "pre" / f"{filename}.tif"
    s1_pre = dataPath / "S1" / "pre" / f"{filename}.tif"
    alos_pre = dataPath / "ALOS" / "pre" / f"{filename}.tif"

    url_dict = {
        's2_post': s2_post,
        's1_post': s1_post,
        'alos_post': alos_post,

        's2_pre': s2_pre,
        's1_pre': s1_pre,
        'alos_pre': alos_pre
    }

    POST = []
    for key in ['s2_post', 's1_post', 'alos_post']: # 's2_post', 's1_post', 'alos_post'
        data = tiff.imread(url_dict[key])

        if normalize:
            data = np.nan_to_num(data, 0)
            if key.split("_")[0] in ['s1', 'alos']: data = (np.clip(data, -30, 0) + 30) / 30

        # print(data.shape)
        POST.append(data)
    POST = np.concatenate(POST, axis=0)

    if True:
        PRE = []
        for key in ['s2_pre', 's1_pre', 'alos_pre']: # 's2_pre', 's1_pre', 'alos_pre'
            data = tiff.imread(url_dict[key])

            if normalize:
                data = np.nan_to_num(data, 0)
                if key.split("_")[0] in ['s1', 'alos']: data = (np.clip(data, -30, 0) + 30) / 30

            # print(data.shape)
            PRE.append(data)
        PRE = np.concatenate(PRE, axis=0)

    # mask
    mask_url = dataPath / "mask" / "poly" / f"{filename}.tif"
    mask = tiff.imread(mask_url)
    if len(mask.shape)<3: mask = mask[np.newaxis,]
    
    lc_url = dataPath / "AUZ" / "landcover" / f"{filename}.tif"
    if os.path.isfile(lc_url): 
        landcover = tiff.imread(lc_url)
        if len(landcover.shape)<3: landcover = landcover[np.newaxis,]
        DATA = np.concatenate((POST, PRE, landcover), axis=0)
    else:
        DATA = np.concatenate((POST, PRE), axis=0)

    DATA = np.concatenate((DATA, mask), axis=0)
    C, H, W = DATA.shape
    # print(DATA.shape)

    n_samples = int(H*W*sample_ratio)
    print(f"n_samples / total_pixels: {n_samples} / {H*W}")
    ss = np.random.randint(0, H*W, n_samples)
    # print(ss)
    samples = DATA.reshape(C, H*W).transpose()[ss,:]
    print(samples.shape)

    return samples

if __name__ == "__main__":

    # Model (can also use single decision tree)
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    dataPath = Path("/home/p/u/puzhao/wildfire-s1s2-dataset-ca-tiles") / "test_images"
    BAND_NAMES = ['S2_B2', 'S2_B3', 'S2_B4', 'S2_B8', 'S2_B11', 'S2_B12', 'S1_ND', 'S1_VH', 'S1_VV', 'AL_ND', 'AL_VH', 'AL_VV']
    POST_PRE = [f"{band}_{phase}" for phase in ['post', 'pre'] for band in BAND_NAMES]
    POST_PRE_LC_MASK = POST_PRE + ['Label']

    print(POST_PRE_LC_MASK, len(POST_PRE_LC_MASK))

    sample_ratio = 0.05
    fileList = os.listdir(dataPath / "S2" / "post")

    SAMPLES = []
    for SEED, filename in enumerate(fileList[:3]):
        filename = filename.split(".tif")[0]
        print(f"---------------- {filename} ---------------")
        samples = sampling_over_event(dataPath, filename, SEED) # n_samples x d_features

        SAMPLES.append(samples)
    
    SAMPLES = np.concatenate(SAMPLES, axis=0)

    df = pd.DataFrame(SAMPLES, columns=POST_PRE_LC_MASK)
    print(df)
    df.to_csv('wildfire-s1s2alos-canada-samples.csv')

    NUM_of_TREES = 5
    model = RandomForestClassifier(n_estimators=NUM_of_TREES, verbose=True)

    # Train
    print(f"total number of samples: {SAMPLES.shape[0]}")
    print(f"percentage of postive samples: {100 * sum(SAMPLES[:,-1:])[0] / SAMPLES.shape[0]:.2f}%\n")
    model.fit(SAMPLES[:,:-1], SAMPLES[:,-1:])

    print(model.feature_importances_)





