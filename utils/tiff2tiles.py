import os
import numpy as np
from utils.GeoTIFF import GeoTIFF
from pathlib import Path


geotiff = GeoTIFF()
def geotiff_tiling(phase, src_url, dst_dir, BANDS, BANDS_INDEX, tile_size=256, tiling=False):
    
    event_id = os.path.split(src_url)[-1][:-4].split("_")[0]
    print(f"{event_id}: tiling")
    print(f"Bands: {BANDS}")

    _, _, im_data = geotiff.read(src_url)
    im_data = np.nan_to_num(im_data, 0)
    im_data = im_data[BANDS_INDEX, :, :]

    if 'test' == phase: 
        test_img_dir = Path(str(dst_dir).replace('test', 'test_images'))
        geotiff.save(
                url= test_img_dir / f"{event_id}.tif", 
                im_data=im_data, 
                bandNameList=BANDS
            )

    # print(im_data.dtype.name)
    # print(im_data.shape)
    # print(im_data.min(), im_data.max())

    if tiling:
        C, H, W = im_data.shape
        # print(C, H, W)

        H_ = (H // tile_size + 1) * tile_size - H
        W_ = (W // tile_size + 1) * tile_size - W

        # print(H_, W_)
        # select bands
        bottom_pad = np.flip(im_data[:, H-H_:H, :], axis=1)
        # print(bottom_pad.shape)

        # dim2
        im_data_expanded = np.hstack((im_data, bottom_pad))
        right_pad = np.flip(im_data_expanded[:, :, W-W_:W], axis=2)

        # dim3
        im_data_expanded = np.dstack((im_data_expanded, right_pad))
        # print(im_data_expanded.shape)

        _, H1, W1 = im_data_expanded.shape
        for i in range(0, H1 // tile_size):
            for j in range(0, W1 // tile_size):
                tile = im_data_expanded[:, i*256:(i+1)*256, j*256:(j+1)*256]
                geotiff.save(
                    url=dst_dir / f"{event_id}_{i}_{j}.tif", 
                    im_data=tile, 
                    bandNameList=BANDS,
                    tiling=tiling
                )



if __name__ == "__main__":

    BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    BANDS_INDEX = [0, 1, 2, 6, 8, 9]

    src_url = "D:\wildfire-s1s2-dataset-ak\S2\post/ak6186714639320190717.tif"
    dst_dir = Path("G:\PyProjects\smp-seg-pytorch\outputs")

    geotiff_tiling(src_url, dst_dir, BANDS, BANDS_INDEX, tile_size=256)