import gdal
import os, glob
from imageio import imread
from pathlib import Path 
import numpy as np

class GeoTIFF:
    """ GeoTIFF is written to read and save GeoTiff data"""

    # read image files
    def read(self, url):
        if not isinstance(url, str): url = str(url)
        raster = gdal.Open(url)  # open file

        im_width = raster.RasterXSize  # get width
        im_height = raster.RasterYSize  # get height

        im_geotrans = raster.GetGeoTransform()  # get geoTransform
        im_proj = raster.GetProjection()  # get Projection
        im_data = raster.ReadAsArray(0, 0, im_width, im_height)  # read data as array

        if len(im_data.shape)==2: im_data = im_data[np.newaxis,]

        del raster
        self.im_proj, self.im_geotrans = im_proj, im_geotrans
        return im_proj, im_geotrans, im_data

    # write tiff file
    def save(self, url, im_data, bandNameList=None, im_proj=None, im_geotrans=None):
        # gdal data types include:
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        if (im_proj is None) or (im_geotrans is None): im_proj, im_geotrans = self.im_proj, self.im_geotrans

        # check the datatype of raster data
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # get the dimension
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        
        if len(im_data.shape) == 2:
            im_data = im_data[np.newaxis,]
            im_bands, im_height, im_width = im_data.shape

        # create output folder
        outputFolder = os.path.split(url)[0]
        if not os.path.exists(outputFolder): os.makedirs(outputFolder)

        # create the output file
        driver = gdal.GetDriverByName("GTiff")  # specify the format
        if not isinstance(url, str): url = str(url)
        raster = driver.Create(url, im_width, im_height, im_bands, datatype, options=["TILED=YES",
                                                                                    "COMPRESS=LZW",
                                                                                    "INTERLEAVE=BAND"])

        if (raster != None):
            raster.SetGeoTransform(im_geotrans)  # write affine transformation parameter
            raster.SetProjection(im_proj)  # write Projection
        else:
            print("Fails to create output file !!!")
        
        # write bands one by one
        for idx in range(0, im_bands):
            rasterBand = raster.GetRasterBand(idx+1)
            rasterBand.SetNoDataValue(0)

            if bandNameList is not None: rasterBand.SetDescription(bandNameList[idx])
            rasterBand.WriteArray(im_data[idx, ...])            

        del raster

    def convert_png_to_geotiff(self, src_url, dst_url, geo_url, bandNameList=None, norm=True):
        im_proj, im_geotrans, _ = self.read(geo_url)
        im_data = imread(src_url) 

        im_data = imread(src_url) # H*W*C
        if len(im_data.shape)==2: im_data = im_data[np.newaxis,]
        im_data = im_data.transpose(2,0,1) # C*H*W

        if norm: im_data = im_data / 255.0
        self.save(dst_url, im_data, bandNameList, im_proj, im_geotrans)



class GRID:
    """ GeoTIFF is written to read and save GeoTiff data"""

    # read image files
    def read_data(self, url):
        if not isinstance(url, str): url = str(url)
        raster = gdal.Open(url)  # open file

        im_width = raster.RasterXSize  # get width
        im_height = raster.RasterYSize  # get height

        im_geotrans = raster.GetGeoTransform()  # get geoTransform
        im_proj = raster.GetProjection()  # get Projection
        im_data = raster.ReadAsArray(0, 0, im_width, im_height)  # read data as array

        if len(im_data.shape)==2: im_data = im_data[np.newaxis,]

        del raster
        self.im_proj, self.im_geotrans = im_proj, im_geotrans
        return im_proj, im_geotrans, im_data

    # write tiff file
    def write_data(self, url, im_data, im_proj=None, im_geotrans=None, bandNameList=None):
        # gdal data types include:
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        if (im_proj is None) or (im_geotrans is None): im_proj, im_geotrans = self.im_proj, self.im_geotrans

        # check the datatype of raster data
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # get the dimension
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        
        if len(im_data.shape) == 2:
            im_data = im_data[np.newaxis,]
            im_bands, im_height, im_width = im_data.shape

        # create output folder
        outputFolder = os.path.split(url)[0]
        if not os.path.exists(outputFolder): os.makedirs(outputFolder)

        # create the output file
        driver = gdal.GetDriverByName("GTiff")  # specify the format
        if not isinstance(url, str): url = str(url)
        raster = driver.Create(url, im_width, im_height, im_bands, datatype, options=["TILED=YES",
                                                                                    "COMPRESS=LZW",
                                                                                    "INTERLEAVE=BAND"])

        if (raster != None):
            raster.SetGeoTransform(im_geotrans)  # write affine transformation parameter
            raster.SetProjection(im_proj)  # write Projection
        else:
            print("Fails to create output file !!!")
        
        # write bands one by one
        for idx in range(0, im_bands):
            rasterBand = raster.GetRasterBand(idx+1)
            rasterBand.SetNoDataValue(0)

            if bandNameList is not None: rasterBand.SetDescription(bandNameList[idx])
            rasterBand.WriteArray(im_data[idx, ...])            

        del raster

    def convert_png_to_geotiff(self, src_url, dst_url, geo_url, bandNameList=None, norm=True):
        im_proj, im_geotrans, _ = self.read(geo_url)
        im_data = imread(src_url) 

        im_data = imread(src_url) # H*W*C
        if len(im_data.shape)==2: im_data = im_data[np.newaxis,]
        im_data = im_data.transpose(2,0,1) # C*H*W

        if norm: im_data = im_data / 255.0
        self.save(dst_url, im_data, bandNameList, im_proj, im_geotrans)

if __name__ == "__main__":
    
    workspace = Path("E:\Wildfire_SE_Events_2018_V2\SWE_Data_for_Users_Drive\SE2018Ljusdals")

    geotiff = GeoTIFF()
    
    src_url = workspace / "S2_MSI_RGB" / "20180717T10_S2.png"
    geo_url = workspace / "S2_MSI_Tif" / "20180831T10_S2.tif"
    dst_url = workspace / "Saved_Tif_V1" / "polyMap.tif"

    # geotiff.convert_png_to_geotiff(src_url, dst_url, geo_url, bandNameList=['B12', 'B8', 'B11'], norm=True)

    im_proj, im_geotrans, im_data = geotiff.read(workspace / "S1_SAR_Tif" / "20180728T16_ASC175.tif")
    geotiff.save(workspace / "Saved_Tif_V1" / "polyMap_V2.tif", im_data, ['sarPolyMap'], im_proj, im_geotrans)

    


    