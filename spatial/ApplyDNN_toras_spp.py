# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:44:07 2021
this code will apply the DNN model of biomass or spp richness
to a raster image

Optionally it takes one band of the image, copies it, and gives all the pixels value 1 for ALB, 
2 for HAI and 3 for SCH. this is so that the model knows which exploratory we are dealing with.
It inserts that new band onto the raster. Model is loaded and applied to that raster.

Making no data values is only for visualization purposes.

@author: Javier Muro
"""
#conda install -c conda-forge earthpy
#import earthpy as et
import earthpy.spatial as es
#import earthpy.plot as ep

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

#import tensorflow as tf
from tensorflow import keras

import os
#conda install -c pratyusht pyrsgis
#conda install --channel "pratyusht" package
#import pyrsgis
from pyrsgis import raster
from pyrsgis.convert import changeDimension

import geopandas as gpd
import rioxarray as rxr
#import rasterio as rio
from shapely.geometry import mapping

from glob import glob

explo = 'alb'
year = '2019'

# open with os so that shashes and capitals don't matter
os.chdir(os.path.join('D:/','SeBAS_RS','RS',explo,f'{explo}{year}'))

##############################################################################################
            
# Import all bands in a list and stack them (spp richness)
file_list = glob(f'{explo}*_b*.tif')

# In case we need to remove some band
#file_list = [i for i in file_list if not i.endswith('b06.tif')]

# Delete the items as necessary
# del file_list[11:13]
# del file_list[9]

# resorting if needed because of the band names 
# file_list_s = [file_list[i] for i in [10, 0, 1,2,3,4,5,6,7,8,9]]

# stack bands with earthpy
arr_st, meta = es.stack(file_list)

###############################################################################

# # write the full stack in case it is necessary
# meta.update(count = arr_st.shape[0])
# with rio.open(f'{explo}_{year}_fullstack.tif', 'w', **meta, BIGTIFF='YES') as dst:
#         dst.write(arr_st)

###############################################################################

# # Read the full stack with pyrsgis
# ds1, myrast = raster.read(f'{explo}_{year}_fullstack.tif', bands='all')

# ep.plot_bands(myrast[0])
# ep.hist(myrast[0])
# print(np.amax(myrast))
# print(np.amin(myrast))

##############################################################################

#Change raster dimensions. 
#No need to normalize, since the model performs normalization
myrast_reshape = changeDimension(arr_st)
#myrast_reshape = changeDimension(myrast)

# Read model
pathspp = os.path.join('C:\\','Users','rsrg_javier','Documents','GitHub','SeBAS_project','spatial','SpecRichness_adam_model_S2bands_20July')
model = keras.models.load_model(pathspp)

#Apply model
myrast_pred = model.predict(myrast_reshape)
print(np.amax(myrast_pred))
print(np.amin(myrast_pred))

# Reshape the raster according to the original dimensions
#prediction = np.reshape(predict_masked, (ds1.RasterYSize, ds1.RasterXSize))
prediction = np.reshape(myrast_pred, (arr_st.shape[1], arr_st.shape[2]))

# more than 70 spp per 4 x 4 is not reasonable, so we eliminate those values
#clipped_pred = np.clip(prediction, 0, 100, out=None)
prediction[prediction>70] = np.nan

# Open one of the bands to get the metadata with pyrsgis.raster
ds1, myrast = raster.read(file_list[1], bands=1)

raster.export(prediction, ds1, filename=f'{explo}_spprich_{year}.tif', dtype='float')

# clip:  We have to read the prediction with rasterio to apply rio.clip
# set path to mask and open it
path2mask = os.path.join('D:\\','SeBAS_RS','RS',f'{explo}',f'{explo}2020\\')
mask = gpd.read_file(path2mask+f'{explo}_grass_aoa_2020.shp')
mask.crs

# Get spp richness file
spp = rxr.open_rasterio(f'{explo}_spprich_{year}.tif', masked=True).squeeze()
spp.rio.crs

# clip and export
sppcl = spp.rio.clip(mask.geometry.apply(mapping, mask.crs)) # This is needed if your GDF is in a diff CRS than the raster data
sppcl.rio.to_raster(f'{explo}_spprich_{year}_clip.tif')

##############################################################################











