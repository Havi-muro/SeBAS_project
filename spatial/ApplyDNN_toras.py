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
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os
#conda install -c pratyusht pyrsgis
#conda install --channel "pratyusht" package
import pyrsgis
from pyrsgis import raster
from pyrsgis.convert import changeDimension

import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from shapely.geometry import mapping

from glob import glob

explo = 'hai'

#mydir = 'C:\\Users\\rsrg_javier\\Desktop\\SEBAS\\GIS\\RS\\ALB\\ALB2020\\'
#mydir = f'D:\SeBAS_RS\RS\{explo}\ALB2020\\'


# open with os so that shashes and capitals don't matter
os.chdir(os.path.join('D:/','SeBAS_RS','RS',explo,f'{explo}2020'))

##############################################################################################

# This part is to get a raster with value 1 for Alb, 2 for Hai and 3 for Sch 
# so that the model knows which site we are working on
# Open one band with rasterio
#myimg = rio.open('C:/Users/rsrg_javier/Desktop/SEBAS/GIS/RS/HAI/hai_20200524_b01.tif')
#read it
#fullimg = myimg.read()

#see metadata
#myimg.meta

#Several ways of dividing the bands to get value 1.
#unno = np.divide(fullimg, fullimg)
#uno = fullimg/fullimg+1
#type(uno)

# Write raster. We have to specify the metadata from our original image
# from rio.transform import from_origin
# from rio.crs import CRS
# with rio.open('ALB_bValue1nop.tif', 'w', 
#                    driver = "Gtiff",
#                    height= fullimg.shape[1],
#                    width= fullimg.shape[2],
#                    count = fullimg.shape[0],
#                    dtype = fullimg.dtype,
#                    crs = CRS.from_epsg(32632),
#                    transform = (9.99515425538844, 0.0, 512852.72680448176, 0.0, -9.999610409760084, 5376215.3472465305)
#                    ) as dst:
#     dst.write(uno)
            
# Import all bands in a list and stack them     
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

# Concatenate both np.ndarrays. 
# We have to convert them in list or tuples first (with the parenthesis)
# Not needed for the moment. Models work the same ignoring which exploratory we are modelling
# arr_st = np.concatenate((uno, arr_st))
# type(arr_st)

############   Mask out weird values just for visualization   ################
arr_st_masked = np.ma.masked_values(arr_st, np.amin(arr_st))
  
fig, ax = plt.subplots(figsize=(12, 12))
ep.plot_rgb(arr_st_masked, rgb=(9, 2, 0), ax=ax, title="Sentinel-2")
plt.show()

fig, ax = plt.subplots(figsize=(12, 12))
ep.plot_bands(arr_st_masked)
plt.show()

fig, ax = plt.subplots(figsize=(12, 12))
ep.plot_bands(arr_st)
plt.show()

##############################################################################

# write the full stack
meta.update(count = arr_st.shape[0])
with rio.open(+f'{explo}_fullstack_2017_20july.tif', 'w', **meta, BIGTIFF='YES') as dst:
        dst.write(arr_st)

###############################################################################

# Read raster
ds1, myrast = raster.read(f'{explo}_fullstack_2017_20july.tif', bands='all')

ep.plot_bands(myrast[0])
ep.hist(myrast[0])
print(np.amax(myrast))
print(np.amin(myrast))

##############  Mask out extreme values, only for visualization  ##############
myrast_masked = np.ma.masked_values(myrast, np.amin(myrast))
myrast_masked = np.ma.masked_values(myrast_masked, np.amax(myrast))

print(np.amax(myrast_masked))
print(np.amin(myrast_masked))


# Plot raster and histgram
fig, ax = plt.subplots(figsize=(12, 12))

ep.plot_rgb(myrast_masked, rgb=(3, 2, 1), ax=ax, title="Sentinel-2 NIR-G-R")
plt.show()

ep.hist(myrast_masked)
plt.show()
###############################################################################

#Change raster dimensions. 
#No need to normalize, since the model performs normalization
myrast_reshape = changeDimension(myrast)

# Read model
model = keras.models.load_model('C:/Users/rsrg_javier/Documents/GitHub/SeBAS_project/spatial/SpecRichness_adam_model_S2bands_20July')

#Apply model
myrast_pred = model.predict(myrast_reshape)
print(np.amax(myrast_pred))
print(np.amin(myrast_pred))

#New no data values appear. We can remove them with either max, or mode. 
#We might have to do it twice because of two maximums
ep.hist(myrast_pred, figsize=(5,5))
values, counts = np.unique(myrast_pred, return_counts=True)

ind = np.argmax(counts)
mode = values[ind]  # prints the most frequent element
mode

predict_masked = np.ma.masked_values(myrast_pred, mode)
print(np.amax(predict_masked))
print(np.amin(predict_masked))

# Run the next lines iteratively until the max is reasonable
predict_masked = np.ma.masked_values(predict_masked, np.amax(predict_masked))
print(np.amax(predict_masked))
print(np.amin(predict_masked))

ep.hist(predict_masked, figsize=(5,5))

# Reshape the raster according to the original dimensions
prediction = np.reshape(predict_masked, (ds1.RasterYSize, ds1.RasterXSize))


# set path to mask and open it
aoi = os.path.join(f'{explo}_grass_aoa_2020.shp')
extent = gpd.read_file(aoi)
extent.crs

# clip
sppcl = prediction.rio.clip(extent.geometry.apply(mapping, extent.crs)) # This is needed if your GDF is in a diff CRS than the raster data

plt.imshow(sppcl)
ep.hist(sppcl)

raster.export(sppcl, ds1, filename=f'{explo}_Spprich_adam_2020_20july.tiff', dtype='float')

##############################################################################











