# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:44:07 2021
This code will apply the DNN model of biomass to a raster image

It loos over the list of images, groups them by date, and apply the model

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
import pyrsgis
from pyrsgis import raster
from pyrsgis.convert import changeDimension

import geopandas as gpd
#import rioxarray as rxr
import rasterio as rio
#from shapely.geometry import mapping

from glob import glob

explo = 'alb'
year = '2021'


# open with os so that shashes and capitals don't matter
os.chdir(os.path.join('D:\\','SeBAS_RS','RS',f'{explo}',f'{explo}{year}\\'))

# list all images
file_list1 = glob(f'{explo}*_b*.tif')

# list comprehension to get a list with the dates for the biomass model
sliced = [x[4:12] for x in file_list1]

# remote duplicates so we get the list of dates
sliced_u = list(set(sliced))

# Read model
mpath = os.path.join('C:\\','Users','rsrg_javier','Documents','GitHub','SeBAS_project','spatial','Biomass_model_S2bands')
model = keras.models.load_model(mpath)

# Open one of the bands to get the metadata with pyrsgis.raster
ds1, myrast = raster.read(f'{explo}_{sliced_u[1]}_b01.tif', bands=1)

# stack the images per date and apply the model
for i in range(len(sliced_u)):
    
    # List with bands
    glist = glob(f'{explo}*{sliced_u[i]}*_b*.tif')
    
    # stack bands with earthpy    
    arr_st, meta = es.stack(glist)
    
    #Change raster dimensions
    myrast_reshape = changeDimension(arr_st)
    
    #Apply model
    myrast_pred = model.predict(myrast_reshape)
       
    # Reshape the raster according to the original dimensions
    prediction = np.reshape(myrast_pred, (arr_st.shape[1], arr_st.shape[2]))
    
    # Eliminate edge values
    clipped_pred = np.clip(prediction, 0, 800, out=None)

    # Export raster with pyrsgis using the metadata of one of the images
    raster.export(clipped_pred, ds1, filename=f'{explo}_{sliced_u[i]}_biomass_looped.tif', dtype='float')
    
    ##############################################################################


# Playground

















with rio.open('outname.tif', 'w', **meta) as dst:
    dst.write(prediction, 1)


    # # write the full stack
meta.update(count = 1)
with rio.open(f'{explo}_20200510_biomass_3.tif', 'w', **meta, BIGTIFF='YES') as dst:
        dst.write(prediction)

prediction.rio.to_raster('test_ndvi.tif')

ds1, myrast = raster.read(f'{explo}_20200510_b01.tif', bands='all')
raster.export(prediction, ds1, filename=f'{explo}_20200510_biomass_4.tif', dtype='float')

aoi = os.path.join(f'{explo}_grass_aoa_2020.shp')
extent = gpd.read_file(aoi)
extent.crs

clipped_pred = np.clip(prediction, 0, 800, out=None)
raster.export(clipped_pred, ds1, filename=f'{explo}_20200830_biomass_looped_clip2.tif', dtype='float')


loopls = glob(f'{explo}*looped*.tif')

for i in range(len(loopls)):
    print(loopls[i])
    rast = pyrsgis.raster.read(loopls[i])
    rast
    clipped_p = np.clip(rast, 0, 800, out=None)
    raster.export(clipped_p, ds1, filename=f'{explo}_{sliced_u[i]}_biomass_looped_cl.tif', dtype='float')
    

