# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:44:07 2022

This code collects single bands or tiles from a 
larger area and stacks/composites them together

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
import rasterio
from rasterio.crs import CRS

from glob import glob

mydir = 'C:\\Users\\rsrg_javier\\Desktop\\SEBAS\\GIS\\RS\\HAI\\HAI2020\\'
date = '20200913'
##############################################################################################

# This part is to get a raster with value 1 for Alb, 2 for Hai and 3 for Sch 
# so that the model knows which site we are working on
# Open one band with rasterio
#myimg = rasterio.open('C:/Users/rsrg_javier/Desktop/SEBAS/GIS/RS/HAI/hai_20200524_b01.tif')
#read it
#fullimg = myimg.read()

#see metadata
#myimg.meta

#Several ways of dividing the bands to get value 1.
#unno = np.divide(fullimg, fullimg)
#uno = fullimg/fullimg+1
#type(uno)

# Write raster. We have to specify the metadata from our original image
# from rasterio.transform import from_origin

# with rasterio.open('ALB_bValue1nop.tif', 'w', 
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
# Pay attention because sometimes it is tif and others tiff    
file_list = glob(mydir+f'*{date}*.tif')

# Delete the items as necessary
# del file_list[11:13]
# del file_list[9]

# resorting if needed because of the band names 
# file_list_s = [file_list[i] for i in [10, 0, 1,2,3,4,5,6,7,8,9]]


##############################################################################
# stack bands with earthpy
arr_st, meta = es.stack(file_list)

# write the full stack
meta.update(count = arr_st.shape[0])
with rasterio.open(mydir+f'hai_{date}_st.tif', 'w', **meta, BIGTIFF='YES') as dst:
        dst.write(arr_st)

##############################################################################

# Make a composite
from rasterio.merge import merge
from rasterio.plot import show
#create empty list
src_files_to_mosaic = []

# Open all those files in read mode with rasterio 
# and add those files into our source file list

for img in file_list:
    src=rasterio.open(img)
    src_files_to_mosaic.append(src)

# Merge function returns a single mosaic array and the transformation info
mosaic, out_trans = merge(src_files_to_mosaic)

show(mosaic[0], cmap='terrain')

# update the metadata with the new dimensions
out_meta = src.meta.copy()
out_meta.update({'driver':'GTiff',
                 'height':mosaic.shape[1],
                 'width': mosaic.shape[2],
                 'transform': out_trans,
                 'crs':CRS.from_epsg(32632)
                 })

#write the composite
with rasterio.open(mydir+f'hai_{date}_st.tif','w', **out_meta, BIGTIFF='YES') as dest:
    dest.write(mosaic)


