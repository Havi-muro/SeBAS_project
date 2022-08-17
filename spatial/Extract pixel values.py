# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 08:51:03 2022

This code extract the pixel values for the geometry given
for several images

@author: rsrg_javier
"""

import geopandas as gpd
import pandas as pd
import rasterio as rio
import os
import glob
from rasterstats import zonal_stats
from pyrsgis import raster


import earthpy.plot as ep
import earthpy.spatial as es

import matplotlib.pyplot as plt


#os.chdir('D:')

# set path to shapefile and read as geodf
geodir = os.path.join('C:/','Users','rsrg_javier','Desktop','SEBAS','GIS','Quadrants_complete', 'AEG01.shp')
geo = gpd.read_file(geodir)

geoalb = geo[(geo.explrtr == 'ALB')][['Plot_ID', 'geometry']]

# set path to rasters
rasdir = os.path.join('D:/','SeBAS_RS','RS','RTM_GEE','LAI')

# list all rasters we want
rasls = glob.glob('*2020*.tif')


rasls_alb = rasdir+'\\alb_TOA_LAI2020-09-08.tif'


#ds1, src = raster.read(rasls_alb)


with rio.open(rasls_alb) as src:
     array = src.read(1)
     trans = src.transform # --> here do src.transform instead of src.affine

zs = zonal_stats(geoalb, array, affine=trans)

zsdf = pd.DataFrame(zs)
zsdf['Plot_ID'] = geoalb['Plot_ID']




means = [f['mean'] for f in zs]


import gdal

r_ds = gdal.Open(fn_raster)
p_ds = ogr.Open(fn_zones)




# import geowombat as gw

# fig, ax = plt.subplots(dpi=200)

# # tell gw to read a blue green red
# with gw.config.update(sensor="bgr"):
#     with gw.open(rasls_alb) as src:
        
#         # see that bands names, blue green red are assigned
#         print(src.band)

#         # # remove 0 value, rearrange band order 
#         # temp = src.where(src != 0).sel(band=["red", "green", "blue"])

#         # # plot
#         # temp.gw.imshow(robust=True, ax=ax)

#         # #save to file
#         # temp.gw.to_raster(
#         #     "../temp/LS_scaled_20200518.tif", verbose=0, n_workers=4, overwrite=True
#         # )    
    
fig, ax = plt.subplots() 
    
plt.imshow(src)


ep.plot_rgb(src, rgb=(3,2,1))

# run zonal_stats
stats = zonal_stats(geodir, src)

# zonal_stats gives the yoze count, min, max and mean
# we can use it to calculate the median if needed 
stats[0].keys()

# Get the means
means = [f['mean'] for f in stats]

# Make df with means and plot ids and concatenate them
meansdf = pd.DataFrame(means)
ids = geo[['Plot_ID']]
mylist = [meansdf,ids]

df = pd.concat(mylist, axis=1)
