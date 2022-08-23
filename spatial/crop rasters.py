# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:27:33 2022

@author: rsrg_javier
"""

import os
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
import matplotlib.pyplot as plt
from shapely.geometry import mapping

from pyrsgis import raster


from glob import glob

explo = 'sch'
year = '2020'

# set wd 
os.chdir(os.path.join('D:\\','SeBAS_RS','RS',f'{explo}',f'{explo}{year}\\'))


# set path to data (optional)
mypath = os.path.join('D:\\','SeBAS_RS','RS',f'{explo}',f'{explo}{year}\\')
extent = gpd.read_file(mypath+f'{explo}_grass_aoa_2020.shp')
extent.crs


spp = rxr.open_rasterio(mypath+f'{explo}_Spprich_adam_2020_20July.tiff', masked=True).squeeze()
spp.rio.crs


# plot raster and mask
# f, ax = plt.subplots(figsize=(10, 5))
# spp.plot.imshow()
# extent.plot(ax=ax, alpha=0.8)
# plt.show()


# clip
# Spp richness (one per year)
sppcl = spp.rio.clip(extent.geometry.apply(mapping, extent.crs)) # This is needed if your GDF is in a diff CRS than the raster data

# f, ax = plt.subplots(figsize=(10, 5))
# sppcl.plot.imshow()

sppcl.rio.to_raster(mypath+ f'{explo}_Spprich_adam_2020_20July_cl.tiff')


# Biomass in loop
loopls = glob('*_biomass_looped.tif')

for i in range(len(loopls)):
    bm = rxr.open_rasterio(mypath+loopls[i], masked=True).squeeze()
    bm_cl = bm.rio.clip(extent.geometry.apply(mapping, extent.crs))
    bm_cl.rio.to_raster(f'cl_{loopls[i]}.tif')


    

