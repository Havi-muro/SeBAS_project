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

# set wd 
os.chdir('D:')

# set path to mask if different from wd and open it
aoi = os.path.join('C:/','Users','rsrg_javier','Desktop','SEBAS','GIS','alb_grass_aoa_2020.shp')
extent = gpd.read_file(aoi)
extent.crs


spprichness = os.path.join('D:/','SeBAS_RS','RS','alb','ALB2020','alb_Spprich_adam_2020_20July.tiff')
spp = rxr.open_rasterio(spprichness, masked=True).squeeze()
spp.rio.crs


# plot raster and mask
f, ax = plt.subplots(figsize=(10, 5))
spp.plot.imshow()

extent.plot(ax=ax, alpha=0.8)
plt.show()

# clip
sppcl = spp.rio.clip(extent.geometry.apply(mapping, extent.crs)) # This is needed if your GDF is in a diff CRS than the raster data

f, ax = plt.subplots(figsize=(10, 5))
sppcl.plot.imshow()

output = os.path.join('SeBAS_RS','RS','alb','ALB2020','alb_Spprich_adam_2020_20July_cl.tiff')
sppcl.rio.to_raster(output)


