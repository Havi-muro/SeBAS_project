# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:57:36 2022

Crop rasters using an upper threshold.
Species richness models make impossibly high predictions in some areas where,
for instance, the land cover is not grassland anymore because of changes

@author: rsrg_javier
"""

#import rasterio as rio
from pyrsgis import raster
import numpy as np
#import numpy.ma as ma

import os
from glob import glob


explo = 'alb'
yearl = [2017, 2018, 2019, 2020, 2021]

for year in yearl:
    # open with os so that shashes and capitals don't matter
    os.chdir(os.path.join('D:/','SeBAS_RS','RS',explo,f'{explo}{year}'))
    
    
    fl = glob('*_clip.tif*')
    fl0 = fl[0]
    
    ds1, r = raster.read(fl0)
    
    r[r>81] = np.nan
    
    raster.export(r, ds1, filename=f'{explo}_spprich_{year}_81max.tif', dtype='float')
