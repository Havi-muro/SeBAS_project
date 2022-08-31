# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:58:18 2022

@author: rsrg_javier
"""

import os
from glob import glob

cd D:\SeBAS_RS\RS\HAI\HAI2021

# make lists with the different bands
blue = glob('*blue.tif')
green = glob('*green.tif')
red = glob('*red.tif')
nir = glob('*nir.tif')
nirb = glob('*nirb.tif')
re1 = glob('*re1.tif')
re2 = glob('*re2.tif')
re3 = glob('*re3.tif')
swir1 = glob('*swir1.tif')
swir2 = glob('*swir2.tif')

# rename this variable according to the band
band = swir1

# Loop over the list changing the last part of the name
for name in range(len(band)):
    oldname = band[name]    
    newname = band[name][0:13]+'b09.tif'
    rename = os.rename(oldname, newname)
    
# Playground
frags = ['*green.tif', '*red.tif', '*nir.tif','*nirb.tif','*re1.tif',
         '*re2.tif', '*re3.tif', '*swir1.tif','*swir2.tif']

ind = ['b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b09', 'b10']

for frag in range(len(frags)):
    band = glob(frags[frag])
    #print(band)
    for date in range(len(band)):
        name = band[date]
        print(name)
        #for char in range(len(name)):
            #print(name[char])
        
        
        for (i, j) in zip(name, ind):
            newname = name[i][0:13]+ind[j]+'.tif'
            print(newname)
            
            rename = os.rename(oldname, newname)
    



x=['sch_20210328_swir2.tif', 'sch_20210411_swir2.tif', 'sch_20210425_swir2.tif', 'sch_20210509_swir2.tif', 'sch_20210523_swir2.tif', 'sch_20210606_swir2.tif', 'sch_20210620_swir2.tif', 'sch_20210704_swir2.tif', 'sch_20210718_swir2.tif', 'sch_20210801_swir2.tif', 'sch_20210815_swir2.tif', 'sch_20210829_swir2.tif', 'sch_20210912_swir2.tif', 'sch_20210926_swir2.tif', 'sch_20211010_swir2.tif', 'sch_20211024_swir2.tif']

for i in range(len(x)):
    print(str(x[i]))


for (i, j) in zip(frags, ind):
    print(frags, ind)
















