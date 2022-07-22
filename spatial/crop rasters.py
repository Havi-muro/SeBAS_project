# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:05:35 2022

@author: rsrg_javier
"""

from rasterio.plot import plotting_extent
import geopandas as gpd

landsat_bands_data_path = "D:\\SeBAS_RS\\RS\\alb\\ALB2020\\tomask\\*.tif"
stack_band_paths = glob(landsat_bands_data_path)
stack_band_paths.sort()

# Create output directory and the output path

output_dir = 'D:\\SeBAS_RS\\RS\\alb\\ALB2020\\tomask\\outputs\\'

raster_out_path = os.path.join(output_dir, "raster.tiff")

array, raster_prof = es.stack(stack_band_paths, out_path=raster_out_path)

extent = plotting_extent(array[0], raster_prof["transform"])


fig, ax = plt.subplots(figsize=(12, 12))
ep.plot_rgb(
    array,
    ax=ax,
    stretch=True,
    extent=extent,
    str_clip=0.5,
    title="RGB Image of Un-cropped Raster",
)
plt.show()

ep.hist(array)
plt.show()

path2shp = 'C:/Users/rsrg_javier/Desktop/SEBAS/GIS/'
crop_bound = gpd.read_file(path2shp+'alb_grass_aoa_2020.shp')


# reproject the data
with rasterio.open(stack_band_paths[0]) as raster_crs:
    crop_raster_profile = raster_crs.profile
    crop_bound_utm13N = crop_bound.to_crs(crop_raster_profile["crs"])


    
# crop
band_paths_list = es.crop_all(
    stack_band_paths, output_dir, crop_bound_utm13N, overwrite=True
)

#To crop one band
# Open Landsat image as a Rasterio object in order to crop it

with rasterio.open(stack_band_paths[0]) as src:
    single_cropped_image, single_cropped_meta = es.crop_image(
        src, crop_bound_utm13N
    )

# Create the extent object
single_crop_extent = plotting_extent(
    single_cropped_image[0], single_cropped_meta["transform"]
)

# Plot the newly cropped image
fig, ax = plt.subplots(figsize=(12, 6))
crop_bound_utm13N.boundary.plot(ax=ax, color="red", zorder=10)
ep.plot_bands(
    single_cropped_image,
    ax=ax,
    extent=single_crop_extent,
    title="Single Cropped Raster and Fire Boundary",
)
plt.show()