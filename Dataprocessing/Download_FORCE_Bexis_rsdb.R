###############################################################################
# This code accesses the bexis remote sensing database and downloads the imgs #
# band by band                                                                #

###############################################################################

# Install RSDB package and automatically install updated versions.
# In some cases a restart of R is needed to work with a updated version of RSDB package (in RStudio - Session - Terminate R).
#if(!require("remotes")) install.packages("remotes")
#remotes::install_github("environmentalinformatics-marburg/rsdb/r-package")

library(RSDB)
library(raster)
library(rgdal)
library(data.table)
library(stringr)

# get documentation
#??RSDB

# create connection to RSDB server.
remotesensing <- RSDB::RemoteSensing$new(url = "https://vhrz1078.hrz.uni-marburg.de:8201", userpwd = 'xxxxx')

# name variables
explo = 'hai'
year = '2019'
mypath = paste0('D:/SeBAS_RS/RS/', explo,'/',explo, year, '/')
rasterdb <- remotesensing$rasterdb(paste0('S2_TSI_2017_2021_',explo))

# list time slices
all_imgs <- rasterdb$time_slices$name
imgs2021 <- all_imgs[105:120]
imgs2019 <- all_imgs[53:68]
imgs2018 <- all_imgs[27:42]
imgs2017 <- all_imgs[5:16]

#bands <- c(rasterdb$bands$title)

# Exploratory extent
haiex <-raster::extent(580972, 626815, 5644122, 5694934)
albex <-raster::extent(512852, 544777, 5353793, 5376215)
schex <-raster::extent(793821, 847417, 5860795, 5908471)

if(explo == 'hai'){
  ext <- haiex
} else if (explo == 'alb'){
  ext <- albex
} else if (explo == 'sch'){
  ext <- schex
}

# To download the raster, it has to be band by band.
# name the bands 1-10 with leading zeros
for (time_slice in eval(as.name(paste0('imgs', year)))){
  r <- rasterdb$raster(ext = ext, time_slice = time_slice, band = c(1, 2, 3, 4, 5, 6, 7, 8 ,9, 10))
  for (band in 1:10){
      writeRaster(r[[band]], 
                  filename = paste0(mypath, explo, '_',time_slice, '_b', str_pad(band, 2, pad = "0")), 
                  format = 'GTiff')
  }
}
