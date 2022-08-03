######################################################################
#  This code extract band values for all images in each exploratory  #
#  It creates separate csv for each time slice, and them binds them  #
#  all by column e.g. A01_20170101, A01_20170115...                  #
#  This is done to get all the time series per year and per explo    #
#  in one file (total of 12 files) that then we bind by row          #
######################################################################

# ?RSDB::RasterDB for documentation

# 
# if(!require('remotes')) install.packages('remotes')
# remotes::install_github('environmentalinformatics-marburg/rsdb/r-package')
library(RSDB)
library(raster)
library(rgdal)
library(tidyr)

remotesensing <- RSDB::RemoteSensing$new(url = 'https://vhrz1078.hrz.uni-marburg.de:8201', userpwd = 'javier.muro:Ye72UZYkpU!Y')

explo = 'HAI'
mypath = paste0('C:/Users/Janny/Desktop/SEBAS/', explo, '/')

#rasterdb <- remotesensing$rasterdb('S2_TSI_2017_2021_alb')
#rasterdb <- remotesensing$rasterdb('S2_TSI_2017_2021_hai')
rasterdb <- remotesensing$rasterdb('S2_TSI_2017_2021_sch')

# get time_slices
#rasterdb$time_slices

# get band meta data
#rasterdb$bands

bands <- c(1, 2, 3, 4, 5, 6, 7, 8 ,9, 10)

# list time slices we want
mytimeslices <- rasterdb$time_slices[2]
mytimelist <- as.list(mytimeslices)

# list for sebas fieldwork
#          2020-Jun             2020-Sep        2021-Jun                  2021-Sep
# Alb: 20200621 (15-19) // 20200913 (7-11)  //                  // 20210926 (20-24)
# Hai: 20200607 (8-12)  // 20200913 (14-19) // 20210606 (31-04) // 20210912 (13-17)
# Sch:                                         20210523 (25-28) // 20210912 (6-9)

time_slice <- '20210912'
month <- substring(time_slice, 5, 6)
year <- substring(time_slice, 1, 4)

# get EPSG
#epsg <- substring(rasterdb$geo_code, 6)
r <- rasterdb$raster(rasterdb$extent, time_slice = time_slice, band = c(1, 2, 3, 4, 5, 6, 7, 8 ,9, 10))
#plot(r)
#crs(r)

# load SeBAS geometry or holy areas
plots<-readOGR('C:/Users/Janny/Desktop/SEBAS/Quadrats2021_dd.shp')
crs(plots[1])
# plots <- readOGR( 
#   dsn= paste0(mypath, 'Holyareas_match_HAI.shp') , 
#   verbose=FALSE
# )

# get plot polygons in correct EPSG using the crs of a raster
plots <- spTransform(plots, crs(r))
#class(plots) #if it is a dataframe, we should use st_transform
#summary(plots)
#plotsT <- st_transform(plots, CRS('+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'))


#extract pixel values per plot for all bands
# we have to specify the package, because the function shares name
# with a function from tidyr
imgext<-raster::extract(r,plots, fun=median, df=TRUE )

#bind columns to preserve id and subset useful ones
pixelvalues <- cbind(plots, imgext)
my_cols <- c(1,12,3,14:23)
pixelvaluessub <- pixelvalues[my_cols]

# Add a column with the year and give it the name Year
pixelval_addcol<- data.frame(pixelvaluessub[1],year,pixelvaluessub[,2:13])
pixelval_addcol$month<- with(pixelval_addcol, month)
#names(pixelval_addcol)[names(pixelval_addcol) == 'X2020'] <- 'Year'

pixelval_addcol <- pixelval_addcol %>%drop_na()

#first field is explrtr, so we can use it in the name
write.csv(pixelval_addcol, file=paste0("C:/Users/Janny/Desktop/SEBAS/",pixelval_addcol[1,1], time_slice, ".csv"))



