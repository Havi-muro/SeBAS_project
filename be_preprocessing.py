# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:01:52 2021

This module preprocess the bexis data and selects the variable of study
as well as predictors. It fills nans and removes identified outliers.

Select predictors for species richness by dropping, 
or for biomass, by extracting [[]]

For spatial crossvalidation, comment second to last line so that the exploratory
information is preserved.

@author: Javier Muro

"""

import numpy as np
import pandas as pd

def be_preproc(studyvar):

    # Load datasets
    Mydataset_0 = pd.read_csv ('data/Bexis_Sentinels_full_dataset.csv')
    
    # The year and the ep have been concatenated to sort the observations by
    # Exoloratory, plot number and year so that: 
    # A01-2017,H01-2017,S01-2017,A01-2018,H01-2018,S01-2018...
    # This allows a non-shuffeled kfold cross validation to have
    # all observations/replicates from e.g. A01 for all years in either training or validation
    
    Mydataset_0 = Mydataset_0.sort_values(by='yep')
    
    Mydataset_0.head(5)
    
    # Replace missing values in Sentinel-1 data with median values
    medVH = Mydataset_0['VHMean_May'].median()
    medVV = Mydataset_0['VVMean_May'].median()
    medVVVH = Mydataset_0['VVVH'].median()
    
    Mydataset_0['VHMean_May'] = Mydataset_0['VHMean_May'].fillna(medVH)
    Mydataset_0['VVMean_May'] = Mydataset_0['VVMean_May'].fillna(medVV)
    Mydataset_0['VVVH'] = Mydataset_0['VVVH'].fillna(medVVVH)
    
    
    # Select study variable and predictors by dropping
    # Useful when we want to include the whole time series
    # keep variable 'explo' for spatial cross-validation
    if studyvar == 'SpecRichness':
        Mydataset_vars = Mydataset_0.drop([
            
                        #### Identification variables ####
            'x', 'y', 
            #'explo', 
             #'yep',
             #'Year', 
             #'ep',
                            #### Study variables ####  
                  'Shannon_157',
                  'Simpson_157',
                  'inverse_Simpson_157',
                  'PielouEvenness_157',        
                #'SpecRichness',
                'height_cm',
                "biomass_g",
                'NMDS1',
                'NMDS2',
                'SpecRich_157',
                'Rao_Q_157',
                'Redundancy_157',        
                'Shannon',
                'Simpson',
                'FisherAlpha',
                'PielouEvenness',
                'number_vascular_plants',
                      
                                 #### Predictors ####
                        'S2QF',
                          #"LUI_2015_2018",
                          "SoilTypeFusion" ,
                          'slope',
                          'aspect',
                          'S2Q', 
                          'blue','green', 'red', 'nir', 'nirb', 're1','re2','re3', 'swir1', 'swir2',
                          'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI',
                          'LAI',
                          'VHMean_May',
                          'VVMean_May',
                          'VVVH',
                          'TWI'
               ], axis=1)
    elif studyvar == 'biomass_g':
        # Remove all columns that start with 
        #Mydataset_vars= Mydataset_vars.loc[:,~Mydataset_vars.columns.str.startswith('re3')]
        
        #Mydataset_vars= Mydataset_vars.loc[:,Mydataset_vars.columns.str.startswith('re3')]
        
        # Select study variable and predictors by subsetting
        # keep variable 'explo' for spatial cross-validation
        # SpecRichness or biomass_g
        Mydataset_vars = Mydataset_0[['Year', 'ep', 'biomass_g', 'yep', 'LUI_2015_2018',
                                        'explo',
                                        'blue',
                                        'green', 
                                        'red', 
                                        'nir', 
                                        'nirb', 
                                        're1',
                                        're2',
                                        're3', 
                                        'swir1', 
                                        'swir2',
                                        #'blue_sd_3','green_sd_3', 'red_sd_3', 'nir_sd_3', 'nirb_sd_3', 're1_sd_3','re2_sd_3','re3_sd_3', 'swir1_sd_3', 'swir2_sd_3',
                                        #  ,'S2Q',
                                        #'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI', 'LAI'
                                        #'VHMean_May','VVMean_May','VVVH'
                                        #'SoilTypeFusion'
                                        #,'slope', 'aspect'
                                        #,'TWI'
                                        ]]
    else:
        print('study var names: SpecRichness or biomass_g (as string)')
        
    
    # nir_3 corresponds to Mid may. 
    # The differences in correlation between the orignial band and nir_3
    # correspond to SCH, because the original is two weeks later
    
    Mydataset = Mydataset_vars.dropna()
    
    # Complicated plots at SCH
    #08 swampy
    #10 dry soil
    #12 path inside
    #16 no management for 2 years
    #19 water
    #33 partially grazed
    #35 very dry
    #38 swampy and dung
    #40 dry
    
    # Some plots have trees inside
    trees = ['AEG07','AEG09','AEG25','AEG26','AEG27','AEG47',
             'AEG48','HEG09','HEG21','HEG24','HEG43']
    
    # others are too dry or swamped at moment of biomass collection
    outliers2018 = ['SEG08', 'SEG10', 'SEG11', 'SEG12', 'SEG16', 'SEG18', 
                    'SEG19', 'SEG20', 'SEG31', 'SEG33', 'SEG35', 'SEG36', 
                    'SEG38', 'SEG39', 'SEG40', 'SEG41', 'SEG44', 'SEG46', 
                    'SEG45', 'SEG49', 'SEG50']
    
    # Filter those and drop the variables not needed
    Mydataset = Mydataset.drop(Mydataset[(Mydataset['Year'] == 2018)
                                         &
                                         (Mydataset['ep'].isin(outliers2018))].index)
    
    Mydataset = Mydataset.drop(Mydataset[Mydataset['ep'].isin(trees)].index)
    
    # We need to match predictions with original dataset to display colors by LUI
    MydatasetLUI=Mydataset
    MydatasetLUI.index = np.arange(0, len(MydatasetLUI))
    
    Mydataset = Mydataset.drop(['Year', 'LUI_2015_2018','yep', 'explo'], axis=1)
    
    # Soil and explo are categorica variables. Change it to one-hot encoded.
    # Comment this line if doing spatial cross-validation.
    #Mydataset = pd.get_dummies(Mydataset, prefix='', prefix_sep='')
    return Mydataset, MydatasetLUI
