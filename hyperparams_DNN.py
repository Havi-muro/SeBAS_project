"""
Created on Tue Jan 11 10:08:22 2022

@author: Javier Muro

This code determines the best hyperparameters to chose in a random forest model.
From a list of possible options, it tests all the possible combinations of
hyperparameters and returns the best one. 

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd


#for kfolds
import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.layers.experimental import preprocessing


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from keras.wrappers.scikit_learn import KerasRegressor

#from sklearn.utils import shuffle
cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project

# Load datasets
Mydataset_0 = pd.read_csv ('data/Bexis_S1S2_height_NMDS_RaoQ_Dec2021.csv')

# The year and the ep have been concatenated to sort the observations by
# Exoloratory, plot number and year so that: 
# A01-2017,H01-2017,S01-2017,A01-2018,H01-2018,S01-2018...
# This allows a non-shuffeled kfold cross validation to have
# all observations/replicates from e.g. A01 for all years in either training or validation

Mydataset_0 = Mydataset_0.sort_values(by='yep')

#Mydataset_0 = Mydataset_0[Mydataset_0['explo']=='SCH']

Mydataset_0.head(5)

# Replace missing values in Sentinel-1 data with median values
medVH = Mydataset_0['VHMean_May'].median()
medVV = Mydataset_0['VVMean_May'].median()
medVVVH = Mydataset_0['VVVH'].median()

Mydataset_0['VHMean_May'] = Mydataset_0['VHMean_May'].fillna(medVH)
Mydataset_0['VVMean_May'] = Mydataset_0['VVMean_May'].fillna(medVV)
Mydataset_0['VVVH'] = Mydataset_0['VVVH'].fillna(medVVVH)

# Select study variable and covariates
Mydataset_vars = Mydataset_0.drop([
    
                #### Identification variables ####
    'x', 'y', 
    'explo', 
     #'yep',
     #'Year', 
     #'ep',
                    #### Study variables ####   
        'SpecRichness',
        'height_cm',
        "biomass_g",
        'NMDS1',
        'NMDS2',
        #'SpecRich_157',
        'Rao_Q',
        'Redundancy',        
        'Shannon',
        'Simpson',
        'FisherAlpha',
        'PielouEvenness',
        'number_vascular_plants',
              
                         #### Predictors ####
                  "LUI_2015_2018",
                  "SoilTypeFusion" ,
                  'LAI',
                  'slope',
                  'aspect',
                  'blue','green', 'red', 'nir', 'nirb', 're1','re2','re3', 'swir1', 'swir2',
                  'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI',
                  'VHMean_May',
                  'VVMean_May',
                  'VVVH',
                  'TWI'
             
       ], axis=1)



# Mydataset_vars = Mydataset_0[['Year', 'ep', 'biomass_g'
#                               #,'LAI'
#                                ,'blue_3','green_3', 'red_3', 'nir_3', 'nirb_3', 're1_3', 're2_3', 're3_3', 'swir1_3', 'swir2_3'
#                                #,'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI'
#                                #,'VHMean_May','VVMean_May','VVVH'
#                               # ,'SoilTypeFusion'
#                               # ,'slope'
#                                #,'aspect'
#                                #,'TWI'
#                                ]]


list(Mydataset_vars.columns)

studyvar = 'SpecRich_157'


#replace nas with mean?
Mydataset = Mydataset_vars.dropna()

#08 swampy
#10 dry soil
#12 path inside
#16 no management for 2 years
#19 water
#33 partially grazed
#35 very dry
#38 swampy and dung
#40 dry

#Some plots have trees inside
trees = ['AEG07','AEG09','AEG25','AEG26','AEG27','AEG47','AEG48','HEG09','HEG21','HEG24','HEG43']

#others are too dry or swamped at moment of biomass collection
outliers2018 = ['SEG08', 'SEG10', 'SEG11', 'SEG12', 'SEG16', 'SEG18', 'SEG19', 'SEG20', 'SEG31',  
'SEG33', 'SEG35', 'SEG36', 'SEG38', 'SEG39', 'SEG40', 'SEG41', 'SEG44', 'SEG46', 'SEG45', 'SEG49', 'SEG50']

#Filter those and drop the variables not needed
Mydataset = Mydataset.drop(Mydataset[(Mydataset['Year'] == 2018) & (Mydataset['ep'].isin(outliers2018))].index)
Mydataset = Mydataset.drop(Mydataset[Mydataset['ep'].isin(trees)].index)

Mydataset = Mydataset.drop(['Year', 'ep', 'yep'], axis=1)


#Soil and explo arecategorica variables. Change it to one-hot encoded
#Mydataset = pd.get_dummies(Mydataset, prefix='', prefix_sep='')
print(Mydataset.head())


#Create y (labels) and x (features)
x_columns = Mydataset.columns.drop(studyvar)
x = Mydataset[x_columns].values
y = Mydataset[studyvar].values

# Define the K-fold Cross Validator
kfold = KFold(5, shuffle=False)

# define model
def build_model(nodes):
  model = keras.Sequential([
    #normalizer,
    layers.Dense(nodes, 
                 activation='relu', 
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 input_shape=train_features.shape), #!!! We had to change here to get the shape from the np array
       
    layers.Dense(nodes, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 ),
    
    layers.Dense(nodes, activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model

# K-fold Cross Validation model evaluation
fold = 0
for train, test in kfold.split(x):
    fold+=1
    print(f'Fold#{fold}')
    
    train_features = x[train]
    train_labels = y[train]
    test_features = x[test]
    test_labels = y[test]
    
    #######################################################################################
    # Normalzation
    #######################################################################################
    #Get statistics of the training dataset
    #train_dataset.describe().transpose()[['mean', 'std']]
    
    #Create normalizer layer and adapt it to our data
    #with function preprocessing.Normalization()
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    train_features = normalizer(train_features)
    
    dnn_parameters = {
        'nodes': [2, 16, 64, 128]          
        }
    
    ###############################################################
    # https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
    # The constructor for the KerasClassifier class can take default arguments 
    # that are passed on to the calls to model.fit(), such as the number of 
    # epochs and the batch size.
    
    model = KerasRegressor(build_fn = build_model)
    
    dnn_gridsearch = GridSearchCV(
        estimator = model,
        param_grid= dnn_parameters,
        cv=5,
        n_jobs=8, # Number of cores: Adapt this parameter before reproducing on another machine
        scoring='neg_mean_squared_error',
        verbose=1
        )

    dnn_gridsearch.fit(train_features, train_labels)

    best_estimator = dnn_gridsearch.best_estimator_
           
    print(best_estimator)
    
    rmse_dnn=round((-dnn_gridsearch.score(test_features, test_labels))**(1/2),2)
    
    
