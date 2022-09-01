"""
Created on Mon Dec 14 10:08:22 2020
@author: Janny
This code is a little bit trickier. It is a stand-alone script, meaning that
it doesn't need our other modules. It uses a slightly different normalization,
which is not included in the model structure.

The code returnsthe variable importance of each feature every fold 
using shapley values

With the script shapplots.py we can aggregate the importances by band or by date 
and plot the results

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
#import scipy as sp
#import statistics
import os

#for kfolds
#import sklearn
from sklearn.model_selection import GroupKFold
#from sklearn import metrics
#from sklearn import preprocessing
#from sklearn.preprocessing import Normalizer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing

import shap
shap.initjs()

# set owrking directory
os.chdir(os.path.join('C:/','Users','rsrg_javier','Documents','GitHub','SeBAS_project'))

# Load datasets
Mydataset_0 = pd.read_csv ('data/Bexis_S1S2_height_NMDS_RaoQ_S2Q_May_forest.csv')

Mydataset_0.head(5)

##############################################################################
# Preprocessing abd variable selection
##############################################################################

# Replace missing values in Sentinel-1 data with median values
medVH = Mydataset_0['VHMean_May'].median()
medVV = Mydataset_0['VVMean_May'].median()
medVVVH = Mydataset_0['VVVH'].median()

Mydataset_0['VHMean_May'] = Mydataset_0['VHMean_May'].fillna(medVH)
Mydataset_0['VVMean_May'] = Mydataset_0['VVMean_May'].fillna(medVV)
Mydataset_0['VVVH'] = Mydataset_0['VVVH'].fillna(medVVVH)

# Select study variable and covariates
Mydataset_vars = Mydataset_0.drop(['x', 'y', 
    'explo', 
     'yep',
              #'Year', 'ep',
              #'SpecRichness',
              'height_cm',
              "biomass_g", 
              'Shannon',
              'Simpson',
              'Shannon_157',
                 'Simpson_157',
                 'inverse_Simpson_157',
                 'PielouEvenness_157',
                 'Rao_Q_157',
                 'Redundancy_157',
              'FisherAlpha',
              'PielouEvenness',
              'number_vascular_plants',
              'NMDS1',
              'NMDS2',
              'SpecRich_157',
              
              "LUI_2015_2018",
              
              "SoilTypeFusion",
              #'S2Q',
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

list(Mydataset_vars.columns)

# Make lists to remove correlated bands for species diversity predictions
# removeswir2 = list(Mydataset_vars.filter(regex ='^swir2'))
# removeblue = list(Mydataset_vars.filter(regex ='^blue'))
# removere1 = list(Mydataset_vars.filter(regex ='^re1'))
# removere2 = list(Mydataset_vars.filter(regex ='^re2'))

# removevars = removeblue+removeswir2
# Mydataset_vars=Mydataset_vars.drop(removevars, axis=1)

# Select variables for biomass and height predictions
Mydataset_vars = Mydataset_0[['Year', 'ep', 'biomass_g', 
                                'blue_3','green_3', 'red_3', 'nir_3', 'nirb_3', 're1_3', 're2_3', 're3_3', 'swir1_3', 'swir2_3'
                                #,'blue_sd_3','green_sd_3', 'red_sd_3', 'nir_sd_3', 'nirb_sd_3', 're1_sd_3', 're2_sd_3', 're3_sd_3', 'swir1_sd_3', 'swir2_sd_3'
                                #,'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI'
                                #,'VHMean_May','VVMean_May','VVVH'
                                #,'SoilTypeFusion'
                                #,'slope', 'aspect'
                                #,'TWI'
                                ]]

# nir_3 corresponds to Mid may. 

studyvar = 'biomass_g'

Mydataset = Mydataset_vars.dropna()

# Some plots have trees inside
trees = ['AEG07','AEG09','AEG25','AEG26','AEG27','AEG47',
         'AEG48','HEG09','HEG21','HEG24','HEG43']

# others are too dry or swamped at moment of biomass collection
#08 swampy
#10 dry soil
#12 path inside
#16 no management for 2 years
#19 water
#33 partially grazed
#35 very dry
#38 swampy and dung
#40 dry

outliers2018 = ['SEG08', 'SEG10', 'SEG11', 'SEG12', 'SEG16', 'SEG18', 'SEG19',
                'SEG20', 'SEG31', 'SEG33', 'SEG35', 'SEG36', 'SEG38', 'SEG39',
                'SEG40', 'SEG41', 'SEG44', 'SEG46', 'SEG45', 'SEG49', 'SEG50']

# Eliminate the abnormal observations and drop the variables not needed
Mydataset = Mydataset.drop(Mydataset[(Mydataset['Year'] == 2018) & 
                                     (Mydataset['ep'].isin(outliers2018))].index)

Mydataset = Mydataset.drop(Mydataset[Mydataset['ep'].isin(trees)].index)
Mydataset = Mydataset.drop(['Year'], axis=1)

#Soil and explo arecategorica variables. Change it to one-hot encoded
#Mydataset = pd.get_dummies(Mydataset, prefix='', prefix_sep='')
print(Mydataset.head())

##############################################################################
# define model and history plot functions
##############################################################################

def build_model():
  model = keras.Sequential([
    #normalizer,
    layers.Dense(64, 
                 activation='relu', 
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 input_shape=[len(train_features.keys())]), #!!! We had to change here to get the shape from the np array
       
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 ),
    
    layers.Dense(64, activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    layers.Dense(1)
  ])

  model.compile(loss='mae',
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model

def plot_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  plt.legend()
  plt.grid(True)
  
##############################################################################

#Create y (labels),  x (features) and grouping feature
epg = Mydataset['ep']
x_columns = Mydataset.columns.drop([studyvar, 'ep'])
x = Mydataset[x_columns]
y = Mydataset[studyvar]

# Define the K-fold Cross Validator
gkf = GroupKFold(n_splits=5)
EPOCHS = 500

pred_trues = []
var_imp_list = []

# K-fold Cross Validation model evaluation
fold = 0
for split, (train, test) in enumerate(gkf.split(x, y, groups=epg)):
    fold+=1
    print(f'Fold#{fold}')
        
    train_features = x.iloc[train]
    train_labels = y.iloc[train]
    test_features = x.iloc[test]
    test_labels = y.iloc[test]

###############################################################################
    # Normalzation
###############################################################################
  
    scaler = StandardScaler()
    scaler.fit(train_features)
    normed_train_features= scaler.transform(train_features)
    normed_test_features= scaler.transform(test_features)

###############################################################################
    model = build_model()
    
    #Add an early stopping to avoid overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    
    #Train model
    #Keras fit function expects training features to be as array.
    history = model.fit(
        train_features, 
        train_labels, 
        epochs=EPOCHS, 
        validation_split = 0.2, 
        verbose=0,
        callbacks=[es]
        )
###############################################################################
    # Plot errors
    # Show last few epochs in history
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #plot_loss(history)
       
    # Make predictions on the test data using the model 
    test_predictions = model.predict(normed_test_features).flatten()
    
    c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1)
    c.columns = ['labels', 'preds']
    pred_trues.append(c)
    
    # initiate shap analysis of how the model learns    
    explainer = shap.DeepExplainer(model,normed_train_features)

    shap_values = explainer.shap_values(normed_train_features)

    shap_plot = shap.summary_plot(shap_values, 
                                  feature_names= x.columns,  
                                  plot_type="bar", 
                                  max_display=33)
    
    # Shap_values is a list of matrices. We have to change that.
    vals = np.abs(shap_values).mean(0)
    # Get the feature names
    feature_names = x.columns

# Join in a dataframe the shap_values with their corresponding feature names
# Order remains the same than in the initial df
    feature_importance = pd.DataFrame(list(zip(feature_names, sum(vals))),
                                      columns=['col_name','feature_importance_vals'])
    
    var_imp_list.append(feature_importance)

     
#concatenate all variable importances side by side    
var_imp_df = pd.concat(var_imp_list, axis=1)

# Select useful columns
var_imp_df= var_imp_df.iloc[:,[0,1,3,5,7,9]].set_index('col_name')

# Calculate the mean of each variable importance across folds
var_imp_mean = var_imp_df.mean(axis=1)
var_imp_sd = var_imp_df.std(axis=1)

var_imp = pd.concat([var_imp_mean, var_imp_sd], axis=1)

# var_imp.to_csv(f'results/var_imp_kfold_DNN_{studyvar}_S2Q_May2022.csv')

# Dependence plots
# The partial dependence plot shows the marginal effect 
# one or two features have on the predicted outcome. 
# It tells whether the relationship between the target and a feature is linear, 
# monotonic or more complex. It automatically includes another variable that 
# your chosen variable interacts most with.

shap.dependence_plot('green_3', shap_values[0], train_features)


