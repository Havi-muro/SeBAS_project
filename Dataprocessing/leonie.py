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
import matplotlib.pyplot as plt

cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project


# Load datasets
Mydataset_0 = pd.read_csv ('data/all_reduced2.csv')

Mydataset_0.columns

#studyvars = 'HHLAI', 'Biomass_g_per60cm2',
#       'Biomass_interpolated', 'RPMcalc_cm'

studyvar = 'Biomass_g_per60cm2'

Mydataset_vars = Mydataset_0[[studyvar,
       'blue', 'green', 'red', 'nir', 
       'ndvi', 'evi2', 'evi', 'savi', 'dvi', 'gndvi', 'grvi',
       ]]

Mydataset = Mydataset_vars.dropna()
print(Mydataset.head())
print(list(Mydataset.columns))

def plot_loss(history, EPOCHS, studyvar):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  #plt.legend()
  plt.grid(True)

def kfold_DNN(EPOCHS, studyvar):
    #Create y (labels) and x (features)
    x_columns = Mydataset.columns.drop(studyvar)
    x = Mydataset[x_columns].values
    y = Mydataset[studyvar].values
     
    # K-fold Cross Validation model evaluation
    kfold = KFold(5, shuffle=False)
    #EPOCHS = 200
        
    fold = 0
    for train, test in kfold.split(x):
        fold+=1
        print(f'Fold#{fold}')
        
        train_features = x[train]
        train_labels = y[train]
        test_features = x[test]
        test_labels = y[test]
        
        #######################################################################
        # Normalzation
        #######################################################################
       
        #Create normalizer layer and adapt it to our data
        #with function preprocessing.Normalization()
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))
        
        ###############################################################
        #######################################################################
        model = modelDNN.build_model(normalizer, train_features)
        #model.summary()
        #######################################################################
        
        #Add an early stopping to avoid overfitting
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        
        #Train model
        #Keras fit function expects training features to be as array.
        history = model.fit(
            train_features, 
            train_labels, 
            epochs=EPOCHS, 
            validation_split = 0.2, 
            verbose=0
            ,callbacks=[es]
            )
            
        #######################################################################
        #Plot errors
        # Show last few epochs in history
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
           
        plot_loss(history, EPOCHS, studyvar)
              
        #Predictions
        #Make predictions on the test data using the model, and stored results of each fold
        test_predictions = model.predict(test_features).flatten()
        #predictions_list.extend(test_predictions)
        
        c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1)
        pred_trues.append(c)
                       
    return model


from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import gaussian_kde


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import modelDNN

pred_trues = []

EPOCHS = 500

kfold_DNN(EPOCHS, studyvar)

# We build a df of the accumulated predictions vs labels
pred_truesdf = pd.concat(pred_trues).reset_index(drop=True)
pred_truesdf.columns = ['labels','preds']

y = pred_truesdf['preds']
x = pred_truesdf['labels']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)

plt.ylabel(f'Predicted {studyvar}')
plt.xlabel(f'In situ {studyvar}')
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))
#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()

