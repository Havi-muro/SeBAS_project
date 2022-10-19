# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:08:22 2020
@author: Javier Muro
With this code we can create a model trained using all the observations
No validations is performed here. Accuracy of model is the average 
of the 5 folds of the crossvalidation
"""



#from __future__ import absolute_import, division, print_function, unicode_literals

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import scipy as sp
#import statistics
from keras.callbacks import EarlyStopping


import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import os

os.chdir(os.path.join('C://','Users','rsrg_javier','Documents','GitHub','SeBAS_project'))

# Preprocess data
# in be_preprocessing.py, select the study variable and the predictors
import be_preprocessing

# Create an object with the result of  the preprocessing module
# We have to define this to explore the dataset we work with
# and to relate results to other variables in the plots afterwards
studyvar = 'SpecRichness'

Mydataset = be_preprocessing.be_preproc(studyvar)[0]
Mydataset = Mydataset.drop('ep', axis=1)
print(Mydataset.head())


import modelDNN

# Random Distribution of train and test
train_dataset = Mydataset
    
#Copy features
train_features = train_dataset.copy()

#Create y variables (labels) and x (features)
train_labels = train_features.pop(studyvar)

#######################################################################################
# Normalzation
#######################################################################################
#Create normalizer layer and adapt it to our data
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

####################################################################################### 
model = modelDNN.build_model(normalizer, train_features)
EPOCHS = 500
#######################################################################################
   
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

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

import plot_loss
plot_loss.plot_loss(history, EPOCHS, studyvar)

# Measure RMSE using the validation (0.2%) data
RMSE_val = hist[(hist.epoch == (EPOCHS - 1))][['val_root_mean_squared_error']].squeeze()

outdir = os.path.join('C:\\','Users','rsrg_javier','Documents','GitHub','SeBAS_project', 'spatial')


tf.keras.models.save_model(model, filepath=outdir+f'\\{studyvar}_adam_model_S2bands_02Sep')
model.save(f'spatial/{studyvar}_adam_model_S2b_8bands_re3swir2out_oct2022')



























