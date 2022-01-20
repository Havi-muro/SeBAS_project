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


import be_preprocessing

# Create an object with the result of  the preprocessing module
# We have to define this to explore the dataset we work with
# and to relate results to other variables in the plots afterwards

Mydataset = be_preprocessing.Mydataset
studyvar = 'biomass_g'
MydatasetLUI = be_preprocessing.MydatasetLUI
print(Mydataset.head())
print(list(Mydataset.columns))

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

  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mae',
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model


#Create y (labels) and x (features)

train_labels = Mydataset[[studyvar]].values
train_features = Mydataset.drop(studyvar, axis=1).values

epochs=200

#######################################################################################
# Normalzation
#######################################################################################
#Get statistics of the training dataset
#train_dataset.describe().transpose()[['mean', 'std']]

#Create normalizer layer and adapt it to our data
#with function preprocessing.Normalization()
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
train_features = pd.DataFrame(normalizer(train_features))

model=KerasRegressor(build_fn = build_model)


dnn_parameters = {
    'nodes': [2, 16, 64, 128]          
    }

kfold = KFold(5, shuffle=False)


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



###############################################################
# https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
# The constructor for the KerasClassifier class can take default arguments 
# that are passed on to the calls to model.fit(), such as the number of 
# epochs and the batch size.

model = KerasRegressor(build_fn = build_model)



dnn_gridsearch.fit(train_features, train_labels)

best_estimator = dnn_gridsearch.best_estimator_
       
print(best_estimator)

rmse_dnn=round((-dnn_gridsearch.score(test_features, test_labels))**(1/2),2)
    
    
