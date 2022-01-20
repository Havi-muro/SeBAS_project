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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


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
def build_model(hp):
  model = keras.Sequential([
    #normalizer,
    layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32), 
                 activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 input_shape=[len(train_features.keys())]), 
                # !!! We had to change here to get the shape from the np array
                # input_shape=[len(train_features.keys())])
       
    layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32), 
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    # layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32), 
    #              activation='relu',                 
    #               kernel_regularizer=keras.regularizers.l1(0.01),
    #               ),
    
    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss=['mae'],
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model





#######################################################################################
# Normalzation
#######################################################################################
#Get statistics of the training dataset
#train_dataset.describe().transpose()[['mean', 'std']]

#Create y (labels) and x (features)

# Distribution of train and test
train_dataset = Mydataset.sample(frac=0.8,random_state=0)
test_dataset = Mydataset.drop(train_dataset.index)

train_labels = train_dataset[[studyvar]]
train_features = train_dataset.drop(studyvar, axis=1)

test_labels = test_dataset[[studyvar]]
test_features = test_dataset.drop(studyvar, axis=1)

#Create normalizer layer and adapt it to our data
#with function preprocessing.Normalization()
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

from tensorflow import keras
from tensorflow.keras import layers
# conda install -c conda-forge keras-tuner
import keras_tuner as kt

build_model(kt.HyperParameters())
hp = kt.HyperParameters()

import os

tuner = kt.Hyperband(
    hypermodel=build_model,
    objective=kt.Objective('root_mean_squared_error', direction='min'),
    max_epochs=10,
    executions_per_trial=2,
    overwrite=True,
    directory=os.path.normpath('C:/keras_tuner_dir'),
    project_name='keras_tuner_demo'
)


# Print a summary of the search space:
tuner.search_space_summary()

# Start the grid search
tuner.search(train_features, train_labels, epochs=10, validation_split=0.2)
#validation_split=0.2
#validation_data=(test_features, test_labels)


best_model = tuner.get_best_models()[0]
best_model.build(train_features.shape)
best_model.summary()










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
    
    
