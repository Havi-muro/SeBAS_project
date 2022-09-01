"""
Created on Tue Jan 11 10:08:22 2022

@author: Javier Muro

This code determines the best hyperparameters to chose in a for a DNN.
From a list of possible options, it tests all the possible combinations of
hyperparameters and returns the best one. 

Optimal hyperparameters are extracted/printed from the best model 
of each fold

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import sklearn
from sklearn.model_selection import KFold

# conda install -c conda-forge keras-tuner
import keras_tuner as kt
import os


#from sklearn.utils import shuffle
cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project


import be_preprocessing

# Create an object with the result of  the preprocessing module
# We have to define this to explore the dataset we work with
# and to relate results to other variables in the plots afterwards

Mydataset = be_preprocessing.Mydataset

#studyvar = 'biomass_g'
studyvar = 'SpecRichness'

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
                # input_shape=[len(train_features.keys())]),
                 input_shape=train_features.shape),
    
    #layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1, default=0.5)),
       
    layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32), 
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    #layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1, default=0.5)),

        
    layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32),
                 activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    #layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1, default=0.5)),

    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss=['mae'],
                optimizer=hp.Choice('optimizer', values=['adam', 'adagrad', 'RMSprop']),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model


#Create y (labels) and x (features)
x_columns = Mydataset.columns.drop(studyvar)
x = Mydataset[x_columns].values
y = Mydataset[studyvar].values

# Define the K-fold Cross Validator
kfold = KFold(5, shuffle=False)

best_list = []

# K-fold Cross Validation model evaluation
fold = 0
for train, test in kfold.split(x):
    fold+=1
    print(f'Fold#{fold}')
    
    train_features = x[train]
    train_labels = y[train]
    test_features = x[test]
    test_labels = y[test]

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    build_model(kt.HyperParameters())
    hp = kt.HyperParameters()

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective=kt.Objective('root_mean_squared_error', direction='min'),
        max_epochs=500,
        executions_per_trial=2,
        overwrite=True,
        directory=os.path.normpath(f'C:/keras_tuner_dir/Fold{fold}'),
        project_name='keras_tuner_demo',
    )

    # Print a summary of the search space:
    # tuner.search_space_summary()

    # Start the grid search using 20% of the training data for validation (aka dev_set)
    tuner.search(train_features, train_labels, epochs=500, validation_split=0.2)
    #validation_split=0.2
    #validation_data=(test_features, test_labels)


    best_model = tuner.get_best_models()[0]
    best_model.build(train_features.shape)
    best_list.append(best_model)
    #tuner.results_summary()
    #print(tuner.results_summary(2))


# Print the results of the best model for each fold
print(best_list[0].summary())
print(best_list[1].summary())
print(best_list[2].summary())
print(best_list[3].summary())
print(best_list[4].summary())

print(best_list[0].optimizer)
print(best_list[1].optimizer)
print(best_list[2].optimizer)
print(best_list[3].optimizer)
print(best_list[4].optimizer)
