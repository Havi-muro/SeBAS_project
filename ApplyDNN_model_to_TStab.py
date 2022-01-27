# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:58:34 2021

Apply biomass model to tablular data for predictions across time

@author: rsrg_javier
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import statistics

#for kfolds
# import sklearn
# from sklearn.model_selection import KFold
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.preprocessing import Normalizer

# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras


Mydataset_0 = pd.read_csv ('C:/Users/rsrg_javier/Desktop/SEBAS/python/Sentinel2_TS_by_timeslice_alt.csv')

Mydataset_vars = Mydataset_0.drop(['ep', 'time_slice'], axis=1)

model = keras.models.load_model('C:/Users/rsrg_javier/Desktop/SEBAS/python/BiomassModel')

predictions = model.predict(Mydataset_vars).flatten()

# Create a new column with the results of the predictions in the test_dataset
Mydataset_0['preds'] = predictions
  
# Merge the predicted results with the original dataset
# Mydataset_preds = pd.merge(Mydataset_0,test_dataset[['preds']], how = 'right',left_index = True, right_index = True)

Mydataset_0.to_csv('C:/Users/rsrg_javier/Desktop/SEBAS/python/TS_predictions_biomass_alt.csv')

