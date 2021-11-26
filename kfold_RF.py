"""
Created on Mon Dec 14 10:08:22 2020
@author: Janny
This code builds a sequential deep learning model in keras. 
Data is normalized using the preprocessing.Normalzation() function
Ttraining and validation points are distributed following a kfold approach
for each exploratory
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import statistics

#for kfolds
import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers.experimental import preprocessing

#from sklearn.utils import shuffle


import be_preprocessing
# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset
studyvar = 'biomass_g'


# Define the K-fold Cross Validator
kfold = KFold(5, shuffle=False)

RMSE_test_list = []
RRMSE_test_list = []
RMSE_val_list = []
rsq_list = []
p_list = []
predictions_list=[]

importance_list = []

#Create y (labels) and x (features)
x_columns = Mydataset.columns.drop(studyvar)
x = Mydataset[x_columns].values
y = Mydataset[studyvar].values

# K-fold Cross Validation model evaluation
def kfold_RF():
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
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))
        
        #######################################################################
        # Build model
        model = RandomForestRegressor(n_estimators=500, 
                                      max_features=int(train_features.shape[1]/3))
        # model.summary()
        #######################################################################
       
        # Train model
        history = model.fit(train_features, train_labels)
            
        #######################################################################
        #Predictions
        #Make predictions on the test data using the model 
        test_predictions = model.predict(test_features).flatten()
        predictions_list.extend(test_predictions)
        
        # Measure this fold's RMSE using the test data
        RMSE_test = np.sqrt(metrics.mean_squared_error(test_predictions,test_labels))
        #print(f"RMSE test data: {RMSE_test}")
        
        # Calculate r2 between predicted adn test data
        linreg = sp.stats.linregress(test_predictions ,test_labels)
        rsq = linreg.rvalue **2
        rsq_list.append(rsq)
        p = linreg.pvalue
        #p_list.append(p)
            
        #Calculate the relative root mean squared error
        test_mean = np.mean(test_labels)
        RRMSE_test = (RMSE_test / test_mean)
        RMSE_test_list.append(RMSE_test)
        RRMSE_test_list.append(RRMSE_test)
               
        importance = model.feature_importances_
        importance_list.append(importance)