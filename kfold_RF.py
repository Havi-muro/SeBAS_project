"""

Created on Mon Dec 14 10:08:22 2021

@author: Javier Muro 

This module defines the function to run a Random Forest model using a k-fold
approach. Data is normalized using the preprocessing.Normalzation() function.

The output is a series of lists with accuracy values and predictor
importance of each fold

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

#for kfolds
from sklearn.model_selection import KFold
#from sklearn.preprocessing import Normalizer

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers.experimental import preprocessing

import be_preprocessing

# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset

pred_trues = []

importance_list = []


# K-fold Cross Validation model evaluation
def kfold_RF(studyvar):
    
    # Define the K-fold Cross Validator
    kfold = KFold(5, shuffle=False)
    
    #Create y (labels) and x (features)
    x_columns = Mydataset.columns.drop(studyvar)
    x = Mydataset[x_columns].values
    y = Mydataset[studyvar].values
    
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
             
        c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1)
        pred_trues.append(c)
               
        importance = model.feature_importances_
        importance_list.append(importance)