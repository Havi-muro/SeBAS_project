"""

Created on Mon Dec 14 10:08:22 2021

@author: Javier Muro 

This module defines the function to run a Random Forest model using a 
group k-fold approach. 

Data is normalized using the preprocessing.Normalzation() function.

The output is a series of lists with accuracy values and predictor
importance of each fold

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

#for kfolds
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers.experimental import preprocessing

import be_preprocessing

# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset

pred_trues = []

importance_list = []


def gkfold_RF(studyvar):
    #Create y (labels) and x (features)
    epg = Mydataset['ep']
    x_columns = Mydataset.columns.drop([studyvar, 'ep'])
    x = Mydataset[x_columns].values
    y = Mydataset[studyvar].values
     
    # Group K-fold Cross Validation model evaluation
    gkf = GroupKFold(n_splits=5)
        
    fold = 0
    for split, (train, test) in enumerate(gkf.split(x, y, groups=epg)):
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