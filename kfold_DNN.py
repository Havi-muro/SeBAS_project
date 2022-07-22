# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:54:29 2021


This module defines the function that runs the the neural network model
and runs the k-fold cross validation
The output is a series of lists with accuracy values and predictor
importance of each fold

Predictor importance is calculated with a very basic leaf one out function
Use the shap module for better results

@author: Javier Muro

"""
from sklearn.model_selection import KFold
from sklearn import metrics
#from sklearn import preprocessing
#from sklearn.preprocessing import Normalizer

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#Import function to display loss
from plot_loss import plot_loss

#Preprocess data
import be_preprocessing
import modelDNN

# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset

# Create empty lists to store results of folds
RMSE_test_list = []
RRMSE_test_list = []
RMSE_val_list = []
rsq_list = []

predictions_list = []
pred_trues = []



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
        model.summary()
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
            ,callbacks=[es])
            
        #######################################################################
        #Plot errors
        # Show last few epochs in history
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
           
        plot_loss(history, EPOCHS, studyvar)
        
        # Measure this fold's RMSE using the validation (0.2%) data
        RMSE_val = hist[(hist.epoch == (EPOCHS - 1))][['val_root_mean_squared_error']].squeeze()
        print(f"RMSE validation data: {RMSE_val}")
        
        #Predictions
        #Make predictions on the test data using the model, and stored results of each fold
        test_predictions = model.predict(test_features).flatten()
        #predictions_list.extend(test_predictions)
        
        c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1)
        pred_trues.append(c)
        
        # Measure this fold's RMSE using the test data
        RMSE_test = np.sqrt(metrics.mean_squared_error(test_predictions,test_labels))
        #print(f"RMSE test data: {RMSE_test}")
        
        # Calculate r2 between predicted and test data
        linreg = sp.stats.linregress(test_predictions, test_labels)
        rsq = linreg.rvalue **2
        #rsq = metrics.r2_score(test_predictions, test_labels)

        rsq_list.append(rsq)
        # p = linreg.pvalue
        # p_list.append(p)
            
        #Calculate the relative root mean squared error
        test_mean = np.mean(test_labels)
        RRMSE_test = (RMSE_test / test_mean)
        RMSE_test_list.append(RMSE_test)
        RRMSE_test_list.append(RRMSE_test)
                
    return model
          
if __name__ == "__main__":
    kfold_DNN(EPOCHS, studyvar)