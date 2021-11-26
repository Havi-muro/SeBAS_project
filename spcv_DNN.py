# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:54:29 2021


This module defines the function that runs the the neural network model
and runs the k-fold cross validation
The output is a series of lists with accuracy values and predictor
importance of each fold

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
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.utils import shuffle

#Import function to display loss
from plot_loss import plot_loss

#Preprocess data
import be_preprocessing
import modelDNN

# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset
studyvar = 'biomass_g' 

# Create empty lists to store results of folds
RMSE_test_list = []
RRMSE_test_list = []
RMSE_val_list = []
rsq_list = []

predictions_list = []

LOFO_list = []
LOFO_Ordered_list = []

def spcv_DNN(EPOCHS, train_dataset, test_dataset):
 
    for iteration in range(5):
    # Recommended to shuffle here, since despite the shuffle is True by default when we fit the model,
    # the validation data has to be suffled before it is separated from the training data
        #Mydataset = shuffle(Mydataset)
        
       
        # Copy features
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        
        # Create y variables (labels) and x (features)
        train_labels = train_features.pop(studyvar)
        test_labels = test_features.pop(studyvar)
               
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
       
        #Train model
        #Keras fit function expects training features to be as array.
        history = model.fit(
            train_features, 
            train_labels, 
            epochs=EPOCHS, 
            validation_split = 0.2, 
            verbose=0)
            
        #######################################################################
        #Plot errors
        # Show last few epochs in history
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
           
        plot_loss(history, EPOCHS)
        
        # Measure this fold's RMSE using the validation (0.2%) data
        RMSE_val = hist[(hist.epoch == (EPOCHS - 1))][['val_root_mean_squared_error']].squeeze()
        #print(f"RMSE validation data: {RMSE_val}")
        
        #Predictions
        #Make predictions on the test data using the model, and stored results of each fold
        test_predictions = model.predict(test_features).flatten()
        predictions_list.extend(test_predictions)
        
        # Measure this fold's RMSE using the test data
        RMSE_test = np.sqrt(metrics.mean_squared_error(test_predictions,test_labels))
        #print(f"RMSE test data: {RMSE_test}")
        
        # Calculate r2 between predicted and test data
        linreg = sp.stats.linregress(test_predictions ,test_labels)
        rsq = linreg.rvalue **2
        rsq_list.append(rsq)
        p = linreg.pvalue
        # p_list.append(p)
            
        #Calculate the relative root mean squared error
        test_mean = np.mean(test_labels)
        RRMSE_test = (RMSE_test / test_mean)
        RMSE_test_list.append(RMSE_test)
        RRMSE_test_list.append(RRMSE_test)
        
        # # Function to calculate predictors importance via leave one out
        # def LOFO(model, X, Y):
        #     OneOutScore = []
        #     n = X.shape[0]
        #     for i in range(0,X.shape[1]):
        #         newX = X.copy()
        #         newX[:,i] = 0 #np.random.normal(0,1,n) #I had to change this from newX.iloc[:,i] because I am not working with pd. dataframe but with numpy arrays
        #         OneOutScore.append(model.evaluate(newX, Y, verbose=0))
        #     OneOutScore = pd.DataFrame(OneOutScore[:])
        #     ordered = np.argsort(-OneOutScore.iloc[:,0])
        #     return(OneOutScore, ordered)
    
        # # apply on the model
        # LOFO, LOFO_Ordered = LOFO(model, train_features, train_labels)
        # LOFO_list.append(LOFO)
        # LOFO_Ordered_list.append(LOFO_Ordered)
          
if __name__ == "__main__":
    spcv_DNN(EPOCHS, train_dataset, test_dataset)