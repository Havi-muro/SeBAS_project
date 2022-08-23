# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:54:29 2021


This module defines the function that runs the the neural network model
and runs a group  k-fold cross validation
The output is a series of lists with accuracy values and predictor
importance of each fold

Use the shap module for predictor importance

@author: Javier Muro

"""
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn import metrics

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
studyvar = 'SpecRichness'

EPOCHS = 500
# and a list to store results
pred_trues = []
testfeatures_order = []

#def gkfold_DNN(EPOCHS, studyvar):
    #Create y (labels) and x (features)
epg = Mydataset['ep']
x_columns = Mydataset.columns.drop([studyvar, 'ep'])
x = Mydataset[x_columns].values
y = Mydataset[studyvar].values
 
# K-fold Cross Validation model evaluation
gkf = GroupKFold(n_splits=5)
    
fold = 0
for split, (train_index, test_index) in enumerate(gkf.split(x, y, groups=epg)):
    fold+=1
    print(f'Fold#{fold}')
    
    train_features = x[train_index]
    train_labels = y[train_index]
    test_features = x[test_index]
    test_labels = y[test_index]
    
    testfeatures_order= pd.DataFrame(test_features)
    testfeatures_order.columns = x_columns
    testfeatures_order.append(testfeatures_order)
           
    #######################################################################
    # Normalzation
    #######################################################################
   
    #Create normalizer layer and adapt it to our data
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    
    #######################################################################
    model = modelDNN.build_model(normalizer, train_features)
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
        ,callbacks=[es]
        )
        
    #######################################################################
    #Plot errors
    # Show last few epochs in history
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
       
    plot_loss(history, EPOCHS, studyvar)
          
    #Predictions
    #Make predictions on the test data using the model, and stored results of each fold
    test_predictions = model.predict(test_features).flatten()
    
    # cv_preds = cross_val_predict(model, 
    #                          test_features, 
    #                          test_labels, 
    #                          groups = epg)
    
    # create df with the training features and give back the names
    #train_df = pd.DataFrame(train_features)
    #train_df.columns = x_columns
            
    # concatenate labels and predictions
    c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1)
    c.columns = ['labels', 'preds']
    
    # TODO: this should work, but for some reason, the predictions and
    # labels are not in the same order than the original df
    
    # concatenate training df with predictions
    # cd = pd.concat([c, train_df], axis=1).dropna()
    
    # merge training and predictions with original df using the training features
    # cdf = pd.merge(Mydataset, cd, on=list(x_columns), how='right')

    pred_trues.append(c)
    #cv_trues.append(cv_preds)
                        
    return model

    


          
if __name__ == "__main__":
    gkfold_DNN(EPOCHS, studyvar)