# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:44:48 2022

Pixel statistics

@author: rsrg_javier
"""
cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

##############################################################################
# Process data
##############################################################################

# Data with the response variable and time series pixel values
df1_0 = pd.read_csv('data/Bexis_data_jonnas_Spekker.csv')

# Data with standard deviation pixel values for May
sd_0 = pd.read_csv('data/Bexis_S1S2_TS_sd_height_NMDS_RaoQ_Dec2021.csv')

# Data with pixel values for May from the holy area.
holy_0 = pd.read_csv('data/Biomass_S2_Holy_hai2017fixed.csv')

#Drop unused columns
df1 = df1_0.drop(['number_vascular_plants','explo',
                 'SpecRichness',
                 'height_cm',
                 'biomass_g',
                 'NMDS1',
                 'NMDS2',
                 'Shannon',
                 'Simpson',
                 'FisherAlpha',
                 'PielouEvenness',
                 'LUI_2015_2018',
                 'SoilTypeFusion',
                 'slope',
                 'aspect',
                 'S2Q',
                 'TWI'], axis=1)

holy = holy_0.drop(['Unnamed: 0','explo'], axis=1).rename(columns={'year':'Year'})

# get only the sd columns
sd1 = sd_0.filter(like='sd_')
sd2 = sd_0[['Year', 'ep']]
sd3 = [sd1, sd2]
sd4 = pd.concat(sd3, axis=1)

#Merge the pixel values from the holy area and the median pixel values
df_holy = df1.merge(holy, on =['ep', 'Year'])

#Create indices with the median and pixel values
df_holy['ndvi_median'] = (df_holy['nir_3']-df_holy['red_3'])/(df_holy['nir_3']+df_holy['red_3'])
df_holy['ndvi_pixel'] = (df_holy['nir']-df_holy['red'])/(df_holy['nir']+df_holy['red'])

df_holy['nir-red_median'] = (df_holy['nir_3']/df_holy['red_3'])/100
df_holy['nir-red_pixel'] = (df_holy['nir']/df_holy['red'])/100


# Plot using the ratios, ndvi or nir values
fig = sns.regplot(data=df_holy, y='ndvi_median', x='ndvi_pixel', 
            ci=100, 
            fit_reg=True, 
            robust=True,
            n_boot=100,
            line_kws={"color": "red"})


fig = sns.regplot(data=df_holy, y=df_holy['nir_3']/100, x=df_holy['nir']/100, 
            ci=100, 
            fit_reg=True, 
            robust=True,
            n_boot=100,
            line_kws={"color": "red"})
plt.xlabel('nir (%) of single pixel')
plt.ylabel('nir (%) median of 50x50 m plot')

plt.savefig('nir pixel vs median.svg')


##############################################################################
# Plot all the median  of all observations sorted from lower to higher and their standard deviation

# Merge df with sd values
dfsd = df1.merge(sd4, on=['Year', 'ep'])

# Lazy way of getting field 0-571 to sort observations for the x axis
dfsd1 = dfsd[['nir_3', 'nir_sd_3','ep', 'Year']].sort_values(by='nir_3').reset_index().reset_index()

# Create upper and lower boundaries with the std
dfsd1['upper'] = dfsd1['nir_3']+dfsd['nir_sd_3']
dfsd1['lower'] = dfsd1['nir_3']-dfsd['nir_sd_3']

fig = sns.lineplot(data=dfsd1, x='level_0', y='nir_3')
plt.xlabel('Observations sorted by nir values')
plt.ylabel('nir reflectance * 10000')

plt.fill_between(x=dfsd1['level_0'], y1=dfsd1['upper'], y2=dfsd1['lower'], alpha=0.2, color='green')

# plt.savefig('nir median with sd.svg')

##############################################################################
# Plot the histogram of the std for a ratio (dfsd2) or for nir (dfsd1)
dfsd2 = dfsd[['nir_3', 'red_3','red_sd_3', 'nir_sd_3','ep', 'Year']]

# Attention!!! I cannot compute a ratio of the standard deviation
# I should have computed the standard deviation of the ratio
# dfsd2['red_sd_3_sd'] = dfsd2['red_sd_3']/dfsd2['red_3']
# dfsd2['nir_sd_3_sd'] = dfsd2['nir_sd_3']/dfsd2['nir_3']
# dfsd2['ratio_sd'] = dfsd2['nir_sd_3_sd'] / dfsd2['red_sd_3_sd']


fig = sns.histplot(dfsd2['nir_sd_3']/100, binwidth=0.20, kde=True)
plt.xlabel('Standard deviation of nir reflectance (%) from S2')
plt.axvline(dfsd2['nir_sd_3'].median()/100,
            color='red')
plt.savefig('Standard deviation of nir reflectance (%).svg')





























##############################################################################

# Prepare dataset to model biomass with holy area values
Mydataset = dff[['yep', 'biomass_g', 'blue',
 'green',
 'red',
 'nir',
 'nirb',
 're1',
 're2',
 're3',
 'swir1',
 'swir2']].sort_values(by='yep').drop('yep', axis=1).dropna()

studyvar='biomass_g'
EPOCHS=500


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# define model
def build_model(normalizer, train_features):
  model = keras.Sequential([
    normalizer,
    layers.Dense(64, 
                 activation='relu', 
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 input_shape=train_features.shape),#!!! We had to change here to get the shape from the np array
   
    #layers.Dropout(0.2),
        
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 ),
    
    #layers.Dropout(0.2),

    layers.Dense(64, activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    layers.Dropout(0.2),

    
    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mae',
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model

def plot_loss(history, EPOCHS, studyvar):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  plt.legend()
  plt.grid(True)

# Set k-fold pipeline

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
        model = build_model(normalizer, train_features)
        #model.summary()
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
        
        
        # Measure this fold's RMSE using the validation (0.2%) data
        RMSE_val = hist[(hist.epoch == (EPOCHS - 1))][['val_root_mean_squared_error']].squeeze()
        print(f"RMSE validation data: {RMSE_val}")
        
        #Predictions
        #Make predictions on the test data using the model, and stored results of each fold
        test_predictions = model.predict(test_features).flatten()
        #predictions_list.extend(test_predictions)
        
        c = pd.concat([pd.Series(test_labels), pd.Series(test_predictions)], axis=1 )
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

kfold_DNN(EPOCHS, studyvar)

import statistics
#print average RMSE
print(f"RMSE test Mean: {statistics.mean(RMSE_test_list)}")
print(f"RMSE test StdDev: {statistics.stdev(RMSE_test_list)}")

#print average RRSME
print(f"RRMSE test Mean: {statistics.mean(RRMSE_test_list)}") 
print(f"RRMSE test StDev: {statistics.stdev(RRMSE_test_list)}")

#print average r squared in validation
print(f"r squared Mean: {statistics.mean(rsq_list)}") 
print(f"r squared StdDev: {statistics.stdev(rsq_list)}")

#Make a density plot
from scipy.stats import gaussian_kde

# We build a df of the accumulated predictions vs labels
pred_truesdf = pd.concat(pred_trues).reset_index(drop=True)
pred_truesdf.columns = ['labels','preds']

y = pred_truesdf['preds']
x = pred_truesdf['labels']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)

plt.ylabel(f'Predicted {studyvar}')
plt.xlabel(f'In situ {studyvar}')
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))

#plt.savefig(f'{studyvar} allfolds_densitypolot_RF.svg')# plot without the line

#add a r=1 line
line = np.array([0,max(pred_truesdf['labels'])])
plt.plot(line,line,lw=1, c="black")
plt.show()