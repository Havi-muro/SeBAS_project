# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:08:22 2020
@author: Javier Muro

This code imports and executes the modules to preprocess the data 
(be_preprocessing). the user can train and validate a random forest (kfold_RF)
or a neural network (modelDNN and kfold_DNN). 
Training and validation points are distributed following a kfold approach
across the three exploratories. Variable importances are also calculated. 

Alternatively, the user can perform a spatial cross-validation, selecting
two exploratories for training and the remaining for validation.

Data is normalized using the preprocessing.Normalzation() function

TODO: lists created by modules do not empty 
when running the model a second time. Have to fix

@author: Javier Muro
"""

#conda activate earth-analytics-python

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import statistics
import math
import os
from scipy.stats import gaussian_kde

os.chdir(os.path.join('C://','Users','rsrg_javier','Documents','GitHub','SeBAS_project'))

# Preprocess data
# in be_preprocessing.py, select the  the predictors
import be_preprocessing

# select the study variable ('SpecRichness' or 'biomass_g') and instantiate 
# the resulting dataframes. One is the dataframe to analyze, and MydatasetLUI
# contains other info such as LUI to plot the predictions colored by 
# another environmental variable
studyvar = 'SpecRichness'

Mydataset = be_preprocessing.be_preproc(studyvar)[0]

# We can create an aditional dataset that contains other data
# we use it to relate results to other variables in the plots afterwards
MydatasetLUI = be_preprocessing.be_preproc(studyvar)[1]
print(Mydataset.head())
print(list(Mydataset.columns))

##############################################################################

# 1.- K-fold with DNN
import gkfold_DNN

EPOCHS = 500

gkfold_DNN.gkfold_DNN(EPOCHS, studyvar)

# build a df of the accumulated predictions vs labels
pred_trues = gkfold_DNN.pred_trues
pred_truesdf = pd.concat(pred_trues)

# build also a df with the test features in the order in which they were split
testfeatures_order = gkfold_DNN.testfeatures_order2
testfeatures_orderdf = pd.concat(testfeatures_order)

# concatenate both laterally. They should correspond.
rsdf = pd.concat([testfeatures_orderdf, pred_truesdf], axis=1)

# merge it with the original dataset using the features as joining field
# This allows us to display preds vs lables along with other variables
Mypredictions = pd.merge(rsdf, MydatasetLUI)

##############################################################################

# 2.- kfold approach with RF
import gkfold_RF
gkfold_RF.gkfold_RF(studyvar)

# build a df of the accumulated predictions vs labels
pred_trues = gkfold_RF.pred_trues
pred_truesdf = pd.concat(pred_trues)

# build also a df with the test features in the order in which they were split
testfeatures_order = gkfold_RF.testfeatures_order2
testfeatures_orderdf = pd.concat(testfeatures_order)

# concatenate both laterally. They should correspond.
rsdf = pd.concat([testfeatures_orderdf, pred_truesdf], axis=1)

# merge it with the original dataset using the features as joining field
# This allows us to display preds vs lables along with other variables
Mypredictions = pd.merge(rsdf, MydatasetLUI)

importance_list = gkfold_RF.importance_list

##############################################################################
# 3.- Spatial cross-validation approach with DNN by selecting
# training and test samples according to exploratory
# It is necessary to keep the column "explo" in be_preprocessing.py

import spcv_DNN

# Choose which site is used for test and which one(s) for training
EPOCHS = 200
Mydataset['explo'] = MydatasetLUI['explo']
Mydataset=Mydataset.drop('ep', axis=1)
train_dataset = Mydataset[(Mydataset['explo']=='SCH')       
                          | (Mydataset['explo'] == 'HAI')   # take this line out to use only 1 site for training
                          ].drop(['explo'], axis=1)
                            
test_dataset = Mydataset[Mydataset['explo']=='ALB'].drop(['explo'], axis=1)

spcv_DNN.spcv_DNN(EPOCHS, train_dataset, test_dataset, studyvar)

# Put results as variables in global environment

RMSE_test_list = spcv_DNN.RMSE_test_list
RRMSE_test_list = spcv_DNN.RRMSE_test_list
rsq_list = spcv_DNN.rsq_list

predictions_list = spcv_DNN.predictions_list


###############################################################################
# Plots
###############################################################################

#Make a density plot

y = Mypredictions['preds']#.astype(float)
x = Mypredictions['labels']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)

#plt.xlabel('Biomass $(g/m^{2})$')
#plt.ylabel('Predicted biomass $(g/m^{2})$')
plt.ylabel(f'Predicted {studyvar}')
plt.xlabel(f'In situ {studyvar}')
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))
#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()

fig.savefig(f'results/{studyvar}_densitypolot_RF_S1_S2sr_S2ratios_TE.svg')

###############################################################################
# Plot displaying another variable for color

Mypredictions = Mypredictions.rename(columns = {'LUI_2015_2018':'LUI'})

# Plot predictions color coded with LUI
myplot = sns.scatterplot(data=Mypredictions,
                         y='preds',
                         x=studyvar,
                         hue = 'LUI', 
                         palette='viridis',
                         #cmap = 'Reds',
                         linewidth=0,
                         s = 20
                         )
plt.ylabel(f'{studyvar} Predicted')
plt.xlabel(f'{studyvar} insitu values')
myplot.legend(title="LUI")
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))

#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()
#fig.savefig(f'{studyvar}allfolds_plot2.svg')

###############################################################################
# more error metrics
# the 5 fold is run 5 times and average accuracy values are reported
###############################################################################

"""
% Matlab function to calculate model evaluation statistics 
% S. Robeson, November 1993
% Author of matlab function: Zbynek Malenovksy

% zb(1):  mean of observed variable 
% zb(2):  mean of predicted variable 
% zb(3):  std dev of observed variable 
% zb(4):  std dev of predicted variable 
% zb(5):  correlation coefficient
% zb(6):  intercept of OLS regression
% zb(7):  slope of OLS regression
% zb(8):  mean absolute error (MAE)
% zb(9):  index of agreement (based on MAE)
% zb(10): root mean squared error (RMSE)
% zb(11): relative root mean squared error (RMSE)
% zb(12): RMSE, systematic component
% zb(13): RMSE, unsystematic component
% zb(14): index of agreement (based on RMSE)

""" 
# I have to import in each iteration
import gkfold_DNN 
# import kfold_RF


met_ls=[]
for i in range(3):
    
    EPOCHS = 500
    
    # We build a df of the accumulated predictions vs labels
    # for DNN or RF

    gkfold_DNN.gkfold_DNN(EPOCHS, studyvar)
    pred_trues = gkfold_DNN.pred_trues

    #kfold_RF.kfold_RF(studyvar)
    #pred_trues = kfold_RF.pred_trues
    
    pred_truesdf = pd.concat(pred_trues).reset_index(drop=True)
    
    # Select the last batch of predictions.
    # Predictions accumulate when we iterate 10 times
    # even if we delete all the variables
    pred_truesdf = pred_truesdf.tail(Mydataset.shape[0])
    
    # Predictions vs labels
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
    # add a r=1 line
    line = np.array([0,max(Mydataset[studyvar])])
    plt.plot(line,line,lw=1, c="black")
    plt.show()
    
    # Calculate additional metrics
    n = len(x)
    so = x.sum()
    sp = y.sum()
    
    sumo2 = (x**2).sum()
    sump2 = (y**2).sum()
    
    sum2 = ((x-y)**2).sum()
    sumabs = abs(x-y).sum()
    
    sumdif = (x-y).sum()
    cross = (x*y).sum()
    
    obar = x.mean()
    pbar = y.mean()
    
    sdo = math.sqrt(sumo2/n - obar*obar)
    sdp = math.sqrt(sump2/n-pbar*pbar)
    c = cross/n - obar*pbar
    r = c/(sdo*sdp)
    r2 = r**2
    b = r*sdp/sdo
    a = pbar - b*obar
    mse = sum2/n
    mae = sumabs/n
    
    msea = a**2
    msei = 2*a*(b-1)*obar
    msep = ((b-1)**2) *sumo2/n
    mses = msea + msei + msep
    mseu = mse - mses
    
    rmse = math.sqrt(mse)
    rrmse = rmse/obar
    rmses = math.sqrt(mses)
    rmseu = math.sqrt(mseu)
        
    pe1 = (abs(y-obar) + abs(x-obar)).sum()
    pe2 = ((abs(y-obar) + abs(x-obar))**2).sum()
    d1 = 1 - n*mae/pe1;
    d2 = 1 - n*mse/pe2;
    
    zb = [obar,pbar,sdo,sdp,r,a,b,mae,d1,rmse,rrmse,rmses,rmseu,d2]
    
    results = [r2, rrmse, rmses, rmseu]
    
    met_ls.append(results)
 
# Check that length = number of loops    
len(met_ls)

# Extract specific metrics and calculate mean and sd

r2_hat = statistics.mean([x[0] for x in met_ls])
r2_sd = statistics.pstdev([x[0] for x in met_ls])

rrmse_hat = statistics.mean([x[1] for x in met_ls])
rrmse_sd = statistics.pstdev([x[1] for x in met_ls])

rmses_hat = statistics.mean([x[2] for x in met_ls])
rmses_sd = statistics.pstdev([x[2] for x in met_ls])

rmseu_hat = statistics.mean([x[3] for x in met_ls])
rmseu_sd = statistics.pstdev([x[3] for x in met_ls])

print('r2_hat: ' '%.2f'% r2_hat)
print('r2_sd: ''%.2f'% r2_sd)

print('rrmse_hat: ' '%.2f'% rrmse_hat)
print('rrmse_sd: ' '%.2f'% rrmse_sd)

print(f'rmses_hat: ' '%.2f'% rmses_hat)
print(f'rmses_sd: ' '%.2f'%rmses_sd)

print(f'rmseu_hat: ' '%.2f'% rmseu_hat)
print(f'rmseu_sd: ' '%.2f'%rmseu_sd)

# Shut down spider after every run, because some variables
# or coefficients remain stored somewhere
