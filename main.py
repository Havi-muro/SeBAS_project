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

cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project

#conda activate earth-analytics-python

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import statistics

# Preprocess data
# in be_preprocessing.py, select the study variable and the predictors
import be_preprocessing

# show fewer decimals at print
pd.options.display.float_format = '{:.1f}'.format


# Create an object with the result of  the preprocessing module
# We have to define this to explore the dataset we work with
# and to relate results to other variables in the plots afterwards

Mydataset = be_preprocessing.Mydataset
studyvar = 'SpecRichness'
MydatasetLUI = be_preprocessing.MydatasetLUI
print(Mydataset.head())
print(list(Mydataset.columns))

##############################################################################

# 1.- K-fold with DNN
import kfold_DNN

EPOCHS = 500

kfold_DNN.kfold_DNN(EPOCHS, studyvar)

#Put results as variables in global environment
RMSE_test_list = kfold_DNN.RMSE_test_list
RRMSE_test_list = kfold_DNN.RRMSE_test_list
rsq_list = kfold_DNN.rsq_list

# We build a df of the accumulated predictions vs labels
pred_trues = kfold_DNN.pred_trues
pred_truesdf = pd.concat(pred_trues).reset_index(drop=True)
pred_truesdf.columns = ['labels','preds']

##############################################################################

# 2.- kfold approach with RF
import kfold_RF
kfold_RF.kfold_RF(studyvar)

# Put results as variables in global environment
RMSE_test_list = kfold_RF.RMSE_test_list
RRMSE_test_list = kfold_RF.RRMSE_test_list
rsq_list = kfold_RF.rsq_list

predictions_list = kfold_RF.predictions_list

importance_list = kfold_RF.importance_list

##############################################################################
# 3.- Spatial cross-validation approach with DNN by selecting
# training and test samples according to exploratory
# It is necessary to keep the column "explo" in be_preprocessing.py

import spcv_DNN

# Choose which site is used for test and which one(s) for training
EPOCHS = 200
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

#print average RMSE
print(f"RMSE test Mean: {statistics.mean(RMSE_test_list)}")
print(f"RMSE test StdDev: {statistics.stdev(RMSE_test_list)}")

#print average RRSME
print(f"RRMSE test Mean: {statistics.mean(RRMSE_test_list)}") 
print(f"RRMSE test StDev: {statistics.stdev(RRMSE_test_list)}")

#print average r squared in validation
print(f"r squared Mean: {statistics.mean(rsq_list)}") 
print(f"r squared StdDev: {statistics.stdev(rsq_list)}")


###############################################################################
# Plots
###############################################################################

#Make a density plot
from scipy.stats import gaussian_kde

y = pred_truesdf['preds']
x = pred_truesdf['labels']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)

#plt.xlabel('Biomass $(g/m^{2})$')
#plt.ylabel('Predicted biomass $(g/m^{2})$')
plt.ylabel(f'Predicted plant height')
plt.xlabel(f'In situ plant height')
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))
#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()

#fig.savefig(f'results/{studyvar} allfolds_densitypolot_DNN.svg')

###############################################################################

# Merge the predicted results with the original dataset
LUI_pred_vs_trues = pd.merge(MydatasetLUI,pred_truesdf, 
                       how = 'right',
                       left_index = True, 
                       right_index = True)

LUI_pred_vs_trues = LUI_pred_vs_trues.rename(columns = {'LUI_2015_2018':'LUI'})


# Plot predictions color coded with LUI
myplot = sns.scatterplot(data=LUI_pred_vs_trues,
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
###############################################################################

"""
% Matlab function to calculate model evaluation statistics 
% S. Robeson, November 1993
%
% y(1):  mean of observed variable 
% y(2):  mean of predicted variable 
% y(3):  std dev of observed variable 
% y(4):  std dev of predicted variable 
% y(5):  correlation coefficient
% y(6):  intercept of OLS regression
% y(7):  slope of OLS regression
% y(8):  mean absolute error (MAE)
% y(9):  index of agreement (based on MAE)
% y(10): root mean squared error (RMSE)
% y(11): RMSE, systematic component
% y(12): RMSE, unsystematic component
% y(13): index of agreement (based on RMSE)

""" 


n = len(pred_truesdf['labels'])
so = pred_truesdf['labels'].sum()
sp = pred_truesdf['preds'].sum()

sumo2 = (pred_truesdf['labels']**2).sum()
sump2 = (pred_truesdf['preds']**2).sum()

sum2 = ((pred_truesdf['labels']-pred_truesdf['preds'])**2).sum()
sumabs = abs(pred_truesdf['labels']-pred_truesdf['preds']).sum()

sumdif = (pred_truesdf['labels']-pred_truesdf['preds']).sum()
cross = (pred_truesdf['labels']*pred_truesdf['preds']).sum()

obar = pred_truesdf['labels'].mean()
pbar = pred_truesdf['preds'].mean()

import math
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
rmses = math.sqrt(mses)
rmseu = math.sqrt(mseu)


pe1 = (abs(pred_truesdf['preds']-obar) + abs(pred_truesdf['labels']-obar)).sum()
pe2 = ((abs(pred_truesdf['preds']-obar) + abs(pred_truesdf['labels']-obar))**2).sum()
d1 = 1 - n*mae/pe1;
d2 = 1 - n*mse/pe2;

y = [obar,pbar,sdo,sdp,r,a,b,mae,d1,rmse,rmses,rmseu,d2]   

# pred_truesdf.to_csv('results/testa.csv')  