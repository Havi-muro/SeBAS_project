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

# Create an object with the result of  the preprocessing module
# We have to define this to explore the dataset we work with
# and to relate results to other variables in the plots afterwards

Mydataset = be_preprocessing.Mydataset
studyvar = 'SpecRich_157'
MydatasetLUI = be_preprocessing.MydatasetLUI
print(Mydataset.head())
print(list(Mydataset.columns))

##############################################################################

# 1.- K-fold with DNN
import kfold_DNN

EPOCHS = 200 

kfold_DNN.kfold_DNN(EPOCHS, studyvar)

#Put results as variables in global environment
RMSE_test_list = kfold_DNN.RMSE_test_list
RRMSE_test_list = kfold_DNN.RRMSE_test_list
rsq_list = kfold_DNN.rsq_list

predictions_list = kfold_DNN.predictions_list

#LOFO_list = kfold_DNN.LOFO_list
#LOFO_Ordered_list = kfold_DNN.LOFO_Ordered_list

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
EPOCHS = 300
train_dataset = Mydataset[(Mydataset['explo']=='HAI')       
                        #  | (Mydataset['explo'] == 'SCH')   # take this line out to use only 1 site for training
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

# Variable importances

###############################################################################

# Very basic function to infer variable importance. Use shap instead.

# list comprehension to get the first column loss
lofo1 = [item[0] for item in LOFO_list]

#transform list to df and rename variables
lofodf = pd.DataFrame(lofo1)
colnames = list(Mydataset.drop(studyvar, axis=1).columns)
lofodf.columns = colnames

sns.barplot(data=lofodf, 
            capsize = 0.1,
            errwidth=1,
            color='green')
plt.ylabel('loss value')

# create stats to export to csv
lofo_stats = lofodf.describe()
lofo_statst = lofo_stats.transpose()
# lofo_statst.to_csv(f'C:/Users/rsrg_javier/Desktop/SEBAS/python/Var_imp_DNN_{studyvar}_ALLvars.csv')

###############################################################################

# RF
varimp = pd.DataFrame(importance_list)
colnames = list(Mydataset.drop(studyvar, axis=1).columns)
varimp.columns = colnames

sns.barplot(data=varimp, 
            capsize = 0.1,
            errwidth=1,
            color='green')
plt.ylabel('loss value')

# create stats to export to csv
varimp_stats = varimp.describe()
varimp_statst = varimp_stats.transpose()
varimp_statst.to_csv(f'results/Var_imp_kfold_RF_{studyvar}.csv')

###############################################################################

# Plots

###############################################################################

# Plot predictions vs labels for all folds
# Create a new column with the results of the predictions in the test_dataset
test_preds = pd.DataFrame(predictions_list)
test_preds.columns = ['preds']
  
# Merge the predicted results with the original dataset
test_preds2 = pd.merge(MydatasetLUI,test_preds, 
                       how = 'right',
                       left_index = True, 
                       right_index = True)

preds_test = test_preds2[['Year', 'ep', studyvar, 'preds']]
#preds_test.to_csv('results/preds_vs_test_SpeccRich_157_RF.csv')

test_preds2 = test_preds2.rename(columns = {'LUI_2015_2018':'LUI'})

lr = sp.stats.linregress(test_preds2['preds'], test_preds2[studyvar])
total_rsq = lr.rvalue **2

#we can use also sns.kdeplot
myplot = sns.scatterplot(data=test_preds2,
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
#plt.savefig(f'{studyvar}allfolds_plot2.svg')# plot without the line

#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()

###############################################################################

###############################################################################

#Make a density plot
from scipy.stats import gaussian_kde

y = test_preds2['preds']
x = test_preds2[studyvar]

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)

#plt.xlabel('Biomass $g/m^{2}$')
#plt.ylabel('Predicted biomass $g/m^{2}$')
plt.ylabel(f'Predicted species richness')
plt.xlabel(f'In situ species richness')
plt.xlim(0, max(Mydataset[studyvar]))
plt.ylim(0, max(Mydataset[studyvar]))

plt.savefig(f'{studyvar} allfolds_densitypolot_RF.svg')# plot without the line

#add a r=1 line
line = np.array([0,max(Mydataset[studyvar])])
plt.plot(line,line,lw=1, c="black")
plt.show()
