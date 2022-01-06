"""
Created on Mon Dec 14 10:08:22 2020
@author: Janny
This code is a little but trickier. It is a stand-alone script, meaning that
it doesn't need our other modules. It uses a slightly different normalization,
which is not included in the model structure.

It gives you the variable importance of each feature in fold

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.preprocessing import StandardScaler

import shap
shap.initjs()

cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project
# Load datasets
Mydataset_0 = pd.read_csv ('data/Bexis_S1S2_height_NMDS_RaoQ_Dec2021.csv')

# The year and the ep have been concatenated to sort the observations by
# Exoloratory, plot number and year so that: 
# A01-2017,H01-2017,S01-2017,A01-2018,H01-2018,S01-2018...
# This allows a non-shuffeled kfold cross validation to have
# all observations/replicates from e.g. A01 for all years in either training or validation

Mydataset_0 = Mydataset_0.sort_values(by='yep')

#Mydataset_0 = Mydataset_0[Mydataset_0['explo']=='SCH']

Mydataset_0.head(5)


# Replace missing values in Sentinel-1 data with median values
medVH = Mydataset_0['VHMean_May'].median()
medVV = Mydataset_0['VVMean_May'].median()
medVVVH = Mydataset_0['VVVH'].median()

Mydataset_0['VHMean_May'] = Mydataset_0['VHMean_May'].fillna(medVH)
Mydataset_0['VVMean_May'] = Mydataset_0['VVMean_May'].fillna(medVV)
Mydataset_0['VVVH'] = Mydataset_0['VVVH'].fillna(medVVVH)


# Select study variable and covariates
Mydataset_vars = Mydataset_0.drop(['x', 'y', 
    'explo', 
     'yep',
              #'Year', 'ep',
              'SpecRichness',
              'height_cm',
              "biomass_g", 
              'Shannon',
              'Simpson',
              'FisherAlpha',
              'PielouEvenness',
              'number_vascular_plants',
              'NMDS1',
              'NMDS2',
              #'SpecRich_157',
              'Rao_Q',
              'Redundancy',
              
              "LUI_2015_2018",
              
              "SoilTypeFusion" ,
              'LAI',
              'slope',
              'aspect',
              'blue','green', 'red', 'nir', 'nirb', 're1','re2','re3', 'swir1', 'swir2',
              
              'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI',
              'VHMean_May',
              'VVMean_May',
              'VVVH',
              'TWI'
             
       ], axis=1)

list(Mydataset_vars.columns)

# Make lists to remove correlated bands for species diversity predictions
removeswir2 = list(Mydataset_vars.filter(regex ='^swir2'))
removeblue = list(Mydataset_vars.filter(regex ='^blue'))
removere1 = list(Mydataset_vars.filter(regex ='^re1'))
removere2 = list(Mydataset_vars.filter(regex ='^re2'))

removevars = removeblue+removeswir2

Mydataset_vars=Mydataset_vars.drop(removevars, axis=1)


# Select variables for biomass and height predictions
Mydataset_vars = Mydataset_0[['Year', 'ep', 'SpecRich_157', 
                                'blue_3','green_3', 'red_3', 'nir_3', 'nirb_3', 're1_3', 're2_3', 're3_3', 'swir1_3', 'swir2_3'
                                #,'blue_sd_3','green_sd_3', 'red_sd_3', 'nir_sd_3', 'nirb_sd_3', 're1_sd_3', 're2_sd_3', 're3_sd_3', 'swir1_sd_3', 'swir2_sd_3'
                                ,'EVI','SAVI', 'GNDVI', 'ARVI', 'CHLRE', 'MCARI','NDII','MIRNIR', 'MNDVI', 'NDVI'
                                ,'VHMean_May','VVMean_May','VVVH'
                                #,'SoilTypeFusion'
                                ,'slope', 'aspect'
                                ,'TWI'
                                ]]

# nir_3 corresponds to Mid may. 
# The differences in correlation between the orignial band and nir_3
# correspond to SCH, because the original is two weeks later
studyvar = 'SpecRich_157'


#replace nas with mean?
Mydataset = Mydataset_vars.dropna()

#08 swampy
#10 dry soil
#12 path inside
#16 no management for 2 years
#19 water
#33 partially grazed
#35 very dry
#38 swampy and dung
#40 dry

#Some plots have trees inside
trees = ['AEG07','AEG09','AEG25','AEG26','AEG27','AEG47','AEG48','HEG09','HEG21','HEG24','HEG43']

#others are too dry or swamped at moment of biomass collection
outliers2018 = ['SEG08', 'SEG10', 'SEG11', 'SEG12', 'SEG16', 'SEG18', 'SEG19', 'SEG20', 'SEG31',  
'SEG33', 'SEG35', 'SEG36', 'SEG38', 'SEG39', 'SEG40', 'SEG41', 'SEG44', 'SEG46', 'SEG45', 'SEG49', 'SEG50']

#Filter those and drop the variables not needed
Mydataset = Mydataset.drop(Mydataset[(Mydataset['Year'] == 2018) & (Mydataset['ep'].isin(outliers2018))].index)
Mydataset = Mydataset.drop(Mydataset[Mydataset['ep'].isin(trees)].index)

Mydataset = Mydataset.drop(['Year', 'ep'], axis=1)

#Soil and explo arecategorica variables. Change it to one-hot encoded
Mydataset = pd.get_dummies(Mydataset, prefix='', prefix_sep='')
print(Mydataset.head())


# define model
def build_model():
  model = keras.Sequential([
    #normalizer,
    layers.Dense(64, 
                 activation='relu', 
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 input_shape=[len(train_features.keys())]), #!!! We had to change here to get the shape from the np array
       
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 ),
    
    layers.Dense(64, activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model

def plot_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  plt.legend()
  plt.grid(True)

#Create y (labels) and x (features)
x_columns = Mydataset.columns.drop(studyvar)
x = Mydataset[x_columns]
y = Mydataset[studyvar]

# Define the K-fold Cross Validator
kfold = KFold(5, shuffle=False)

RMSE_test_list = []
RRMSE_test_list = []
RMSE_val_list = []
rsq_list = []

var_imp_list = []

# K-fold Cross Validation model evaluation
fold = 0
for train, test in kfold.split(x):
    fold+=1
    print(f'Fold#{fold}')
        
    train_features = x.iloc[train]
    train_labels = y.iloc[train]
    test_features = x.iloc[test]
    test_labels = y.iloc[test]

###############################################################################
    # Normalzation
###############################################################################
  
    scaler = StandardScaler()
    scaler.fit(train_features)
    normed_train_features= scaler.transform(train_features)
    normed_test_features= scaler.transform(test_features)

###############################################################################
    model = build_model()
    #model.summary()
    EPOCHS = 200
###############################################################################
    #Train model
    history = model.fit(
        normed_train_features, 
        train_labels, 
        epochs=EPOCHS, 
        validation_split = 0.2, 
        verbose=0)
###############################################################################
    #Plot errors
    # Show last few epochs in history
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
       
    #plot_loss(history)
    
    # Measure this fold's RMSE using the validation (0.2%) data
    RMSE_val = hist[(hist.epoch == (EPOCHS - 1))][['val_root_mean_squared_error']].squeeze()
    #print(f"RMSE validation data: {RMSE_val}")
    
    #Predictions
    #Make predictions on the test data using the model 
    test_predictions = model.predict(normed_test_features).flatten()
    
    # Measure this fold's RMSE using the test data
    RMSE_test = np.sqrt(metrics.mean_squared_error(test_predictions,test_labels))
    #print(f"RMSE test data: {RMSE_test}")
    
    # Calculate r2 between predicted adn test data
    linreg = sp.stats.linregress(test_predictions ,test_labels)
    rsq = linreg.rvalue **2
    rsq_list.append(rsq)
        
    #Calculate the relative root mean squared error
    test_mean = np.median(test_labels)
    RRMSE_test = (RMSE_test / test_mean)
    RMSE_test_list.append(RMSE_test)
    RRMSE_test_list.append(RRMSE_test)
    
    explainer = shap.DeepExplainer(model,normed_train_features)

    shap_values = explainer.shap_values(normed_train_features)

    shap_plot = shap.summary_plot(shap_values, 
                                  feature_names= x.columns,  
                                  plot_type="bar", 
                                  max_display=33)
    
    # Shap_values is a list of matrices. We have to change that.
    vals = np.abs(shap_values).mean(0)
    # Get the feature names
    feature_names = x.columns

# Join in a dataframe the shap_values with their corresponding feature names
# Order remains the same than in the initial df
    feature_importance = pd.DataFrame(list(zip(feature_names, sum(vals))),
                                      columns=['col_name','feature_importance_vals'])
    
    var_imp_list.append(feature_importance)

     
# print average RMSE
print(f"RMSE test Mean: {statistics.mean(RMSE_test_list)}")
print(f"RMSE test StdDev: {statistics.stdev(RMSE_test_list)}")

# print average RRSME
print(f"RRMSE test Mean: {statistics.mean(RRMSE_test_list)}") 
print(f"RRMSE test StDev: {statistics.stdev(RRMSE_test_list)}")

# print average r squared in validation
print(f"r squared Mean: {statistics.mean(rsq_list)}") 
print(f"r squared StdDev: {statistics.stdev(rsq_list)}")

#concatenate all variable importances side by side    
var_imp_df = pd.concat(var_imp_list, axis=1)

# Select useful columns
var_imp_df= var_imp_df.iloc[:,[0,1,3,5,7,9]].set_index('col_name')

# Calculate the mean of each variable importance across folds
var_imp_mean = var_imp_df.mean(axis=1)
var_imp_sd = var_imp_df.std(axis=1)

var_imp = pd.concat([var_imp_mean, var_imp_sd], axis=1)

var_imp.to_csv(f'var_imp_kfold_DNN_{studyvar}.csv')

# Dependence plots
# The partial dependence plot shows the marginal effect 
# one or two features have on the predicted outcome. 
# It tells whether the relationship between the target and a feature is linear, 
# monotonic or more complex. It automatically includes another variable that 
# your chosen variable interacts most with.

 shap.dependence_plot('re2_10', shap_values[0], train_features)

##############################################################################
#                              Tests & notes
##############################################################################
# print the JS visualization code to the notebook


# In case we need to flaten
# def flatten(t):
#     t = t.reshape(1, -1)
#     t = t.squeeze()
#     return t
# train_features_np = flatten(train_features)

# In case we need to subset the records
# select a set of background examples to take an expectation over
#background = normed_train_features[np.random.choice(normed_train_features.shape[0],100, replace=False)]

# explain predictions of the model on the background set
explainer = shap.DeepExplainer(model,normed_train_features)

# or pass the tensos directly
# explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)

# We might get a common error here, 
#raise KeyError(key) from err
#KeyError: 0
# https://github.com/slundberg/shap/issues/1456
# perhaps because the output shape is None, 10, 64 (run model.summary())
# solved if we work with dfs
shap_values = explainer.shap_values(normed_train_features)

# Display variable importance
shap_plot = shap.summary_plot(shap_values, 
                              feature_names= x.columns,  
                              plot_type="bar", 
                              max_display=33)

#Display interaction plot. doesn't work.
shap.dependence_plot("swir1_2", shap_values.numpy(), x)

# Extract importance values to table

# Shap_values is a list of matrices. We have to change that.
vals = np.abs(shap_values).mean(0)
# Get the feature names
feature_names = x.columns

# Join in a dataframe the shap_values with their corresponding feature names
# Order remains the same than in the initial df
feature_importance = pd.DataFrame(list(zip(feature_names, sum(vals))),
                                  columns=['col_name','feature_importance_vals'])

# We could sort the values according to their importance
# but we might want to keep cronological order
#feature_importance.sort_values(by=['feature_importance_vals'],
#                               ascending=False, inplace=True)
feature_importance.head()

feature_importance.to_csv('results/feature_importance_test2.csv')


x_cat = x.copy()
soil_decoding = {
    0: 'Braunerde',
    1: 'Erdniedermoor',
    2: 'Fahlerde',
    3: 'Mulmniedermoor',
    4: 'Parabraunerde',
    5: 'Pseudogley',
    6: 'Rendzina'
}
x_cat['SoilTypeFusion'] = x_cat['SoilTypeFusion'].map(soil_decoding)

X_cat.head(3)



# In case we need to change the shape of the model.
# We set a new output shape, but I don't know if it is right.
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

newInput = Input(batch_shape=(1,64)) # TODO: try (1,)
newOutputs = model(newInput)
newModel = Model(newInput,newOutputs)


##############################################################################


plt.scatter(test_labels, test_predictions)
plt.xlim(min(Mydataset[studyvar]), max(Mydataset[studyvar]))
plt.ylim(min(Mydataset[studyvar]), max(Mydataset[studyvar]))
plt.ylabel('Predicted')
plt.xlabel('True values')
#add a r=1 line
x = np.array([0,max(Mydataset[studyvar])])
plt.plot(x,x,lw=1, c="black")
plt.show()