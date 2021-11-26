# SeBAS_modules

This project contains the python modules necessary to train and validate a feed-forward neural network (NN) model and implement it onto satellite imagery to create spatially continuous predictions of plant traits e.g. biomass, plant height, or species diversity. It uses tensor flow and keras API.

be_preprocessing contains the data cleaning routine and selection of study variable and predictors.

Once the variables are selected, main.py is used to train and validate the model. Several options are offered:
 1.- k-fold cross validation using a sequential NN
 2.- k-fold cross validation using a random forest 
 3.- a spatial cross validation where exploratories are permuted between training and validating datasets
 
 Variable importances are calculated in main.py with the % increase in the mean squared error for the random forest, and a leaf-one-out approach for the NN.
 However, the stand-alone module Variable_importances_Shap.py contains an better way to derive variable importances of the NN using SHAP (SHapley Additive exPlanations). It calculates the variable importances of each predictor in each neuron/node and estimates their total contribution to the model.
 
 The sub-folder spatial contains a script to export the model (create_DNNmodel_allobs.py). This model is applied to a raster image with the module ApplyDNN_toras.py. The image must have the same band configuration than the variables used to calibrate the model. An example dataset for alb is provided in the subfolder data/rs that contains the 10 bands of sentinel-2 from rsdb of bexis.
