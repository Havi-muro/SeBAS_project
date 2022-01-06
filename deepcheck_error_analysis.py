# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:05:48 2022

@author: rsrg_javier
"""

#conda install -c deepchecks deepchecks
from deepchecks import Dataset
from deepchecks.checks import RegressionErrorDistribution

preds_df =pd.DataFrame(kfold_DNN.predictions_list).rename(columns={0:'preds'})
preds_ds = Dataset(preds_df, label='preds')


#model = kfold_DNN.model
# perhaps it is better to use the model calibrated with all observations

RegressionErrorDistribution().run(preds_ds, model)



from deepchecks.checks import ModelErrorAnalysis
ModelErrorAnalysis(min_error_model_score=0.3).run(train_data, test_data, model)

Example of ModelErrorAnalysis result