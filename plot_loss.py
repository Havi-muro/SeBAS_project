# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:52:55 2021

Plot loss history of DNN

@author: rsrg_javier
"""
import matplotlib.pyplot as plt
#Preprocess data
import be_preprocessing

# Create an object with the result of  the preprocessing module
Mydataset = be_preprocessing.Mydataset

def plot_loss(history, EPOCHS, studyvar):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  plt.legend()
  plt.grid(True)