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


def plot_loss(history, EPOCHS, studyvar):
  Mydataset = be_preprocessing.be_preproc(studyvar)[0]  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylim([0, max(Mydataset[studyvar])])
  plt.xlim([0,EPOCHS])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error {studyvar}')
  #plt.legend()
  plt.grid(True)