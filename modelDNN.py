# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:35 2021

This module defines the DNN structure of the neural network model

@author: Javier Muro

"""
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
                 input_shape=train_features.shape), #!!! We had to change here to get the shape from the np array
       
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(0.01),
                 ),
    
    layers.Dense(64, activation='relu',
                  kernel_regularizer=keras.regularizers.l1(0.01),
                  ),
    
    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mae',
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #metrics=['mae','mse'])
  return model

if __name__ == "__main__":
    build_model(normalizer, train_features)