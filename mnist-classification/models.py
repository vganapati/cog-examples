#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:05:10 2022

@author: vganapa1
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import os

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

def model_instance(restore=False,
                   checkpoint_dir = 'training_checkpoints'):
  # Create an instance of the model
  model = MyModel()
    
  # create checkpoint
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(model = model)    
    
  if restore: # restore a checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  return(model, checkpoint, checkpoint_prefix)

