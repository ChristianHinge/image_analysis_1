# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:05:36 2021

@author: simon
"""

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf 
from keras import backend as K
import numpy as np 
import matplotlib.pyplot as plt
from dataloader_test import DataLoader, IDs, load_train_val_data
from keras.models import load_model
from tensorflow.python.client import device_lib
from scipy.spatial.distance import dice
from scipy.spatial.distance import jaccard
from Unet_works import Unet

print(device_lib.list_local_devices())

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%

slice_ID = "pt_094_sl_23"

unet_model_path = "checkpoints/first low LR model/model_120.hdf5"

classification_model_path = "checkpoints/first low LR model/model_120.hdf5"




