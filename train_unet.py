# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:14:51 2021

@author: simon
"""

from Unet_works import Unet
from dataloader_test import DataLoader, IDs, preprocess
from data_split import get_data_split_IDs
import os

data_dir = "data/Patients_CT"
norm_dir = "/data/normalized"

cwd = os.getcwd()

#%%

if not os.path.exists(cwd + norm_dir):
    for ID in IDs:
        print(ID)
        pt = ID.split("_")[1]
        sl = ID.split("_")[3]
        preprocess(pt,sl)

#%%

import tensorflow as tf
import os 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))


#%%
from tensorflow import keras
from keras.callbacks import CSVLogger

train_IDs, val_IDs, test_IDs = get_data_split_IDs(IDs)
d_train = DataLoader(train_IDs,batch_size=2)

#'C:/Users/simon/OneDrive/Dokumenter/Adv_image_analysis/training.log'
csv_logger = CSVLogger(cwd + '/training.log')

unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model

Unet_model.fit(d_train,epochs=3, callbacks = [csv_logger])


#val_data = (X_re, Y_re)
#Unet_model.fit(dl,epochs=3,callbacks= [csv_logger], validation_data = val_data)
