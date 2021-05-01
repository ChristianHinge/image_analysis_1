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

data_dir = "data/normalized"



unet_model_path = "checkpoints/0.1_step_decay_LR/model_120.hdf5"
swap = True 

classification_model_path = "checkpoints/first low LR model/model_120.hdf5"



#%% load patient data 

pt = slice_ID.split("_")[1]
sl = slice_ID.split("_")[3]

# Load bone and brain slice
if swap:
    im_bone  = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
    im_brain = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
    im_seg = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
else:
    im_bone  = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
    im_brain = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
    im_seg = np.load(data_dir + f"/{pt}/seg/{sl}.npy")   
    
    
X_infer = np.zeros((1,512,512,2))
Y_seg_true = np.zeros((1,512,512))

X_infer[0,:,:,0] = im_bone
X_infer[0,:,:,1] = im_brain
Y_seg_true = im_seg

#%%


##### UNET PREDICTION #######
BS = 2

Unet_model = load_model(unet_model_path)

Y_seg_pred = np.round(Unet_model.predict(X_infer,batch_size=BS).squeeze())

#%%

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)

if swap:
    plt.imshow(X_infer[0,:,:,0])
else:
    plt.imshow(X_infer[0,:,:,1])

plt.subplot(1,2,2)
plt.imshow(Y_seg_pred[:,:])


#%%

##### CLASSIFICATION PREDICTION ######
# X_class = np.zeros((1,512,512,3))
# X_class[0,:,:,:2] = X_infer
# X_class[0,:,:,2] = Y_seg_pred

# classification_model = load_model(classification_model_path)

# Y_class_pred = classification_model.predict(X_class,batch_size=BS)



#%% diagnose plot 

