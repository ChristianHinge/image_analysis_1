# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:07:27 2021

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

print(device_lib.list_local_devices())

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%

model_path = "checkpoints/first low LR model/model_120.hdf5"

train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val_true, X_test, Y_test, X_train_test, Y_train_test = load_train_val_data(IDs)

print(val_IDs)
print(len(val_IDs))

#%%

BS = 2

#load model and make prediction 
Unet_model = load_model(model_path)

Y_val_pred = np.round(Unet_model.predict(X_val,batch_size=BS).squeeze())


#%%

#compute metrics with predictions 

def dice_coef(y_true, y_pred, smooth=1):
    
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    volume_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    if volume_sum!=0:
        smooth = 0
        
    return (2. * intersection + smooth) / (volume_sum + smooth)


def IoU_coef(y_true, y_pred,smooth=1):
    
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    if union!=0:
        smooth = 0

    return (intersection + smooth) / (union + smooth)


DSC2 = []
JSC = []

DSC = []
IoU = []

BS = 2

for b in range(1):#int(Y_val_true.shape[0]/BS)):
    
    bb = b*BS

    Y_val_true_batch = Y_val_true[bb:bb+BS,:,:]
    Y_val_pred_batch = Y_val_pred[bb:bb+BS,:,:]
    
    DSC_batch = dice_coef(tf.convert_to_tensor(Y_val_true_batch), tf.convert_to_tensor(Y_val_pred_batch))
    IoU_batch = IoU_coef(tf.convert_to_tensor(Y_val_true_batch), tf.convert_to_tensor(Y_val_pred_batch))
    
    DSC.append(DSC_batch)
    IoU.append(IoU_batch)
    
    
print(np.array(DSC).mean())
print(np.array(IoU).mean())


#%%
a = Y_val_true[20:40,:,:]
b = Y_val_pred[20:40,:,:]

IoU_batch = IoU_coef(tf.convert_to_tensor(a), tf.convert_to_tensor(b))
print(IoU_batch.numpy())

m = tf.keras.metrics.MeanIoU(num_classes=2)
m.update_state(a, b)
c=m.result().numpy()
print(c)



#%%
m = tf.keras.metrics.MeanIoU(num_classes=2)
m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
m.result().numpy()

m.reset_states()
m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
               sample_weight=[0.3, 0.3, 0.3, 0.1])
m.result().numpy()



#%%

# def dice_coef2(mask_gt,mask_pred, smooth=1):
#     volume_sum = mask_gt.sum() + mask_pred.sum()    
#     mask_gt_f = mask_gt.flatten()
#     mask_pred_f = mask_pred.flatten()
    
#     volume_intersect = (mask_gt_f * mask_pred_f).sum()
    
#     return (2*volume_intersect + smooth) / (volume_sum + smooth)



